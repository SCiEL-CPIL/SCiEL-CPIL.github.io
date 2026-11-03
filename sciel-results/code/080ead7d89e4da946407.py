import time
import random
from math import log
import numpy as np
from itertools import combinations

def fast_score(U):
    """
    Faster scoring using numpy for larger sets, falling back to python for small.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums
    # i <= j
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences
    # i < j
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    # Unique positive diffs
    num_pos_diffs = len(np.unique(diffs))
    # Total diffs = 2 * num_pos_diffs + 1 (for 0)
    num_diffs = 2 * num_pos_diffs + 1
    
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def geometric_set(base, m):
    """
    Construct a set U = {0} U { floor(base^k) for k=1..m }
    """
    U = [0]
    val = 1
    for _ in range(m):
        val = int(val * base)
        if val <= 0:
            break
        if val not in U:
            U.append(val)
    return sorted(set(U))

def structured_set(n):
    """
    Generate structured sets from additive combinatorics constructions.
    Uses modular arithmetic patterns that tend to have good sumset/difference-set properties.
    """
    candidates = []
    
    # Golomb-ruler-like: all pairwise differences distinct
    # Known optimal Golomb rulers for small n
    golomb_rulers = {
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 41],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67],
    }
    if n in golomb_rulers:
        candidates.append(golomb_rulers[n])
    
    # Power-of-prime-like constructions
    for p in [2, 3, 5, 7]:
        U = [0]
        val = 1
        for k in range(1, n):
            val *= p
            U.append(val)
        candidates.append(sorted(set(U)))
    
    # Mixed base constructions
    bases = [(2, 3), (3, 5), (2, 5), (3, 7)]
    for b1, b2 in bases:
        U = [0]
        for k in range(1, n):
            if k % 2 == 1:
                U.append(b1 ** k)
            else:
                U.append(b2 ** (k // 2))
        candidates.append(sorted(set(U)))
    
    return candidates

def golomb_ruler_construct(n, deadline):
    """
    Construct a Golomb ruler (all pairwise differences distinct) of length n.
    Uses greedy + local search. A Golomb ruler maximizes |U-U| = n(n-1)+1
    while keeping elements small, which helps the ratio.
    """
    if n < 2:
        return [0], fast_score([0, 1]) if n >= 2 else 0.0
    
    # Start with known optimal or near-optimal rulers
    known_rulers = {
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 41],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67],
        13: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67, 75],
    }
    
    if n in known_rulers:
        best = known_rulers[n][:]
        best_score = fast_score(best)
    else:
        # Greedy construction
        best = [0]
        for i in range(1, n):
            # Try to place next element as small as possible while maintaining Golomb property
            candidate = i * i  # start with quadratic growth
            while True:
                # Check if adding candidate maintains Golomb property
                diffs = set()
                valid = True
                for a, b in combinations(best + [candidate], 2):
                    d = abs(a - b)
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if valid:
                    best.append(candidate)
                    break
                candidate += 1
        best_score = fast_score(best)
    
    # Local search to improve the ruler
    current = best[:]
    current_score = best_score
    
    T = 0.05
    cooling = 0.99
    start_time = time.time()
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        n_cur = len(current)
        max_val = max(current) if current else 1
        
        # Move: perturb a non-zero element
        i = random.randint(1, n_cur - 1)
        scale = max(1, int(max_val * 0.05))
        new_val = current[i] + random.randint(-scale, scale)
        if new_val <= 0:
            new_val = current[i] + random.randint(1, scale)
        
        new_set = current[:]
        new_set[i] = new_val
        new_set = sorted(set(new_set))
        if len(new_set) != n_cur:
            continue
        
        new_score = fast_score(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = current[:]
        else:
            delta = current_score - new_score
            if delta < 0.3 and random.random() < np.exp(-delta / max(T, 0.001)):
                current = new_set
                current_score = new_score
        
        T *= cooling
        if T < 0.001:
            T = 0.001
    
    return best, best_score


def ternary_search_geometric(deadline):
    """
    Search over base b and length m for geometric-like sets.
    """
    best_score = -1
    best_set = [0, 1]
    
    # Range of bases: 1.1 to 5.0
    # Range of m: 3 to 30
    
    # Coarse search
    b_low, b_high = 1.1, 5.0
    m_low, m_high = 3, 30
    
    # We can't do full 2D ternary easily without many evaluations.
    # Let's iterate over m and ternary search b for each m.
    
    for m in range(m_low, m_high + 1):
        if time.time() > deadline - 10:
            break
        
        b_low, b_high = 1.1, 5.0
        # Ternary search for optimal base for this m
        for _ in range(10):  # 10 iterations of ternary search is ~30 evaluations
            if time.time() > deadline - 5:
                break
            b1 = b_low + (b_high - b_low) / 3
            b2 = b_high - (b_high - b_low) / 3
            
            U1 = geometric_set(b1, m)
            U2 = geometric_set(b2, m)
            
            s1 = fast_score(U1)
            s2 = fast_score(U2)
            
            if s1 > s2:
                b_high = b2
            else:
                b_low = b1
                
            if s1 > best_score:
                best_score = s1
                best_set = U1
            if s2 > best_score:
                best_score = s2
                best_set = U2
                
    return best_set, best_score

def simulated_annealing(U, deadline, max_iter=2000):
    """
    Simulated annealing to improve the set U.
    Moves:
    1. Perturb an element (local move).
    2. Add an element.
    3. Remove an element.
    4. Swap two elements.
    5. Scale entire set (change max element).
    6. Replace an element with a value from a candidate pool.
    7. Golomb-preserving move: adjust element to maintain distinct differences.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    
    T = 0.1  # Initial temperature
    cooling = 0.995
    
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        max_val = max(current)
        # Weighted move selection: perturb most common, then add, remove, swap, scale, replace, golomb
        r = random.random()
        if r < 0.30:
            move_type = 'perturb'
        elif r < 0.45:
            move_type = 'add'
        elif r < 0.55:
            move_type = 'remove'
        elif r < 0.65:
            move_type = 'swap'
        elif r < 0.75:
            move_type = 'scale'
        elif r < 0.88:
            move_type = 'replace'
        else:
            move_type = 'golomb'
        
        new_set = current[:]
        
        if move_type == 'perturb' and n > 1:
            i = random.randint(1, n-1)
            old_val = current[i]
            # Use multiple perturbation scales
            scale_choice = random.random()
            if scale_choice < 0.4:
                scale = max(1, int(max_val * 0.03))
            elif scale_choice < 0.7:
                scale = max(1, int(max_val * 0.1))
            else:
                scale = max(1, int(max_val * 0.25))
            new_val = old_val + random.randint(-scale, scale)
            if new_val <= 0:
                new_val = old_val + random.randint(1, scale)
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
        
        elif move_type == 'add' and n < 15:
            max_val = max(current)
            if random.random() < 0.3:
                new_val = max_val + random.randint(1, max(2, int(max_val * 0.15)))
            elif random.random() < 0.6:
                new_val = random.randint(1, max_val)
            else:
                new_val = random.randint(max_val + 1, max_val * 2)
            new_set.append(new_val)
            new_set = sorted(set(new_set))
            if len(new_set) == n:
                continue
        
        elif move_type == 'remove' and n > 4:
            i = random.randint(1, n-1)
            new_set.pop(i)
        
        elif move_type == 'swap' and n > 2:
            i, j = random.sample(range(1, n), 2)
            new_set[i], new_set[j] = new_set[j], new_set[i]
            new_set = sorted(new_set)
        
        elif move_type == 'scale' and n > 1:
            # Scale all elements except 0 by a small factor
            factor = random.choice([0.85, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3])
            new_set = [0] + [max(1, int(v * factor)) for v in current[1:]]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
        
        elif move_type == 'replace' and n > 1:
            # Replace a random element with a value from a structured candidate pool
            i = random.randint(1, n-1)
            # Generate candidates based on current set structure
            candidates_val = set()
            # Fractions of max_val
            for f in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                candidates_val.add(max(1, int(max_val * f)))
            # Values near existing elements
            for v in current[1:]:
                for d in [-3, -2, -1, 1, 2, 3]:
                    candidates_val.add(max(1, v + d))
            # Power-like values
            for p in [2, 3, 5]:
                val = 1
                for _ in range(min(15, max_val + 1)):
                    if val > max_val * 1.5:
                        break
                    candidates_val.add(val)
                    val *= p
            candidates_val.discard(0)
            candidates_val.discard(current[i])
            if not candidates_val:
                continue
            new_val = random.choice(list(candidates_val))
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
        
        elif move_type == 'golomb' and n > 2:
            # Golomb-preserving move: try to adjust an element to reduce sum collisions
            # while trying to keep differences distinct
            i = random.randint(1, n-1)
            old_val = current[i]
            
            # Compute current differences involving element i
            diffs_before = set()
            for j in range(n):
                if j != i:
                    diffs_before.add(abs(current[i] - current[j]))
            
            # Try values that might reduce sum collisions
            # Sums involving element i: current[i] + current[j] for j != i
            # We want to choose new_val such that new_val + current[j] doesn't collide with other sums
            
            # Compute all other sums (not involving i)
            other_sums = set()
            for j in range(n):
                for k in range(j+1, n):
                    if j != i and k != i:
                        other_sums.add(current[j] + current[k])
            
            # Try candidate values
            best_candidate = old_val
            best_candidate_score = -1
            
            # Generate candidates: values that avoid sum collisions
            candidates = set()
            for offset in range(-max(5, int(max_val * 0.1)), max(5, int(max_val * 0.1)) + 1):
                cv = old_val + offset
                if cv > 0:
                    candidates.add(cv)
            # Also try values based on fractions
            for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                candidates.add(max(1, int(max_val * f)))
            
            # Limit candidates for speed
            candidate_list = list(candidates)[:50]
            random.shuffle(candidate_list)
            
            for cv in candidate_list:
                if cv == old_val:
                    continue
                # Check if this value creates duplicate differences
                new_diffs = set()
                valid = True
                for j in range(n):
                    if j != i:
                        d = abs(cv - current[j])
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                if not valid:
                    continue
                
                # Check sum collisions
                new_sums = set()
                for j in range(n):
                    if j != i:
                        s = cv + current[j]
                        if s in other_sums:
                            # Collision! This is bad
                            continue
                        new_sums.add(s)
                
                # This candidate avoids sum collisions
                new_set_candidate = current[:]
                new_set_candidate[i] = cv
                new_set_candidate = sorted(set(new_set_candidate))
                if len(new_set_candidate) != n:
                    continue
                
                s = fast_score(new_set_candidate)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = cv
                    new_set = new_set_candidate
            
            if best_candidate == old_val:
                continue
        
        else:
            continue
            
        new_score = fast_score(new_set)
        
        if new_score < 0:
            if random.random() < T:
                current = new_set
                current_score = new_score
        else:
            if new_score > current_score:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current[:]
            else:
                delta = current_score - new_score
                if delta < 0.5 and random.random() < np.exp(-delta / max(T, 0.001)):
                    current = new_set
                    current_score = new_score
        
        T *= cooling
        if T < 0.001:
            T = 0.001
            
    return best, best_score

def high_precision_polish(U, deadline):
    """
    Final high-precision polish: try all single-element perturbations with adaptive steps,
    plus pair swaps and element replacement.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    
    n = len(current)
    max_val = max(current)
    
    # Adaptive step size based on set density
    step_range = max(3, min(15, int(max_val * 0.02)))
    
    # Phase 1: Single-element perturbations with adaptive range
    for i in range(1, n):
        if time.time() > deadline:
            break
        old_val = current[i]
        for offset in range(-step_range, step_range + 1):
            if offset == 0:
                continue
            new_val = old_val + offset
            if new_val <= 0:
                continue
            test_set = current[:]
            test_set[i] = new_val
            test_set = sorted(set(test_set))
            if len(test_set) != n:
                continue
            s = fast_score(test_set)
            if s > best_score:
                best_score = s
                best = test_set[:]
                current = test_set[:]
                old_val = new_val
    
    # Phase 2: Try replacing each element with a value in a wider range
    if time.time() < deadline - 5:
        for i in range(1, n):
            if time.time() > deadline:
                break
            old_val = current[i]
            # Try values at various fractions of max_val
            for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5]:
                new_val = max(1, int(max_val * frac))
                if new_val == old_val:
                    continue
                test_set = current[:]
                test_set[i] = new_val
                test_set = sorted(set(test_set))
                if len(test_set) != n:
                    continue
                s = fast_score(test_set)
                if s > best_score:
                    best_score = s
                    best = test_set[:]
                    current = test_set[:]
    
    # Phase 3: Pair swaps
    if time.time() < deadline - 2 and n > 2:
        for i in range(1, n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                test_set = current[:]
                test_set[i], test_set[j] = test_set[j], test_set[i]
                test_set = sorted(test_set)
                s = fast_score(test_set)
                if s > best_score:
                    best_score = s
                    best = test_set[:]
                    current = test_set[:]
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: evaluator kills at 150s, return well before
    
    best_set = [0, 1, 4, 9, 11]
    best_score = fast_score(best_set)
    
    # 1. Geometric Search
    geo_set, geo_score = ternary_search_geometric(deadline)
    if geo_score > best_score:
        best_score = geo_score
        best_set = geo_set[:]
        
    # 2. Golomb ruler construction for various sizes
    for n in range(4, 14):
        if time.time() > deadline - 30:
            break
        gr_set, gr_score = golomb_ruler_construct(n, time.time() + 5)
        if gr_score > best_score:
            best_score = gr_score
            best_set = gr_set[:]
    
    # 3. Add known good candidates (Golomb Rulers, structured sets, etc.)
    # Focus on small sets (n=4,5,6) which are in the sweet spot per cell findings
    candidates = [
        # Optimal Golomb rulers for small n
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 41],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        geo_set[:]
    ]
    
    # Add structured sets from additive combinatorics
    for n in range(4, 11):
        candidates.extend(structured_set(n))
    
    # Generate some random sparse sets with diverse parameters
    random.seed(42)
    for _ in range(20):
        n_elem = random.randint(4, 10)
        max_val = random.randint(20, 200)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
        
    # Geometric sets with specific ratios
    for ratio in [1.5, 1.8, 2.0, 2.5, 3.0]:
        U = [0]
        val = 1
        for _ in range(10):
            U.append(val)
            val = int(val * ratio)
            if val > 3000:
                break
        candidates.append(U)
    
    # Add some hand-crafted small sets that might have good ratio
    handcrafted = [
        [0, 1, 3, 7],
        [0, 1, 5, 8],
        [0, 2, 5, 9],
        [0, 1, 4, 8, 13],
        [0, 1, 5, 9, 14],
        [0, 2, 7, 12, 18],
        [0, 1, 4, 10, 16],
        [0, 3, 7, 12, 18],
        [0, 1, 6, 11, 17],
        [0, 2, 8, 14, 20],
        # Additional Golomb-like sets
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67, 75],
        [0, 2, 5, 9, 14, 20, 27, 35, 44, 54],
        [0, 1, 5, 10, 16, 23, 31, 40, 50, 61],
    ]
    candidates.extend(handcrafted)
    
    # Evaluate candidates
    scored_candidates = []
    for U in candidates:
        U_clean = sorted(set(U))
        if 0 not in U_clean or len(U_clean) < 2:
            continue
        s = fast_score(U_clean)
        if s > 0:
            scored_candidates.append((s, U_clean))
            
    scored_candidates.sort(reverse=True)
    
    # 4. Run SA on top candidates with more diverse starts
    sa_budget = deadline - 35  # Leave 35s for polish and re-evaluation
    # Run SA on top 8 candidates
    for idx, (score, U) in enumerate(scored_candidates[:8]):
        if time.time() > sa_budget:
            break
        # Allocate time based on rank
        remaining = sa_budget - time.time()
        if idx < 3:
            alloc = max(8, min(remaining, 20))
        else:
            alloc = max(5, min(remaining, 12))
        refined, refined_score = simulated_annealing(U, time.time() + alloc)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
    
    # 5. Also run SA on the global best set itself for additional refinement
    if time.time() < deadline - 30:
        remaining = deadline - 30 - time.time()
        if remaining > 5:
            refined, refined_score = simulated_annealing(best_set, time.time() + min(remaining, 15))
            if refined_score > best_score:
                best_score = refined_score
                best_set = refined[:]
    
    # 6. Final high-precision polish on the best set
    if time.time() < deadline - 10:
        refined, refined_score = high_precision_polish(best_set, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
    
    # 7. One more SA round on the polished result if time allows
    if time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining > 3:
            refined, refined_score = simulated_annealing(best_set, time.time() + min(remaining, 10))
            if refined_score > best_score:
                best_score = refined_score
                best_set = refined[:]
    
    return best_set

# Test
if __name__ == "__main__":
    print(f"Best set: {construct()}")
