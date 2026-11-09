"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import log
from itertools import combinations

def _compute_score(U):
    """Compute the score for a given set U using NumPy for speed."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    if len(U_arr) == 0:
        return 0.0
    if 0 not in U_arr:
        return 0.0
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = U_arr[-1]
    if max_u == 0:
        return 1.0
    
    # Compute U+U and U-U
    # For small n, this is very fast
    sumset = np.unique(U_arr[:, None] + U_arr[None, :])
    diffset = np.unique(U_arr[:, None] - U_arr[None, :])
    
    len_sum = len(sumset)
    len_diff = len(diffset)
    
    if len_sum == 0:
        return 0.0
        
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
    
    score = 1 + log(ratio) / denominator
    return score

def _exhaustive_small_search(max_val=30, max_size=6):
    """Find the best small sets by exhaustive search."""
    best_set = [0, 1, 2, 5]
    best_score = _compute_score(best_set)
    
    range_val = range(max_val + 1)
    
    for size in range(3, max_size + 1):
        for comb in combinations(range_val, size - 1):
            # Always include 0
            candidate = sorted([0] + list(comb))
            # Skip if 0 is already in comb (it won't be since range starts at 0 but we exclude 0 from comb generation implicitly by checking)
            # Actually combinations(range_val, size-1) includes 0. We need to ensure 0 is in the set.
            # If 0 is not in comb, we add it. If 0 is in comb, we have a duplicate.
            # Let's just generate combinations of 1..max_val and add 0.
            pass # The above loop is inefficient. Let's do it properly.
    
    # Proper exhaustive search
    candidates = list(range(1, max_val + 1))
    
    for size in range(3, max_size + 1):
        for comb in combinations(candidates, size - 1):
            candidate = sorted([0] + list(comb))
            score = _compute_score(candidate)
            if score > best_score:
                best_score = score
                best_set = candidate
                
    return best_set, best_score

def _local_search(U, max_iterations=50000, timeout=10.0):
    """Local search to improve the set."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        U_set = set(current)
        max_u = max(U_set) if U_set else 0
        n = len(U_set)
        
        # Generate candidates for addition
        candidates = set()
        
        # 1. Small perturbations
        for x in list(U_set):
            for offset in [-3, -2, -1, 1, 2, 3, 4, 5]:
                c = x + offset
                if c >= 0:
                    candidates.add(c)
        
        # 2. Structure-preserving: sums and diffs of pairs
        # Limit to first 20 elements to keep it fast
        elems = list(U_set)
        if n > 20:
            elems = elems[:20]
        for idx_a in range(len(elems)):
            for idx_b in range(len(elems)):
                s = elems[idx_a] + elems[idx_b]
                if s <= 2 * max_u + 100:
                    candidates.add(s)
                d = abs(elems[idx_a] - elems[idx_b])
                if d > 0 and d <= 2 * max_u + 100:
                    candidates.add(d)
        
        # 3. Powers
        p = 1
        while p <= 2 * max_u + 100:
            candidates.add(p)
            p *= 2
        p = 1
        while p <= 2 * max_u + 100:
            candidates.add(p)
            p *= 3
            
        # 4. Random candidates in range
        for _ in range(20):
            candidates.add(random.randint(0, 2 * max_u + 100))
            
        # 5. Specific "gap" candidates
        if max_u > 0:
            candidates.add(max_u + 1)
            candidates.add(max_u + 2)
            candidates.add(max_u * 2)
            candidates.add(max_u * 2 + 1)
            
        candidate_list = list(candidates)
        random.shuffle(candidate_list)
        
        improved = False
        for cand in candidate_list:
            if time.time() - start_time > timeout:
                break
            if cand in U_set:
                continue
            new_set = U_set | {cand}
            new_score = _compute_score(new_set)
            if new_score > current_score + 1e-9:
                current = sorted(new_set)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                improved = True
                break
        
        if not improved:
            # Try removing an element
            removable = [x for x in current if x != 0]
            random.shuffle(removable)
            for cand in removable[:30]:
                if time.time() - start_time > timeout:
                    break
                new_set = U_set - {cand}
                if len(new_set) < 2:
                    continue
                new_score = _compute_score(new_set)
                if new_score > current_score + 1e-9:
                    current = sorted(new_set)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
                    improved = True
                    break
        
        if not improved and i > 50:
            # Try a swap: remove one, add another
            if random.random() < 0.5:
                removable = [x for x in current if x != 0]
                if removable:
                    rem = random.choice(removable)
                    # Try adding a random element in a wider range
                    new_elem = random.randint(0, 2 * max_u + 200)
                    new_set = (U_set - {rem}) | {new_elem}
                    if len(new_set) >= 2:
                        new_score = _compute_score(new_set)
                        if new_score > current_score + 1e-9:
                            current = sorted(new_set)
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best = list(current)
    
    return best, best_score

def _simulated_annealing(U, timeout=15.0):
    """Simulated annealing to escape local optima."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    T = 0.5  # Initial temperature
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        T *= cooling
        if T < 1e-6:
            break
        
        U_set = set(current)
        max_u = max(U_set) if U_set else 0
        n = len(U_set)
        
        # Generate a random perturbation
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a random element
            new_elem = random.randint(0, 2 * max_u + 200)
            if new_elem in U_set:
                continue
            new_set = U_set | {new_elem}
        elif move_type < 0.7:
            # Remove a random element (not 0)
            removable = [x for x in current if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            new_set = U_set - {rem}
            if len(new_set) < 2:
                continue
        elif move_type < 0.85:
            # Shift a random element
            removable = [x for x in current if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            shift = random.randint(-10, 10)
            new_elem = rem + shift
            if new_elem < 0 or new_elem in U_set:
                continue
            new_set = (U_set - {rem}) | {new_elem}
        else:
            # Swap two elements
            removable = [x for x in current if x != 0]
            if len(removable) < 2:
                continue
            rem1, rem2 = random.sample(removable, 2)
            new_elem1 = rem1 + random.randint(-5, 5)
            new_elem2 = rem2 + random.randint(-5, 5)
            if new_elem1 < 0 or new_elem2 < 0:
                continue
            if new_elem1 in U_set or new_elem2 in U_set:
                continue
            new_set = (U_set - {rem1, rem2}) | {new_elem1, new_elem2}
        
        if len(new_set) < 2:
            continue
            
        new_score = _compute_score(new_set)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta > 0 or random.random() < np.exp(-delta / max(T, 1e-9)):
            current = sorted(new_set)
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        
        # Occasionally do a local search step
        if random.random() < 0.1:
            local_best, local_score = _local_search(current, max_iterations=1000, timeout=1.0)
            if local_score > current_score:
                current = local_best
                current_score = local_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Start with known good small sets
    best_result = [0, 1, 2, 4]
    best_score = _compute_score(best_result)
    
    # 1. Exhaustive search for small sets
    # This is fast for small max_val and max_size
    if time.time() < deadline - 5:
        ex_set, ex_score = _exhaustive_small_search(max_val=30, max_size=6)
        if ex_score > best_score:
            best_score = ex_score
            best_result = ex_set
        
        # Try slightly larger exhaustive search if time permits
        if time.time() < deadline - 10:
            ex_set2, ex_score2 = _exhaustive_small_search(max_val=50, max_size=5)
            if ex_score2 > best_score:
                best_score = ex_score2
                best_result = ex_set2

    # Generate diverse initial sets
    initial_sets = []
    
    # 1. Powers of 2
    p2 = [0]
    v = 1
    while v <= 5000:
        p2.append(v)
        v *= 2
    initial_sets.append(p2)
    
    # 2. Powers of 3
    p3 = [0]
    v = 1
    while v <= 5000:
        p3.append(v)
        v *= 3
    initial_sets.append(p3)
    
    # 3. Mixed powers
    mixed = [0, 1]
    v = 3
    while v <= 5000:
        mixed.append(v)
        v *= 2
    initial_sets.append(mixed)
    
    # 4. Quadratic residues modulo a prime (often good difference sets)
    for p in [101, 103, 107, 109]:
        qr = [0]
        for i in range(1, p):
            val = (i * i) % p
            if val not in qr:
                qr.append(val)
        initial_sets.append(sorted(set(qr)))
        
    # 5. Random sparse sets
    for _ in range(5):
        n = random.randint(10, 50)
        max_val = random.randint(100, 1000)
        rand_set = {0}
        while len(rand_set) < n:
            rand_set.add(random.randint(0, max_val))
        initial_sets.append(sorted(rand_set))
        
    # 6. Dense low, sparse high
    dense_sparse = list(range(10))
    v = 20
    while v <= 1000:
        dense_sparse.append(v)
        v *= 2
    initial_sets.append(sorted(set(dense_sparse)))
    
    # 7. Fibonacci
    fib = [0, 1]
    while fib[-1] <= 5000:
        fib.append(fib[-1] + fib[-2])
    initial_sets.append(fib)
    
    # 8. Add the best exhaustive result
    initial_sets.append(best_result)
    
    # Sort initial sets by score to try best ones first
    initial_sets.sort(key=lambda s: _compute_score(s), reverse=True)
    
    # Run local search on each initial set
    for init_set in initial_sets:
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 2
        if remaining < 2:
            break
            
        # Allocate time based on potential
        alloc_time = min(8.0, remaining * 0.8)
        
        improved, score = _local_search(init_set, max_iterations=50000, timeout=alloc_time)
        
        if score > best_score:
            best_score = score
            best_result = improved
    
    # Run simulated annealing from the best result found so far
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 5:
            sa_result, sa_score = _simulated_annealing(best_result, timeout=remaining)
            if sa_score > best_score:
                best_score = sa_score
                best_result = sa_result
    
    # Final local search polish
    if time.time() < deadline - 1:
        remaining = deadline - time.time() - 0.5
        if remaining > 1:
            final_result, final_score = _local_search(best_result, max_iterations=20000, timeout=remaining)
            if final_score > best_score:
                best_score = final_score
                best_result = final_result
    
    # Final check
    final_score = _compute_score(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 3]
    
    return best_result
# EVOLVE-BLOCK-END
