import time
from math import log, floor, sqrt, exp
import random
import numpy as np

def ap_based_search(deadline, max_val_limit=2000, max_n_limit=30):
    """
    Model U as a set of disjoint arithmetic progressions with rapidly increasing
    common differences. Optimize parameters via stochastic coordinate ascent.
    """
    best_score = -1
    best_set = None
    
    # Try different numbers of APs
    for num_aps in range(2, 8):
        if time.time() > deadline - 10:
            break
        
        # Initialize with rapidly increasing differences
        # Start values and differences that grow geometrically
        base_diff = random.randint(2, 5)
        
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            # Generate initial AP parameters
            # Each AP: (start, diff, length)
            aps = []
            current_start = 1
            current_diff = base_diff * random.randint(1, 3)
            
            for i in range(num_aps):
                length = random.randint(2, max(2, (max_n_limit - len(aps)) // (num_aps - i)))
                length = min(length, max_n_limit - len(aps))
                if length < 2:
                    break
                aps.append([current_start, current_diff, length])
                # Next AP starts after the end of this one, with larger diff
                end_val = current_start + (length - 1) * current_diff
                current_start = end_val + random.randint(1, max(1, current_diff // 2))
                current_diff = current_diff * random.randint(2, 4)
            
            if len(aps) < 2:
                continue
            
            # Build initial set
            U = set([0])
            for start, diff, length in aps:
                for k in range(length):
                    val = start + k * diff
                    if 0 < val <= max_val_limit:
                        U.add(val)
            
            U = sorted(U)
            if len(U) < 2:
                continue
            
            current_score = score_set(U)
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
            
            # Stochastic coordinate ascent on AP parameters
            no_improvement = 0
            max_no_improvement = 50
            
            while time.time() < deadline - 10 and no_improvement < max_no_improvement:
                improved = False
                
                # Randomly pick an AP and a parameter to modify
                ap_idx = random.randint(0, len(aps) - 1)
                param_type = random.random()
                
                old_aps = [list(a) for a in aps]
                
                if param_type < 0.3:
                    # Modify start
                    delta = random.randint(-5, 5)
                    new_start = aps[ap_idx][0] + delta
                    if new_start > 0:
                        aps[ap_idx][0] = new_start
                elif param_type < 0.6:
                    # Modify difference
                    delta = random.randint(-2, 3)
                    new_diff = aps[ap_idx][1] + delta
                    if new_diff >= 2:
                        aps[ap_idx][1] = new_diff
                elif param_type < 0.8:
                    # Modify length
                    delta = random.randint(-1, 2)
                    new_length = aps[ap_idx][2] + delta
                    if new_length >= 2 and new_length <= max_n_limit:
                        aps[ap_idx][2] = new_length
                else:
                    # Add or remove an AP
                    if random.random() < 0.5 and len(aps) < 7:
                        # Add new AP
                        last_start = aps[-1][0] + (aps[-1][2] - 1) * aps[-1][1]
                        new_start = last_start + random.randint(1, max(1, aps[-1][1]))
                        new_diff = aps[-1][1] * random.randint(2, 4)
                        new_length = random.randint(2, 5)
                        aps.append([new_start, new_diff, new_length])
                    elif len(aps) > 2:
                        # Remove last AP
                        aps.pop()
                
                # Rebuild set from APs
                U_new = set([0])
                for start, diff, length in aps:
                    for k in range(length):
                        val = start + k * diff
                        if 0 < val <= max_val_limit:
                            U_new.add(val)
                
                U_new = sorted(U_new)
                if len(U_new) < 2 or len(U_new) > max_n_limit:
                    aps = old_aps
                    no_improvement += 1
                    continue
                
                new_score = score_set(U_new)
                if new_score == -1:
                    aps = old_aps
                    no_improvement += 1
                    continue
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(U_new)
                    no_improvement = 0
                else:
                    aps = old_aps
                    no_improvement += 1
            
            # After AP search, do a quick local polish on the best found
            if best_set and time.time() < deadline - 5:
                U = list(best_set)
                cur_s = score_set(U)
                for _ in range(20):
                    if time.time() > deadline - 5:
                        break
                    n = len(U)
                    # Try replacing one element
                    if n > 2:
                        idx = random.randint(1, n - 1)
                        old_val = U[idx]
                        new_val = old_val + random.randint(-3, 3)
                        if new_val > 0 and new_val not in U:
                            U_new = [U[j] if j != idx else new_val for j in range(n)]
                            U_new = sorted(set(U_new))
                            if len(U_new) == n:
                                s = score_set(U_new)
                                if s > cur_s:
                                    U = U_new
                                    cur_s = s
                                    if cur_s > best_score:
                                        best_score = cur_s
                                        best_set = list(U)
    
    return best_score, best_set

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            # Expand limit if we are growing
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try a few random candidates
                random.shuffle(candidates)
                for c in candidates[:10]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Pick a random candidate
                random.shuffle(candidates)
                for new_val in candidates[:10]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def simulated_annealing(U, deadline, max_iter=5000):
    """
    Simulated annealing with Ising-like energy model.
    Energy penalizes sumset collisions and rewards difference-set expansion.
    """
    U = sorted(set(U))
    n = len(U)
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    # Temperature schedule
    T_start = 0.1
    T_end = 0.001
    cooling_rate = (T_end / T_start) ** (1.0 / max(max_iter, 1))
    T = T_start
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Generate a random move: flip one element (add/remove/replace)
        move_type = random.random()
        new_U = list(U)
        
        if move_type < 0.4:
            # Add a random element
            if len(new_U) < 30:
                max_val = max(new_U) if new_U else 10
                new_val = random.randint(1, max_val * 2)
                if new_val not in new_U:
                    new_U.append(new_val)
                    new_U.sort()
                else:
                    continue
        elif move_type < 0.7:
            # Remove a random element (not 0)
            if len(new_U) > 2:
                removable = [x for x in new_U if x != 0]
                if removable:
                    idx = new_U.index(random.choice(removable))
                    new_U.pop(idx)
                else:
                    continue
        else:
            # Replace a random element
            if len(new_U) > 1:
                idx = random.randint(1, len(new_U) - 1)
                old_val = new_U[idx]
                max_val = max(new_U) if new_U else 10
                new_val = random.randint(1, max_val * 2)
                if new_val not in new_U:
                    new_U[idx] = new_val
                    new_U.sort()
                else:
                    continue
        
        if len(new_U) != len(U) and (len(new_U) < 2 or len(new_U) > 30):
            continue
            
        new_score = score_set(new_U)
        if new_score == -1:
            continue
            
        delta_score = new_score - current_score
        
        # Metropolis criterion
        if delta_score > 0 or random.random() < exp(delta_score / T):
            U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        
        # Cool down
        T *= cooling_rate
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates (from Inspiration)
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Geometric Search (from Current)
    # Quick coarse grid to find a good geometric seed
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 40:
            break
        for m in range(2, 21):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U

    # 3. Dynamic Local Search (from Inspiration)
    # Use the best found so far as the primary seed, and structured ones as secondary seeds
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Cycle through seeds
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        # Run local search
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds for future restarts? 
            # To keep it simple, we just update the primary seed list
            if len(seeds) < 5:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 4. AP-Based Structured Search
    if time.time() < deadline - 30:
        ap_score, ap_set = ap_based_search(deadline, max_val_limit=2000, max_n_limit=30)
        if ap_score > best_score:
            best_score = ap_score
            best_set = ap_set
    
    # 5. Simulated Annealing Phase
    if time.time() < deadline - 20:
        # Run SA on the current best and a few diverse seeds
        sa_seeds = [best_set]
        for cand in structured_candidates[:2]:
            if cand != best_set:
                sa_seeds.append(cand)
        
        for seed in sa_seeds:
            if time.time() > deadline - 10:
                break
            sa_score, sa_set = simulated_annealing(seed, deadline, max_iter=3000)
            if sa_score > best_score:
                best_score = sa_score
                best_set = sa_set
    
    # 6. Final Polish: Fine-grained local search on the best set
    # This addresses the "refinement machinery" requirement.
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 500:
            improved = False
            iteration += 1
            n = len(current)
            # Try replacing each element with nearby values (small delta)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Try deltas in a wider range but smaller steps
                for delta in range(-10, 11):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element near the max
            if not improved and n < 25 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max_val - 10, max_val + 11):
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

# Ensure 0 is in the set and it's a list of nonnegative integers
# The function construct returns a list.
# The evaluator calls construct().

if __name__ == "__main__":
    # Quick test
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
