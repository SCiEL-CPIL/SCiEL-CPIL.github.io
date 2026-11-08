import time
from math import log
import random
from itertools import combinations

def score_set(U):
    """Compute the score for a set U."""
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

def exhaustive_search(n, max_val, deadline):
    """Find best set of size n (including 0) with elements <= max_val."""
    best_s = -1
    best_c = None
    range_vals = range(1, max_val + 1)
    for comb in combinations(range_vals, n - 1):
        if time.time() > deadline - 5:
            break
        U = [0] + list(comb)
        s = score_set(U)
        if s > best_s:
            best_s = s
            best_c = U
    return best_s, best_c

def simulated_annealing(U, deadline, T_init=1.0, cooling=0.9995):
    """Simulated annealing local search."""
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    T = T_init
    
    while time.time() < deadline - 3:
        if T < 0.01:
            break
            
        # Choose a move
        move_type = random.random()
        new_U = list(U)
        
        if move_type < 0.5:
            # Replace an element
            if len(new_U) < 2:
                continue
            idx = random.randint(1, len(new_U) - 1)
            old_val = new_U[idx]
            if random.random() < 0.7:
                delta = random.randint(-15, 15)
                new_val = old_val + delta
            else:
                new_val = random.randint(1, max(20, max(new_U) * 2))
            
            if new_val <= 0 or new_val in U:
                continue
            new_U[idx] = new_val
            new_U = sorted(set(new_U))
            
        elif move_type < 0.8:
            # Add an element
            if len(new_U) < 40:
                if random.random() < 0.5:
                    new_val = random.randint(max(1, max(new_U) - 10), max(new_U) + 30)
                else:
                    new_val = random.randint(1, max(20, max(new_U)))
                
                if new_val in U:
                    continue
                new_U.append(new_val)
                new_U = sorted(set(new_U))
                
        else:
            # Remove an element
            if len(new_U) > 2:
                idx = random.randint(1, len(new_U) - 1)
                new_U.pop(idx)
                new_U = sorted(set(new_U))
        
        if len(new_U) < 2:
            continue
            
        new_score = score_set(new_U)
        
        # Acceptance criterion
        if new_score >= current_score:
            U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        else:
            delta = new_score - current_score
            if random.random() < pow(2.718281828, delta / T):
                U = new_U
                current_score = new_score
        
        T *= cooling
        
    return best_score, best_set

def collision_aware_search(U, deadline):
    """Local search that tracks sum/diff collisions to guide moves."""
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        if no_improve > max_no_improve:
            break
            
        improved = False
        move_type = random.random()
        
        if move_type < 0.4:
            # Replace: try values that might reduce sum collisions
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                
                # Generate candidates near old_val and at strategic positions
                candidates = set()
                for delta in range(-20, 21):
                    v = old_val + delta
                    if v > 0 and v not in U:
                        candidates.add(v)
                # Add some random candidates
                for _ in range(10):
                    v = random.randint(1, max(30, max(U) * 2))
                    if v not in U:
                        candidates.add(v)
                
                for new_val in list(candidates)[:15]:
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U = sorted(set(new_U))
                    if len(new_U) != len(U):
                        continue
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
                        
        elif move_type < 0.7:
            # Add: try values that spread out differences
            if len(U) < 35:
                # Try values at gaps in the current set
                gaps = []
                for i in range(len(U) - 1):
                    mid = (U[i] + U[i+1]) // 2
                    if mid > 0 and mid not in U:
                        gaps.append(mid)
                # Also try near max
                gaps.extend([max(U) + d for d in range(1, 15) if max(U) + d not in U])
                
                for new_val in gaps[:10]:
                    if new_val <= 0:
                        continue
                    new_U = U + [new_val]
                    new_U = sorted(set(new_U))
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
                        
        else:
            # Remove: try removing elements that cause many sum collisions
            if len(U) > 3:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
        
        if improved:
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        else:
            no_improve += 1
            
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_set = [0, 1]
    best_score = score_set(best_set)
    
    # Phase 1: Structured seeds
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        [0, 1, 3, 7, 15, 31, 63, 127, 255],
        [0, 1, 3, 4, 9, 10, 12, 13, 28, 29, 31, 32],
        [0, 1, 4, 10, 20, 35, 56, 84, 120],
        [0, 1, 3, 9, 27, 81, 243, 729],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 3, 6, 10, 15, 21, 28],
        [0, 1, 2, 5, 7, 10, 13, 17, 21],
        [0, 1, 4, 9, 11],
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Phase 2: Exhaustive search for small n (key structural improvement)
    exhaustive_seeds = []
    
    if time.time() < deadline - 80:
        s, c = exhaustive_search(3, 60, deadline)
        if s > best_score:
            best_score = s
            best_set = c
        exhaustive_seeds.append(c)
        
    if time.time() < deadline - 50:
        s, c = exhaustive_search(4, 40, deadline)
        if s > best_score:
            best_score = s
            best_set = c
        exhaustive_seeds.append(c)
        
    if time.time() < deadline - 25:
        s, c = exhaustive_search(5, 25, deadline)
        if s > best_score:
            best_score = s
            best_set = c
        exhaustive_seeds.append(c)
    
    # Phase 3: Refinement using multiple search methods
    # Use best set and exhaustive seeds as starting points
    search_seeds = [best_set]
    for seed in exhaustive_seeds:
        if seed and seed != best_set:
            search_seeds.append(seed)
    
    # Add random perturbations
    for _ in range(5):
        pert = best_set[:]
        if len(pert) > 1:
            idx = random.randint(1, len(pert) - 1)
            pert[idx] += random.randint(-8, 8)
            if pert[idx] > 0:
                pert = sorted(set(pert))
                if len(pert) >= 2 and 0 in pert:
                    search_seeds.append(pert)
    
    # Run simulated annealing on each seed
    for seed in search_seeds:
        if time.time() > deadline - 15:
            break
        sa_score, sa_set = simulated_annealing(seed, deadline)
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_set
    
    # Run collision-aware search on the best result
    if time.time() < deadline - 10:
        ca_score, ca_set = collision_aware_search(best_set, deadline)
        if ca_score > best_score:
            best_score = ca_score
            best_set = ca_set
    
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
