import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    n = 16
    k = 4
    all_sets = list(combinations(range(1, n+1), k))
    N = len(all_sets)
    
    # Represent each set as a 16-bit integer
    all_bits = [0] * N
    for i, s in enumerate(all_sets):
        mask = 0
        for elem in s:
            mask |= (1 << (elem - 1))
        all_bits[i] = mask
    
    # Precompute popcount
    popcount = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Precompute pairwise intersection sizes: inter[i][j] = |set_i ∩ set_j|
    # N=1820, so this is 1820*1820 = ~3.3M entries, fits in memory
    inter = [[0] * N for _ in range(N)]
    for i in range(N):
        bi = all_bits[i]
        for j in range(i+1, N):
            v = popcount(bi & all_bits[j])
            inter[i][j] = v
            inter[j][i] = v
    
    def check_conflict(new_idx, family_list):
        """Fast conflict check using precomputed intersections."""
        # For each pair (A,B) in family, check if |S∩A| = |S∩B| = |A∩B|
        # Group by |S∩A| value
        groups = {}
        for A_idx in family_list:
            sA = popcount(all_bits[new_idx] & all_bits[A_idx])
            groups.setdefault(sA, []).append(A_idx)
        
        for size, group in groups.items():
            if len(group) < 2:
                continue
            glen = len(group)
            for i in range(glen):
                gi = group[i]
                for j in range(i+1, glen):
                    gj = group[j]
                    if inter[gi][gj] == size:
                        return True, (gi, gj)
        return False, None
    
    random.seed(12345)
    
    best_family = []
    best_size = 0
    
    # Phase 1: Multiple greedy starts
    max_greedy_trials = 50
    for trial in range(max_greedy_trials):
        if time.time() - start_time > DEADLINE:
            break
        
        family = []
        shuffled = list(range(N))
        random.shuffle(shuffled)
        
        for idx in shuffled:
            if time.time() - start_time > DEADLINE:
                break
            conflict, _ = check_conflict(idx, family)
            if not conflict:
                family.append(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
    
    # Phase 2: Local search with simulated annealing
    family = list(best_family)
    family_set = set(family)
    
    T = 3.0
    T_min = 0.005
    cooling = 0.9995
    max_sa_iters = 100000
    sa_iter = 0
    
    while sa_iter < max_sa_iters:
        if time.time() - start_time > DEADLINE:
            break
        sa_iter += 1
        
        action = random.random()
        
        if action < 0.6:
            # Try to add a random set
            idx = random.randint(0, N-1)
            if idx in family_set:
                continue
            conflict, conflicting_pair = check_conflict(idx, family)
            if not conflict:
                family.append(idx)
                family_set.add(idx)
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
            else:
                # Try swap
                A, B = conflicting_pair
                for remove_idx in [A, B]:
                    if time.time() - start_time > DEADLINE:
                        break
                    new_family = [x for x in family if x != remove_idx]
                    conflict2, _ = check_conflict(idx, new_family)
                    if not conflict2:
                        family = new_family
                        family_set = set(family)
                        family.append(idx)
                        family_set.add(idx)
                        if len(family) > best_size:
                            best_size = len(family)
                            best_family = list(family)
                        break
        elif action < 0.85:
            # Swap: remove random, add random
            if not family:
                continue
            remove_idx = random.choice(family)
            add_idx = random.randint(0, N-1)
            if add_idx in family_set:
                continue
            new_family = [x for x in family if x != remove_idx]
            conflict, _ = check_conflict(add_idx, new_family)
            if not conflict:
                family = new_family
                family_set = set(family)
                family.append(add_idx)
                family_set.add(add_idx)
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
        else:
            # Perturbation: remove several random sets
            if len(family) > 25:
                num_remove = random.randint(3, 5)
                for _ in range(num_remove):
                    if family:
                        rem_idx = random.choice(family)
                        family.remove(rem_idx)
                        family_set.discard(rem_idx)
        
        T = max(T_min, T * cooling)
    
    return [sorted(all_sets[i]) for i in best_family]
