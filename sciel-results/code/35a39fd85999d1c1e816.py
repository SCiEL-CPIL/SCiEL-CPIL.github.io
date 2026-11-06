import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    # Generate all C(16,4) = 1820 sets as 16-bit ints
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    n_candidates = len(all_sets)  # 1820
    
    # Precompute pairwise intersection masks
    # inter[i][j] = all_sets[i] & all_sets[j]
    inter = [[0]*n_candidates for _ in range(n_candidates)]
    for i in range(n_candidates):
        ai = all_sets[i]
        for j in range(i+1, n_candidates):
            d = ai & all_sets[j]
            inter[i][j] = d
            inter[j][i] = d

    def greedy(order):
        chosen_indices = []
        # To speed up conflict check, we can keep the list of chosen indices
        # and iterate through pairs.
        for idx in order:
            # Check if idx conflicts with any pair in chosen_indices
            is_conflict = False
            m = len(chosen_indices)
            # Iterate over all pairs in chosen_indices
            for a_idx in range(m):
                a = chosen_indices[a_idx]
                inter_a_idx = inter[a][idx]
                for b_idx in range(a_idx + 1, m):
                    b = chosen_indices[b_idx]
                    # Check if inter[a][b] == inter[a][idx] == inter[b][idx]
                    if inter[a][b] == inter_a_idx and inter[b][idx] == inter_a_idx:
                        is_conflict = True
                        break
                if is_conflict:
                    break
            
            if not is_conflict:
                chosen_indices.append(idx)
                
            # Time check inside loop to prevent timeout
            if time.time() - start_time > DEADLINE:
                break
                
        return chosen_indices

    best_family = []
    
    # 1. Natural Order
    order_nat = list(range(n_candidates))
    fam = greedy(order_nat)
    if len(fam) > len(best_family):
        best_family = fam
    
    # 2. Reversed Order
    if time.time() - start_time < DEADLINE - 2:
        order_rev = list(range(n_candidates - 1, -1, -1))
        fam = greedy(order_rev)
        if len(fam) > len(best_family):
            best_family = fam

    # 3. Random Orders
    rng = random.Random(42)
    # Determine how many random restarts we can afford
    # Estimate time per restart based on first two
    elapsed = time.time() - start_time
    if elapsed > 0:
        time_per_restart = elapsed / 2
        max_restarts = int((DEADLINE - time.time()) / max(time_per_restart, 0.1))
        max_restarts = min(max_restarts, 50) # Cap at 50
    else:
        max_restarts = 20

    for _ in range(max_restarts):
        if time.time() - start_time > DEADLINE - 1:
            break
        order = list(range(n_candidates))
        rng.shuffle(order)
        fam = greedy(order)
        if len(fam) > len(best_family):
            best_family = fam

    # Convert back to sorted lists
    result = [sorted([i+1 for i in range(16) if (all_sets[s] >> i) & 1]) for s in best_family]
    
    return result
