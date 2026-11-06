import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin
    
    # Generate all C(16,4) = 1820 sets as 16-bit ints
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    n_candidates = len(all_sets)  # 1820
    
    # Precompute pairwise distances
    # dist[i][j] = popcount(all_sets[i] & all_sets[j])
    dist = [[0]*n_candidates for _ in range(n_candidates)]
    for i in range(n_candidates):
        ai = all_sets[i]
        for j in range(i+1, n_candidates):
            d = (ai & all_sets[j]).bit_count()
            dist[i][j] = d
            dist[j][i] = d
    
    def check_conflict(family_indices, new_idx):
        """Check if adding new_idx to family creates a sunflower triple."""
        fam = list(family_indices)
        k = new_idx
        n_fam = len(fam)
        for a in range(n_fam):
            for b in range(a+1, n_fam):
                i, j = fam[a], fam[b]
                dij = dist[i][j]
                dik = dist[i][k]
                djk = dist[j][k]
                if dij == dik == djk:
                    return True
        return False
    
    # Greedy: try to build a large family
    best_family = []
    
    # Try multiple random orderings
    random.seed(42)
    best_size = 0
    best_order = None
    
    for trial in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        order = list(range(n_candidates))
        random.shuffle(order)
        
        family = []
        for idx in order:
            if time.time() - start_time > DEADLINE:
                break
            if not check_conflict(family, idx):
                family.append(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Local search on best family
    if time.time() - start_time < DEADLINE - 5:
        # Try to improve via swaps
        in_family = set(best_family)
        for _ in range(50):
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            # Try removing one and adding another
            for remove_idx in best_family:
                if time.time() - start_time > DEADLINE:
                    break
                temp_fam = [x for x in best_family if x != remove_idx]
                for add_idx in range(n_candidates):
                    if add_idx in in_family and add_idx != remove_idx:
                        continue
                    if not check_conflict(temp_fam, add_idx):
                        # Check if this improves (it doesn't change size, but might allow further improvements)
                        # Actually swap doesn't help size. Let's just try adding.
                        pass
                # Try just adding
                if not check_conflict(best_family, add_idx):
                    best_family.append(add_idx)
                    in_family.add(add_idx)
                    improved = True
                    break
            if not improved:
                break
    
    # Convert back to sorted lists
    result = [sorted([i+1 for i in range(16) if (all_sets[s] >> i) & 1]) for s in best_family]
    
    return result
