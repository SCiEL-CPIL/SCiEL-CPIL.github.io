# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 16
    k = 4
    max_target = 250

    # Generate all 4-subsets as bitmasks (0-indexed internally)
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)

    total = len(all_sets)  # C(16,4) = 1820

    # Precompute pairwise intersections for all pairs
    # inter[i][j] = all_sets[i] & all_sets[j]
    # This is a 1820x1820 matrix, ~3.3M entries - feasible
    # But we'll use a more memory-efficient approach

    # Strategy: Use a greedy algorithm with smart ordering
    # Key insight: to avoid sunflowers, we need no three sets where
    # all three pairwise intersections are equal.
    
    # Precompute intersection masks for each set against all others
    # Actually, let's use a different approach: for each set, track which
    # intersection sizes/values it has with existing family members
    
    best_family = []

    def try_greedy(order, max_time):
        """Try greedy addition in given order, return family."""
        F = []
        # For each set in F, track intersection -> list of indices
        inter_maps = []
        
        for idx in order:
            if time.time() > max_time:
                break
            m = all_sets[idx]
            conflict = False

            for i in range(len(F)):
                s = m & F[i]
                lst = inter_maps[i].get(s)
                if lst:
                    for j in lst:
                        if (m & F[j]) == s:
                            conflict = True
                            break
                if conflict:
                    break

            if not conflict:
                idx_new = len(F)
                F.append(m)
                inter_maps.append({})
                for i in range(idx_new):
                    s = m & F[i]
                    if s not in inter_maps[i]:
                        inter_maps[i][s] = []
                    inter_maps[i][s].append(idx_new)
                    if s not in inter_maps[idx_new]:
                        inter_maps[idx_new][s] = []
                    inter_maps[idx_new][s].append(i)

                if len(F) >= max_target:
                    break

        return F

    def local_search(F, max_time):
        """Try to improve family by swapping out conflicting sets."""
        if not F:
            return F
        F = F[:]
        # Build inter_maps
        inter_maps = []
        for i in range(len(F)):
            im = {}
            for j in range(i):
                s = F[i] & F[j]
                if s not in im:
                    im[s] = []
                im[s].append(j)
            inter_maps.append(im)

        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try to add any set not in F
            in_F = set(F)
            for idx in range(total):
                if time.time() > max_time:
                    return F
                m = all_sets[idx]
                if m in in_F:
                    continue
                conflict = False
                for i in range(len(F)):
                    s = m & F[i]
                    lst = inter_maps[i].get(s)
                    if lst:
                        for j in lst:
                            if (m & F[j]) == s:
                                conflict = True
                                break
                    if conflict:
                        break
                if not conflict:
                    # Add it
                    idx_new = len(F)
                    F.append(m)
                    in_F.add(m)
                    inter_maps.append({})
                    for i in range(idx_new):
                        s = m & F[i]
                        if s not in inter_maps[i]:
                            inter_maps[i][s] = []
                        inter_maps[i][s].append(idx_new)
                        if s not in inter_maps[idx_new]:
                            inter_maps[idx_new][s] = []
                        inter_maps[idx_new][s].append(i)
                    improved = True
                    if len(F) >= max_target:
                        return F
        return F

    # Strategy 1: Deterministic greedy (original order)
    order1 = list(range(total))
    F1 = try_greedy(order1, deadline)
    best_family = F1[:]

    # Strategy 2: Greedy with random shuffles
    rng = random.Random(42)
    num_restarts = 500
    for run in range(num_restarts):
        if time.time() > deadline:
            break
        order = list(range(total))
        rng.shuffle(order)
        F = try_greedy(order, deadline)
        if len(F) > len(best_family):
            best_family = F[:]
            if len(best_family) >= max_target:
                break

    # Strategy 3: Greedy with priority based on "diversity"
    # Sort sets by how many times they appear in small intersections
    # Actually, let's try a weighted ordering: prefer sets that are "far" from existing ones
    
    # Strategy 4: Multiple passes with different random seeds
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline:
            break
        rng2 = random.Random(seed)
        for run in range(100):
            if time.time() > deadline:
                break
            order = list(range(total))
            rng2.shuffle(order)
            F = try_greedy(order, deadline)
            if len(F) > len(best_family):
                best_family = F[:]
                if len(best_family) >= max_target:
                    break
        if len(best_family) >= max_target:
            break

    # Strategy 5: Local search on the best family found
    if len(best_family) < max_target and time.time() < deadline:
        improved_F = local_search(best_family, deadline)
        if len(improved_F) > len(best_family):
            best_family = improved_F

    # Convert best_family masks to sorted 1-indexed lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
# EVOLVE-BLOCK-END
