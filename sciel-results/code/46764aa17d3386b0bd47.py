import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    rng = random.Random(42)

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}
    
    # For fast lookup of triplet index
    triplet_idx = [[[0]*v for _ in range(v)] for _ in range(v)]
    for i, s in enumerate(all_subsets):
        a, b, c = s
        triplet_idx[a][b][c] = i
        triplet_idx[a][c][b] = i
        triplet_idx[b][a][c] = i
        triplet_idx[b][c][a] = i
        triplet_idx[c][a][b] = i
        triplet_idx[c][b][a] = i

    covered_mask = 0
    cover_count = [0] * n_subsets
    blocks = []
    
    def get_gain(x, B):
        gain = 0
        B_list = list(B)
        for i in range(len(B_list)):
            u = B_list[i]
            for j in range(i+1, len(B_list)):
                v2 = B_list[j]
                idx = triplet_idx[x][u][v2]
                if not (covered_mask >> idx) & 1:
                    gain += 1
        return gain

    def is_covered():
        return covered_mask == (1 << n_subsets) - 1

    def add_block(bl):
        nonlocal covered_mask
        bl_set = set(bl)
        for s in itertools.combinations(bl, t):
            idx = subset_to_index[s]
            if not (covered_mask >> idx) & 1:
                covered_mask |= 1 << idx
            cover_count[idx] += 1

    def remove_block(bl):
        nonlocal covered_mask
        bl_set = set(bl)
        for s in itertools.combinations(bl, t):
            idx = subset_to_index[s]
            cover_count[idx] -= 1
            if cover_count[idx] == 0:
                covered_mask &= ~(1 << idx)

    # Greedy construction
    while True:
        if time.time() > deadline:
            break
        if is_covered():
            break
        
        seed_idx = -1
        for i in range(n_subsets):
            if cover_count[i] == 0:
                seed_idx = i
                break
        
        if seed_idx == -1:
            break
            
        seed = all_subsets[seed_idx]
        B = set(seed)
        
        while len(B) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in B:
                    continue
                g = get_gain(x, B)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            
            if best_x == -1:
                break
            B.add(best_x)
        
        if not B:
            break
        bl = sorted(B)
        blocks.append(bl)
        add_block(bl)

    # Local search: remove redundant blocks
    def try_remove_redundant():
        nonlocal blocks, covered_mask, cover_count
        improved = True
        while improved:
            if time.time() > deadline:
                return
            improved = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    return
                # Check if block i is redundant
                bl = blocks[i]
                redundant = True
                for s in itertools.combinations(bl, t):
                    idx = subset_to_index[s]
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(bl)
                    blocks.pop(i)
                    improved = True

    # Local search: 1-for-1 swaps
    def try_swaps():
        nonlocal blocks, covered_mask, cover_count
        improved = True
        while improved:
            if time.time() > deadline:
                return
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return
                bl_i = blocks[i]
                # Try removing element x from bl_i and adding it to another block
                for x in bl_i:
                    if time.time() > deadline:
                        return
                    # Temporarily remove x from bl_i
                    new_bl_i = [y for y in bl_i if y != x]
                    if len(new_bl_i) < k:
                        continue
                    
                    # Check if bl_i without x still covers what it uniquely covers
                    # Actually, we need to check if removing x from bl_i doesn't uncover any triplet
                    # that was only covered by bl_i
                    ok = True
                    for s in itertools.combinations(bl_i, t):
                        if x in s:
                            idx = subset_to_index[s]
                            if cover_count[idx] <= 1:
                                ok = False
                                break
                    if not ok:
                        continue
                    
                    # Now try to add x to some other block j
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        bl_j = blocks[j]
                        if x in bl_j:
                            continue
                        if len(bl_j) >= k:
                            continue
                        
                        # Check if adding x to bl_j is beneficial
                        # It should cover at least one new triplet
                        new_covered = 0
                        for u in bl_j:
                            for v in bl_j:
                                if u < v:
                                    idx = triplet_idx[x][u][v]
                                    if cover_count[idx] == 0:
                                        new_covered += 1
                        # For this to be a valid 1-for-1, we need new_covered > 0
                        # But actually we're just moving x, so we need the total count to not increase
                        # Since we're removing from one and adding to another, count stays same
                        # We just need validity maintained
                        if new_covered > 0:
                            # Perform the swap
                            remove_block(bl_i)
                            blocks[i] = new_bl_i
                            add_block(new_bl_i)
                            
                            bl_j.append(x)
                            bl_j.sort()
                            add_block(bl_j)
                            
                            improved = True
                            break
                        if time.time() > deadline:
                            return
                    if improved:
                        break
                if improved:
                    break

    # Run local search
    try_remove_redundant()
    try_swaps()
    try_remove_redundant()

    # If time remains, do more iterations
    while time.time() < deadline - 5:
        try_remove_redundant()
        try_swaps()
        try_remove_redundant()
        if time.time() > deadline - 2:
            break

    return blocks
