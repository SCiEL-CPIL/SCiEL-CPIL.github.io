import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}
    
    # For fast lookup of triplet index
    # triplet_idx[a][b][c] -> index
    # We can use a dictionary or a 3D list. 24x24x24 is small.
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
    
    # Helper to get gain of adding x to block B
    # Gain = number of uncovered triplets {x, u, v} with u,v in B
    def get_gain(x, B):
        gain = 0
        B_list = list(B)
        for i in range(len(B_list)):
            u = B_list[i]
            for j in range(i+1, len(B_list)):
                v = B_list[j]
                idx = triplet_idx[x][u][v]
                if not (covered_mask >> idx) & 1:
                    gain += 1
        return gain

    while True:
        if time.time() > deadline:
            break
            
        # Check if all covered
        if covered_mask == (1 << n_subsets) - 1:
            break
            
        # Find the triplet with the minimum cover count (most "urgent")
        # If count is 0, it's uncovered. We want the one with min count.
        min_count = min(cover_count)
        if min_count > 0:
            # All are covered at least once? 
            # Actually, if min_count > 0, it doesn't mean all are covered.
            # We need to find an uncovered one. 
            # The min count logic works if we track counts correctly.
            # An uncovered triplet has count 0.
            # So we just find the first triplet with count 0.
            pass
        
        # More efficient: find first uncovered
        seed_idx = -1
        for i in range(n_subsets):
            if cover_count[i] == 0:
                seed_idx = i
                break
        
        if seed_idx == -1:
            # All covered
            break
            
        seed = all_subsets[seed_idx]
        
        # Grow block from seed
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
        
        # Update covered mask and counts
        for s in itertools.combinations(bl, t):
            idx = subset_to_index[s]
            if not (covered_mask >> idx) & 1:
                covered_mask |= 1 << idx
            cover_count[idx] += 1

    return blocks
