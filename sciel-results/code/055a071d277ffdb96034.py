import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s kill
    
    # Precompute masks for size 6 subsets for n up to 22?
    # We don't know n yet.
    # We can precompute inside the loop for each n.
    
    for n in range(22, 1, -1):
        if time.time() > deadline:
            break
            
        # Precompute masks for size 6
        # This is useful for the most common check
        masks_6 = []
        subsets_6 = []
        for S in itertools.combinations(range(n), 6):
            mask = 0
            for v in S:
                mask |= (1 << v)
            masks_6.append(mask)
            subsets_6.append(S)
            
        # Try a few random starts for local search
        max_starts = 5 if n >= 21 else 10
        
        for start_attempt in range(max_starts):
            if time.time() > deadline:
                break
                
            # Generate random graph
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Local search
            valid = False
            for step in range(100):
                if time.time() > deadline:
                    break
                    
                bad_subset = None
                bad_size = 0
                
                # Check size 6 using precomputed masks
                found_bad = False
                for i in range(len(masks_6)):
                    mask = masks_6[i]
                    # Check degrees
                    # We can speed this up by checking if all (adj_masks[v] & mask).bit_count() are equal
                    # For v in subsets_6[i]
                    S = subsets_6[i]
                    first_deg = -1
                    regular = True
                    for v in S:
                        d = (adj_masks[v] & mask).bit_count()
                        if first_deg == -1:
                            first_deg = d
                        elif d != first_deg:
                            regular = False
                            break
                    if regular:
                        bad_subset = S
                        bad_size = 6
                        found_bad = True
                        break
                
                if not found_bad:
                    # Check larger sizes
                    # We can do this on the fly
                    for k in range(7, n + 1):
                        if time.time() > deadline:
                            break
                        for S in itertools.combinations(range(n), k):
                            mask = 0
                            for v in S:
                                mask |= (1 << v)
                            first_deg = -1
                            regular = True
                            for v in S:
                                d = (adj_masks[v] & mask).bit_count()
                                if first_deg == -1:
                                    first_deg = d
                                elif d != first_deg:
                                    regular = False
                                    break
                            if regular:
                                bad_subset = S
                                bad_size = k
                                found_bad = True
                                break
                        if found_bad:
                            break
                        if k == n:
                            valid = True
                            break
                
                if valid:
                    break
                elif bad_subset is not None:
                    u, v = random.sample(list(bad_subset), 2)
                    if (adj_masks[u] >> v) & 1:
                        adj_masks[u] &= ~(1 << v)
                        adj_masks[v] &= ~(1 << u)
                    else:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                else:
                    break
            
            if valid:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj_masks[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        
    return {'n': 5, 'edges': []}
