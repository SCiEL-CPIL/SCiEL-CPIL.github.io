import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    # Prioritize higher n
    # Allocate time based on likelihood of success
    # n=47 is harder, but if found, is better.
    # n=46 is likely to be found quickly.
    
    # Try n=46 first? No, maximize n.
    # But if 47 takes 130s and fails, we get 0.
    # So we must be careful.
    # Let's try 47 for a short time, then 46.
    
    attempts = [
        (47, 10.0),
        (46, 120.0),
        (45, 5.0),
        (44, 5.0)
    ]
    
    for n, time_limit in attempts:
        if time.time() > deadline - 2.0:
            break
            
        target_deg = n - 39
        if target_deg < 0:
            target_deg = 0
            
        if target_deg == 0:
            return {'n': n, 'edges': []}

        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            masks = [0] * n
            degrees = [0] * n
            
            success = True
            while min(degrees) < target_deg:
                if time.time() > end_time:
                    success = False
                    break
                
                min_d = min(degrees)
                u_candidates = [i for i in range(n) if degrees[i] == min_d]
                u = random.choice(u_candidates)
                
                mask_u = masks[u]
                non_neighbors = [v for v in range(n) if v != u and not (mask_u & (1 << v))]
                random.shuffle(non_neighbors)
                
                added = False
                for v in non_neighbors:
                    if popcount(mask_u & masks[v]) <= 1:
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        added = True
                        break
                        
                if not added:
                    success = False
                    break
            
            if success:
                # Verify
                valid = True
                if min(degrees) < target_deg:
                    valid = False
                else:
                    for i in range(n):
                        for j in range(i+1, n):
                            if popcount(masks[i] & masks[j]) > 1:
                                valid = False
                                break
                        if not valid:
                            break
                            
                if valid:
                    edges = []
                    for i in range(n):
                        mi = masks[i]
                        for j in range(i+1, n):
                            if mi & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}

    return {'n': 39, 'edges': []}
