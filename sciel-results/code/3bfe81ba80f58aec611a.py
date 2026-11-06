import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer
    
    best_n = 39
    best_edges = []
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        min_deg_req = n - 39
        if min_deg_req < 0:
            min_deg_req = 0
            
        found = False
        while not found and time.time() < deadline:
            # Random permutation
            perm = list(range(n))
            random.shuffle(perm)
            
            neighbors = [set() for _ in range(n)]
            
            # Build graph
            valid_construction = True
            for k in range(n):
                v = perm[k]
                candidates = [perm[i] for i in range(k)]
                
                # Sort by neighbor count to facilitate packing
                candidates.sort(key=lambda u: len(neighbors[u]))
                
                chosen = []
                U = set()
                
                # Greedily add valid neighbors
                for u in candidates:
                    if U.isdisjoint(neighbors[u]):
                        chosen.append(u)
                        U.update(neighbors[u])
                        
                for u in chosen:
                    neighbors[v].add(u)
                    neighbors[u].add(v)
                    
            # Check min degree
            min_deg = min(len(neighbors[i]) for i in range(n))
            if min_deg >= min_deg_req:
                # Verify C4-free
                is_c4_free = True
                for i in range(n):
                    ni = neighbors[i]
                    for j in range(i+1, n):
                        if len(ni.intersection(neighbors[j])) > 1:
                            is_c4_free = False
                            break
                    if not is_c4_free:
                        break
                
                if is_c4_free:
                    edges = []
                    for i in range(n):
                        for j in neighbors[i]:
                            if i < j:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
        
        # If we exhausted time for this n, break and return best found (which might be lower n from previous iterations? No, we go down)
        # Actually, if we don't find n=47, we try n=46.
        # If we find n=46, we return it immediately.
        # If we run out of time, we return the best we found.
        # Since we go from high to low, if we don't find 47, we try 46.
        # If we don't find 46, we try 45...
        # If we run out of time, we should return the best valid graph found so far.
        # But the loop above returns immediately upon finding a valid graph.
        # If it doesn't find one, it continues to the next n.
        # If it runs out of time in the inner loop, it breaks the inner loop, then the outer loop continues?
        # No, `while not found and time.time() < deadline` stops.
        # Then it goes to the next `n` in the outer loop.
        # We should check time before starting a new n.
        
        if time.time() > deadline:
            break

    # Fallback
    return {'n': 39, 'edges': []}
