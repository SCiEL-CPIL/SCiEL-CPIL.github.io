import time
import sys
from random import Random

# Increase recursion limit just in case
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second margin before 150s kill
    
    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Pre-compute color masks
        # list_mask[v] has bits set for colors in lists[v]
        list_mask = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            list_mask[v] = mask
        
        # color[v] = -1 if uncolored, else color
        color = [-1] * n
        # used_mask[v] has bits set for colors used by neighbors of v
        used_mask = [0] * n
        
        # Count of uncolored vertices
        uncolored_count = n
        
        def backtrack():
            nonlocal uncolored_count
            
            if uncolored_count == 0:
                return True
            
            # Deadline check
            if time.time() > deadline:
                return False
            
            # Find uncolored vertex with Minimum Remaining Values (MRV)
            min_count = 4  # Max possible is 3, so 4 is infinity
            best_v = -1
            best_avail = 0
            
            for v in range(n):
                if color[v] != -1:
                    continue
                
                avail = list_mask[v] & ~used_mask[v]
                if avail == 0:
                    return False
                
                # Count bits
                cnt = bin(avail).count('1')
                if cnt < min_count:
                    min_count = cnt
                    best_v = v
                    best_avail = avail
                    if min_count == 1:
                        break
            
            if best_v == -1:
                return False # Should not happen if uncolored_count > 0
            
            uncolored_count -= 1
            
            # Try each available color
            # Iterate through bits in best_avail
            avail = best_avail
            while avail:
                # Get the lowest set bit
                lsb = avail & -avail
                c = lsb.bit_length() - 1
                avail -= lsb
                
                color[best_v] = c
                bit = 1 << c
                
                # Update used_mask for neighbors
                for u in adj[best_v]:
                    used_mask[u] |= bit
                
                if backtrack():
                    return True
                
                # Undo
                for u in adj[best_v]:
                    used_mask[u] ^= bit
                color[best_v] = -1
            
            uncolored_count += 1
            return False
        
        if backtrack():
            # Convert to list of ints
            return [int(c) for c in color]
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Check deadline before starting a new instance
        if time.time() > deadline:
            # If we run out of time, we must still return a valid structure.
            # We generate the instance to get n, then return a dummy coloring.
            n, _, _ = instance(splits, seed)
            out.append([0] * n)
            continue
        
        n, edges, lists = instance(splits, seed)
        res = solve_instance(n, edges, lists)
        
        if res is None:
            # Fallback: If backtracking fails (should not happen for these instances 
            # but could due to timeout or bug), return a dummy valid-length list.
            # This ensures the output structure is valid, though the score for this 
            # instance will be 0.
            out.append([0] * n)
        else:
            out.append(res)
            
    return out
