"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
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

    def solve_instance(n, edges, lists, deadline):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        color = [-1] * n
        avail = [set(l) for l in lists]
        
        # Precompute a good static ordering by degree descending
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        sys.setrecursionlimit(5000)
        
        def backtrack(idx):
            if time.time() > deadline:
                return 'timeout'
            if idx == n:
                return True
            
            v = order[idx]
            
            # Try each available color
            for c in sorted(avail[v]):
                # Check consistency with already colored neighbors
                ok = True
                for u in adj[v]:
                    if color[u] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Assign
                color[v] = c
                
                result = backtrack(idx + 1)
                if result is True:
                    return True
                if result == 'timeout':
                    return 'timeout'
                
                color[v] = -1
            
            return False
        
        result = backtrack(0)
        if result is True:
            return color
        return None

    out = []
    start_time = time.time()
    hard_deadline = start_time + 110.0  # Safety margin before 120s limit

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Calculate remaining time for this instance
        elapsed = time.time() - start_time
        remaining_instances = 4 - len(out)
        time_left = hard_deadline - time.time()
        instance_deadline = time.time() + (time_left / remaining_instances) * 0.95
        
        colouring = solve_instance(n, edges, lists, instance_deadline)
        
        if colouring is None:
            # Fallback: try with a different order (ascending degree)
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            color = [-1] * n
            avail = [set(l) for l in lists]
            order = sorted(range(n), key=lambda x: len(adj[x]))
            
            def backtrack2(idx):
                if time.time() > instance_deadline:
                    return 'timeout'
                if idx == n:
                    return True
                
                v = order[idx]
                
                for c in sorted(avail[v]):
                    ok = True
                    for u in adj[v]:
                        if color[u] == c:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    color[v] = c
                    result = backtrack2(idx + 1)
                    if result is True:
                        return True
                    if result == 'timeout':
                        return 'timeout'
                    color[v] = -1
                
                return False
            
            result = backtrack2(0)
            if result is True:
                colouring = color
        
        if colouring is None:
            # Last resort: all zeros (will likely score 0 for this instance)
            colouring = [0] * n
        
        # Ensure types are correct: list of Python ints
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    return out

# EVOLVE-BLOCK-END
