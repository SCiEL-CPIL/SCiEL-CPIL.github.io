"""Rewritten program using backtracking CSP solver for List Coloring."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard: 120s limit, leave 10s margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 110.0

    def instance(splits, seed, palette=5):
        from random import Random
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
        
        # Dynamic domains: list of bitmasks of available colors
        # Color c corresponds to bit (1 << c)
        domains = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            domains[i] = mask
        
        # Color assignment array
        color = [-1] * n
        uncolored = set(range(n))
        
        def backtrack():
            if not uncolored:
                return True
            
            # MRV: Select uncolored vertex with minimum remaining values
            # Tie-breaker: degree (higher is better for propagation)
            min_v = None
            min_count = 5 # Max possible colors is 5
            min_deg = -1
            
            for v in uncolored:
                # Count bits in domains[v]
                count = domains[v].bit_count()
                if count == 0:
                    return False # Dead end
                if count < min_count or (count == min_count and len(adj[v]) > min_deg):
                    min_count = count
                    min_deg = len(adj[v])
                    min_v = v
            
            v = min_v
            uncolored.remove(v)
            
            # Try each color in the domain of v
            mask = domains[v]
            while mask:
                # Get lowest set bit
                bit = mask & -mask
                c = bit.bit_length() - 1
                mask -= bit
                
                # Check if c is valid (not in colored neighbors' colors)
                valid = True
                for u in adj[v]:
                    if color[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign
                color[v] = c
                bit_val = 1 << c
                
                # Propagate: remove c from domains of uncolored neighbors
                saved_domains = []
                consistent = True
                for u in adj[v]:
                    if u in uncolored:
                        if domains[u] & bit_val:
                            domains[u] ^= bit_val
                            saved_domains.append((u, bit_val))
                            if domains[u] == 0:
                                consistent = False
                                break
                
                if consistent:
                    if backtrack():
                        return True
                
                # Undo
                color[v] = -1
                uncolored.add(v)
                for u, bv in saved_domains:
                    domains[u] |= bv
            
            return False
        
        if backtrack():
            return color
        else:
            # Fallback
            return [l[0] for l in lists]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Check deadline
        if time.time() > DEADLINE:
            # Return best effort - use greedy as fallback
            n, edges, lists = instance(splits, seed)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append([int(c) for c in colour])
            continue
        
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        out.append([int(c) for c in colouring])
    
    return out
# EVOLVE-BLOCK-END
