"""Rewritten program using bitmask CSP solver with AC-3 preprocessing and MRV backtracking."""
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
        
        # Convert lists to bitmasks for fast operations
        # Each color c is represented as bit (1 << c)
        domains = [0] * n
        for i in range(n):
            mask = 0
            for c in lists[i]:
                mask |= (1 << c)
            domains[i] = mask
        
        # AC-3 preprocessing: propagate arc consistency
        # For != constraint, if neighbor j has domain {c}, remove c from domain of i
        queue = []
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        head = 0
        while head < len(queue):
            i, j = queue[head]
            head += 1
            # If domains[j] is a singleton, remove that value from domains[i]
            if domains[j] and (domains[j] & (domains[j] - 1)) == 0:
                # domains[j] is a singleton
                val = domains[j]
                if domains[i] & val:
                    domains[i] &= ~val
                    if domains[i] == 0:
                        return None  # Inconsistent
                    for k in adj[i]:
                        if k != j:
                            queue.append((k, i))
        
        # Final check
        for d in domains:
            if d == 0:
                return None
        
        # MRV backtracking with bitmask forward checking
        color = [-1] * n
        
        def popcount(x):
            return bin(x).count('1')
        
        def backtrack():
            if time.time() > DEADLINE:
                return False
            
            # Find uncolored vertex with minimum remaining values (MRV)
            best_v = -1
            best_count = 5
            for v in range(n):
                if color[v] == -1:
                    cnt = popcount(domains[v])
                    if cnt == 0:
                        return False
                    if cnt < best_count:
                        best_count = cnt
                        best_v = v
                        if cnt == 1:
                            break
            
            if best_v == -1:
                return True  # All vertices colored
            
            v = best_v
            mask = domains[v]
            
            # Try each color in the domain
            while mask:
                c = (mask & (-mask)).bit_length() - 1  # lowest set bit
                mask &= mask - 1  # remove lowest set bit
                
                # Check if color c is valid (no neighbor has color c)
                valid = True
                for u in adj[v]:
                    if color[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign color
                color[v] = c
                old_domains = []
                
                # Forward checking: update domains of uncolored neighbors
                consistent = True
                for u in adj[v]:
                    if color[u] == -1:
                        old = domains[u]
                        old_domains.append((u, old))
                        new = old & ~(1 << c)
                        if new == 0:
                            consistent = False
                            break
                        domains[u] = new
                
                if consistent:
                    if backtrack():
                        return True
                
                # Undo
                color[v] = -1
                for u, old in old_domains:
                    domains[u] = old
            
            return False
        
        if backtrack():
            return color
        else:
            # Fallback: should not happen by Alon-Tarsi, but return something
            return [lists[i][0] for i in range(n)]

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
        if colouring is None:
            # Fallback to greedy
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
        else:
            out.append([int(c) for c in colouring])
    
    return out
# EVOLVE-BLOCK-END
