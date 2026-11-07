# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def construct():
    import random
    from random import Random
    
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

    def ac3(domains, adj, n):
        q = deque()
        for u in range(n):
            for v in adj[u]:
                q.append((u, v))
        
        while q:
            x, y = q.popleft()
            if revise(x, y):
                if not domains[x]:
                    return False
                for z in adj[x]:
                    if z != y:
                        q.append((z, x))
        return True

    def revise(x, y):
        removed = False
        dy = domains[y]
        new_dx = []
        for c in domains[x]:
            if dy != [c]:
                new_dx.append(c)
            else:
                removed = True
        
        if removed:
            domains[x] = new_dx
        return removed

    def solve_instance(n, edges, lists, deadline):
        if time.time() > deadline:
            return None
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        colour = [-1] * n
        domains = [list(lst) for lst in lists]
        
        if not ac3(domains, adj, n):
            return None
        
        for i in range(n):
            if not domains[i]:
                return None

        assigned_count = 0
        # No vertices assigned initially by AC-3, just pruned.
        
        def backtrack():
            nonlocal assigned_count
            if time.time() > deadline:
                return False
            
            if assigned_count == n:
                return True
            
            # 1. Select Variable: MRV
            min_size = float('inf')
            best_v = -1
            best_deg = -1
            
            for v in range(n):
                if colour[v] != -1:
                    continue
                size = len(domains[v])
                if size < min_size:
                    min_size = size
                    best_v = v
                    best_deg = len(adj[v])
                    if min_size <= 1:
                        break
                elif size == min_size and len(adj[v]) > best_deg:
                    best_v = v
                    best_deg = len(adj[v])
            
            if best_v == -1:
                return True # Should not happen if assigned_count < n
            
            v = best_v
            dom = domains[v]
            
            if not dom:
                return False
            
            # 2. Select Value: LCV
            lcv_order = []
            for c in dom:
                constraint_count = 0
                for u in adj[v]:
                    if colour[u] == -1 and c in domains[u]:
                        constraint_count += 1
                lcv_order.append((constraint_count, c))
            
            lcv_order.sort()
            
            for _, c in lcv_order:
                # Check validity
                valid = True
                for u in adj[v]:
                    if colour[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Assign
                colour[v] = c
                assigned_count += 1
                
                # Propagate
                saved_domains = []
                feasible = True
                for u in adj[v]:
                    if colour[u] == -1:
                        if c in domains[u]:
                            old_dom = domains[u]
                            new_dom = [x for x in old_dom if x != c]
                            saved_domains.append((u, old_dom))
                            domains[u] = new_dom
                            if not new_dom:
                                feasible = False
                                break
                
                if not feasible:
                    for u, old_dom in saved_domains:
                        domains[u] = old_dom
                    colour[v] = -1
                    assigned_count -= 1
                    continue
                
                if backtrack():
                    return True
                
                for u, old_dom in saved_domains:
                    domains[u] = old_dom
                colour[v] = -1
                assigned_count -= 1
            
            return False
        
        if backtrack():
            return [int(c) for c in colour]
        
        return None

    start = time.time()
    deadline = start + 110
    
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        if result is not None:
            out.append(result)
        else:
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [0] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v]}
                chosen = 0
                for c in lists[v]:
                    if c not in taken:
                        chosen = c
                        break
                colour[v] = chosen
            out.append([int(c) for c in colour])
    
    return out
# EVOLVE-BLOCK-END
