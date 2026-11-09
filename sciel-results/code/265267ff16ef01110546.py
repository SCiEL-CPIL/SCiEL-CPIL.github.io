import time
import random
import sys

def construct():
    START_TIME = time.time()
    DEADLINE = START_TIME + 140.0

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

    def solve_csp(n, edges, lists, rng, time_limit):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Domains as lists (faster than sets for small domains)
        domains = [list(l) for l in lists]
        
        # Check for empty domains
        for v in range(n):
            if not domains[v]:
                return None
        
        # AC-3 arc consistency
        def ac3(domains, adj):
            queue = []
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                u, v = queue.pop()
                if revise(domains, u, v, adj):
                    if not domains[u]:
                        return False
                    for x in adj[u]:
                        if x != v:
                            queue.append((x, u))
            return True
        
        def revise(domains, i, j, adj):
            removed = False
            for x in list(domains[i]):
                if not any(y != x for y in domains[j]):
                    domains[i].remove(x)
                    removed = True
            return removed
        
        # Initial arc consistency
        if not ac3(domains, adj):
            return None
        
        # Backtracking with MRV, forward checking, and randomized value ordering
        assigned = [-1] * n
        unassigned = list(range(n))
        
        def backtrack():
            if time.time() > time_limit:
                return False
            if not unassigned:
                return True
            
            # MRV: pick unassigned vertex with smallest domain
            min_len = n + 1
            best_idx = -1
            for i, v in enumerate(unassigned):
                l = len(domains[v])
                if l < min_len:
                    min_len = l
                    best_idx = i
                    if min_len == 1:
                        break
            
            if min_len == 0:
                return False
            
            v = unassigned[best_idx]
            unassigned.pop(best_idx)
            
            # Randomize value order
            vals = list(domains[v])
            rng.shuffle(vals)
            
            for val in vals:
                # Check consistency with assigned neighbors
                ok = True
                for u in adj[v]:
                    if assigned[u] == val:
                        ok = False
                        break
                if not ok:
                    continue
                
                assigned[v] = val
                
                # Forward checking
                saved = []
                ok_fc = True
                for u in adj[v]:
                    if assigned[u] == -1:
                        if val in domains[u]:
                            saved.append((u, val))
                            domains[u].remove(val)
                            if not domains[u]:
                                ok_fc = False
                                break
                
                if ok_fc:
                    if backtrack():
                        return True
                
                # Restore
                for u, rem in saved:
                    domains[u].append(rem)
                
                assigned[v] = -1
            
            unassigned.insert(best_idx, v)
            return False
        
        # Multiple randomized restarts
        for attempt in range(50):
            if time.time() > time_limit:
                break
            
            # Reset state
            assigned = [-1] * n
            unassigned = list(range(n))
            rng.shuffle(unassigned)
            
            # Re-apply AC-3 with fresh domains (in case previous restarts modified them)
            fresh_domains = [list(l) for l in lists]
            if not ac3(fresh_domains, adj):
                continue
            
            # Use fresh_domains for this attempt
            domains = fresh_domains
            
            if backtrack():
                return [assigned[v] for v in range(n)]
        
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            # Should not happen, but safety
            break
        
        n, edges, lists = instance(splits, seed)
        rng_solver = random.Random(seed + 1000)
        
        # Allocate time proportionally to difficulty
        # Larger instances get more time
        if splits <= 20:
            time_per_instance = 10.0
        elif splits <= 40:
            time_per_instance = 20.0
        elif splits <= 60:
            time_per_instance = 30.0
        else:
            time_per_instance = 60.0
        
        t_start = time.time()
        time_limit = min(t_start + time_per_instance, DEADLINE - 5.0)
        
        coloring = solve_csp(n, edges, lists, rng_solver, time_limit)
        
        if coloring is None:
            # Fallback: greedy with multiple random orders
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            coloring = None
            for attempt in range(100):
                order = list(range(n))
                rng_solver.shuffle(order)
                # Sort by degree descending for better greedy
                order.sort(key=lambda x: -len(adj[x]))
                col = [-1] * n
                success = True
                for v in order:
                    taken = set()
                    for u in adj[v]:
                        if col[u] != -1:
                            taken.add(col[u])
                    free = [c for c in lists[v] if c not in taken]
                    if free:
                        col[v] = rng_solver.choice(free)
                    else:
                        success = False
                        break
                if success:
                    coloring = col
                    break
            
            if coloring is None:
                # Last resort: just assign first available, may be invalid
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                col = [-1] * n
                for v in range(n):
                    taken = set()
                    for u in adj[v]:
                        if col[u] != -1:
                            taken.add(col[u])
                    free = [c for c in lists[v] if c not in taken]
                    if free:
                        col[v] = free[0]
                    else:
                        col[v] = lists[v][0]
                coloring = col
        
        # Ensure Python int types
        out.append([int(c) for c in coloring])
    
    return out
