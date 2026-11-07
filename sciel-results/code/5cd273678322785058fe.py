"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
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

    def solve_csp(n, edges, lists, rng):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains
        domains = [set(l) for l in lists]
        
        # Sort adjacency for deterministic tie-breaking in MRV if needed
        for i in range(n):
            adj[i].sort()
            
        assignment = [None] * n
        unassigned = set(range(n))
        
        # Heuristic: Find vertex with smallest domain (MRV)
        # If tie, highest degree
        def get_mrv_vertex(unassigned):
            best_v = None
            best_size = float('inf')
            best_deg = -1
            for v in unassigned:
                size = len(domains[v])
                if size == 0:
                    return v # Constraint violation
                deg = len(adj[v])
                if size < best_size or (size == best_size and deg > best_deg):
                    best_size = size
                    best_deg = deg
                    best_v = v
            return best_v

        def backtrack():
            if not unassigned:
                return True
            
            v = get_mrv_vertex(unassigned)
            
            # If a vertex has no available colors, fail
            if not domains[v]:
                return False
            
            # Shuffle colors to avoid deterministic bias if needed, 
            # but for this small problem, fixed order is usually fine.
            # However, randomizing can help if we hit a dead end in a specific structure.
            colors = list(domains[v])
            rng.shuffle(colors)
            
            for c in colors:
                # Try color c
                assignment[v] = c
                unassigned.remove(v)
                
                # Save old domains of neighbors to restore
                old_domains = {}
                consistent = True
                for u in adj[v]:
                    if u in unassigned:
                        old_domains[u] = domains[u].copy()
                        domains[u].discard(c)
                        if not domains[u]:
                            consistent = False
                            break
                
                if consistent and backtrack():
                    return True
                
                # Restore state
                for u in old_domains:
                    domains[u] = old_domains[u]
                
                unassigned.add(v)
                assignment[v] = None
            
            return False

        if backtrack():
            # Convert to list of ints to ensure type correctness
            return [int(c) for c in assignment]
        else:
            return None

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Emergency: if we are out of time, we can't solve the remaining ones.
            # But the first few should be fast. If we fail here, we return what we have?
            # No, we must return a list of 4. 
            # If we haven't solved them, we are in trouble. 
            # But given the speed, this should never happen.
            # To be safe, we will attempt a very fast greedy for any unsolved ones if time is tight, 
            # though it might fail. 
            # Better: just try to solve with a very tight time per instance.
            pass
            
        n, edges, lists = instance(splits, seed)
        
        # Use a specific RNG for search randomization, seeded deterministically for reproducibility 
        # or randomly? Deterministic is safer for "best" search if we don't restart.
        # Since we are doing a single deep search with MRV, it is deterministic if we don't shuffle.
        # Let's shuffle with a seed derived from the instance to be robust against adversarial structures.
        search_rng = Random(seed + 1)
        
        solution = solve_csp(n, edges, lists, search_rng)
        
        if solution is None:
            # Fallback: Try randomized restarts with different seeds
            found = False
            for attempt in range(20):
                if time.time() > deadline:
                    break
                search_rng = Random(seed * 31 + attempt * 7 + 1000)
                solution = solve_csp(n, edges, lists, search_rng)
                if solution is not None:
                    found = True
                    break
            if not found:
                # Last resort: greedy with random order
                greedy_rng = Random(seed + 9999)
                order = list(range(n))
                greedy_rng.shuffle(order)
                solution = [None] * n
                for v in order:
                    available = [c for c in lists[v] if all(solution[u] != c or u not in adj[v] for u in adj[v])]
                    if available:
                        solution[v] = available[0]
                    else:
                        solution[v] = lists[v][0]
        
        # Ensure types are Python int
        out.append([int(c) for c in solution])
        
        # Check time
        if time.time() > deadline:
             # If we are running out of time, we should stop doing expensive things.
             # But we must complete the loop to return 4 items.
             # The remaining instances should be solved instantly if the algorithm is fast.
             pass

    return out
# EVOLVE-BLOCK-END
