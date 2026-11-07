"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

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

    def solve(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        domains = [list(l) for l in lists]
        
        # Precompute degrees for tie-breaking in MRV if needed
        degrees = [len(adj[i]) for i in range(n)]

        def backtrack(domains):
            # Find unassigned variable with smallest domain (MRV)
            min_size = float('inf')
            min_var = -1
            min_degree = -1
            for i in range(n):
                if len(domains[i]) > 0:
                    if len(domains[i]) < min_size:
                        min_size = len(domains[i])
                        min_var = i
                        min_degree = degrees[i]
                    elif len(domains[i]) == min_size:
                        if degrees[i] > min_degree:
                            min_degree = degrees[i]
                            min_var = i
                    if min_size == 1:
                        # Can't get smaller than 1
                        # But we might want to check others if we want best MRV? 
                        # No, 1 is the minimum possible.
                        # However, there might be multiple vars with size 1.
                        # Picking the one with highest degree is good.
                        # The loop above handles this if we don't break immediately.
                        # Let's not break, to find the best among size 1s.
                        pass
            
            if min_var == -1:
                return True
            
            original_domain = domains[min_var]
            
            # Try each value
            for val in original_domain:
                domains[min_var] = [val]
                
                # Forward Check
                valid = True
                undo_log = []
                for nb in adj[min_var]:
                    if len(domains[nb]) > 1:
                        if val in domains[nb]:
                            old_domain = domains[nb][:]
                            new_domain = [c for c in domains[nb] if c != val]
                            domains[nb] = new_domain
                            undo_log.append((nb, old_domain))
                            if not new_domain:
                                valid = False
                                break
                
                if valid:
                    if backtrack(domains):
                        return True
                
                # Undo
                for nb, old_domain in undo_log:
                    domains[nb] = old_domain
                domains[min_var] = original_domain

            return False

        if backtrack(domains):
            return [d[0] for d in domains]
        else:
            # Should not happen
            return [l[0] for l in lists]

    out = []
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Pre-generate instances to know n for dummy fills if needed
    instance_params = [(20, 701), (40, 702), (60, 703), (100, 704)]
    instances_data = []
    for splits, seed in instance_params:
        n, edges, lists = instance(splits, seed)
        instances_data.append((n, edges, lists))
        
    for i in range(4):
        n, edges, lists = instances_data[i]
        
        if time.time() > deadline:
            # Time's up. Fill with dummy.
            out.append([0] * n)
        else:
            sol = solve(n, edges, lists)
            # Ensure Python ints
            sol = [int(c) for c in sol]
            out.append(sol)
            
    return out
# EVOLVE-BLOCK-END
