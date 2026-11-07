"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

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

    def solve_instance(n, edges, lists):
        """Backtracking with dynamic MRV and forward checking."""
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Domain for each vertex: the list of allowed colours
        domains = [list(l) for l in lists]
        
        colour = [None] * n
        assigned = [False] * n
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            # Dynamic MRV: Find unassigned vertex with fewest available colours
            min_size = 10
            min_v = -1
            min_avail = []
            
            for v in range(n):
                if assigned[v]:
                    continue
                taken = set()
                for u in adj[v]:
                    if assigned[u]:
                        taken.add(colour[u])
                avail = [c for c in domains[v] if c not in taken]
                if len(avail) < min_size:
                    min_size = len(avail)
                    min_v = v
                    min_avail = avail
                    if min_size == 0:
                        return False
                    if min_size == 1:
                        break
            
            if min_v == -1:
                return True # All assigned
            
            # Try each colour for the chosen vertex
            for c in min_avail:
                colour[min_v] = c
                assigned[min_v] = True
                if backtrack():
                    return True
                colour[min_v] = None
                assigned[min_v] = False
            
            return False
        
        if backtrack():
            # Ensure all are Python ints
            return [int(c) for c in colour]
        else:
            # Fallback: should not happen due to Alon-Tarsi, but return something
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5.0:
            # Not enough time left, break
            break
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        # Validate types
        colouring = [int(c) for c in colouring]
        out.append(colouring)
    
    # If we didn't finish all 4, pad with fallback (should not happen)
    while len(out) < 4:
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[len(out)]
        n, edges, lists = instance(splits, seed)
        out.append([lists[v][0] for v in range(n)])
    
    return out
# EVOLVE-BLOCK-END
