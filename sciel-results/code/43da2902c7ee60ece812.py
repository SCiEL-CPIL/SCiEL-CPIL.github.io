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

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        init_available = [set(lst) for lst in lists]
        assigned = [-1] * n
        
        def backtrack(assigned, available):
            # Find MRV vertex
            v = -1
            min_len = float('inf')
            for i in range(n):
                if assigned[i] == -1:
                    l = len(available[i])
                    if l < min_len:
                        min_len = l
                        v = i
                        if l == 0:
                            return False
                        if l == 1:
                            break
            
            if v == -1:
                return True
            
            colors = list(available[v])
            
            # LCV Ordering
            scored_colors = []
            for c in colors:
                score = 0
                for u in adj[v]:
                    if assigned[u] == -1 and c in available[u]:
                        score += 1
                scored_colors.append((score, c))
            
            scored_colors.sort()
            
            for score, c in scored_colors:
                new_assigned = assigned[:]
                new_assigned[v] = c
                
                valid = True
                new_available = [s.copy() for s in available]
                new_available[v] = set()
                
                for u in adj[v]:
                    if new_assigned[u] != -1:
                        if new_assigned[u] == c:
                            valid = False
                            break
                    else:
                        if c in new_available[u]:
                            new_available[u].remove(c)
                            if not new_available[u]:
                                valid = False
                                break
                
                if valid:
                    if backtrack(new_assigned, new_available):
                        return True
            
            return False

        if backtrack(assigned, init_available):
            return [int(c) for c in assigned]
        else:
            # Fallback: Greedy
            return greedy(n, edges, lists)

    def greedy(n, edges, lists):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        # Order by degree descending
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        return [int(c) for c in colour]

    out = []
    start_time = time.time()
    deadline = 140.0 # 10s margin before 150s limit

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Timeout: use greedy for this and remaining
            n, edges, lists = instance(splits, seed)
            out.append(greedy(n, edges, lists))
        else:
            n, edges, lists = instance(splits, seed)
            out.append(solve_instance(n, edges, lists))
            
    return out
# EVOLVE-BLOCK-END
