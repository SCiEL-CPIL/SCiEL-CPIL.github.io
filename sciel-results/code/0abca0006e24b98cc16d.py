"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
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
        
        # Initial domains
        domains = [set(l) for l in lists]
        assignment = [-1] * n
        
        # Check time occasionally
        def check_time():
            return time.time() > deadline

        # Recursive backtracking with forward checking
        def backtrack(assigned_count):
            if check_time():
                return False
            
            if assigned_count == n:
                return True
            
            # MRV: Select unassigned vertex with minimum domain size
            min_size = float('inf')
            min_v = -1
            for v in range(n):
                if assignment[v] == -1:
                    size = len(domains[v])
                    if size < min_size:
                        min_size = size
                        min_v = v
                        if size == 0:
                            return False
                        if size == 1:
                            break # Can't get better than 1
            
            if min_v == -1:
                return True # Should not happen if assigned_count < n
            
            # LCV: Order colors by least constraining value
            # We want to try the color that removes the fewest options from neighbors
            colors = list(domains[min_v])
            
            def lcv_score(c):
                count = 0
                for u in adj[min_v]:
                    if assignment[u] == -1 and c in domains[u]:
                        count += 1
                return count
            
            colors.sort(key=lcv_score)
            
            for c in colors:
                # Assign c to min_v
                assignment[min_v] = c
                domains[min_v] = {c}
                
                # Forward checking: remove c from neighbors
                conflict = False
                removed = [] # (neighbor, color_removed)
                for u in adj[min_v]:
                    if assignment[u] == -1:
                        if c in domains[u]:
                            domains[u].remove(c)
                            removed.append(u)
                            if not domains[u]:
                                conflict = True
                                break
                
                if not conflict:
                    if backtrack(assigned_count + 1):
                        return True
                
                # Undo assignment
                assignment[min_v] = -1
                domains[min_v] = lists[min_v] # Restore original list
                for u in removed:
                    domains[u].add(c)
            
            return False

        if backtrack(0):
            return assignment
        else:
            # Fallback: should not happen as all are colorable
            # Return a valid-length list of ints to avoid invalid type errors
            # Greedy fallback
            adj_sets = [set(neigh) for neigh in adj]
            fallback = [0] * n
            for v in range(n):
                used = {fallback[u] for u in adj_sets[v] if fallback[u] != -1}
                for c in lists[v]:
                    if c not in used:
                        fallback[v] = c
                        break
                else:
                    fallback[v] = lists[v][0]
            return fallback

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, we must return something valid structurally
            # But we assume we won't.
            break
            
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists)
        # Ensure types are correct: list of ints
        out.append([int(c) for c in coloring])
        
    # If we didn't finish all 4, fill the rest with a dummy valid structure?
    # The task says "return a list of four lists".
    # If we fail to compute one, we should probably still return 4 lists to be valid.
    # However, the score is "number of properly coloured instances".
    # If we return a wrong structure, it's invalid (score 0).
    # So we must ensure we return 4 lists of correct length.
    
    expected_counts = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, _, _ = instance(splits, seed)
        expected_counts.append(n)
        
    while len(out) < 4:
        idx = len(out)
        n = expected_counts[idx]
        # Generate a dummy coloring (all 0s might violate edges, but structurally valid)
        # To maximize score, we'd want a valid one. But if we timed out, we can't solve.
        # However, the solver is very fast. This is just a safety net.
        out.append([0] * n)

    return out
# EVOLVE-BLOCK-END
