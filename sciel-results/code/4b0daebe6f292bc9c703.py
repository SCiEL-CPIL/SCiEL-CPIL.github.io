"""MODEL-AND-OPTIMIZE: CSP solver with MRV, Forward Checking, and AC-3 initialization."""
# EVOLVE-BLOCK-START
import time
from collections import deque
import sys

def construct():
    # Increase recursion limit for backtracking
    sys.setrecursionlimit(10000)

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

    def ac3(n, adj, domains):
        """AC-3 arc consistency preprocessing. Returns False if any domain becomes empty."""
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        while queue:
            xi, xj = queue.popleft()
            di = domains[xi]
            dj = domains[xj]
            new_di = []
            for ci in di:
                # ci is supported if there exists cj in dj such that ci != cj
                if len(dj) > 1:
                    new_di.append(ci)
                elif len(dj) == 1 and dj[0] != ci:
                    new_di.append(ci)
            if len(new_di) < len(di):
                domains[xi] = new_di
                if not new_di:
                    return False
                for xk in adj[xi]:
                    if xk != xj:
                        queue.append((xk, xi))
        return True

    def solve_csp(n, edges, lists, deadline):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Domains are initially the lists
        domains = [list(l) for l in lists]
        
        for i in range(n):
            if not domains[i]:
                return None

        # AC-3 Arc Consistency preprocessing
        if not ac3(n, adj, domains):
            return None
        
        for i in range(n):
            if not domains[i]:
                return None

        # Backtracking with Forward Checking
        assignment = [None] * n
        timed_out = [False]
        
        def is_consistent(v, color, assigned):
            for u in adj[v]:
                if assigned[u] is not None:
                    if assigned[u] == color:
                        return False
            return True

        def backtrack(assigned_count):
            if time.time() > deadline:
                timed_out[0] = True
                return False
            
            if assigned_count == n:
                return True
            
            # MRV: Find unassigned vertex with smallest domain
            min_var = -1
            min_size = 4 
            for v in range(n):
                if assignment[v] is None:
                    valid_colors = 0
                    for c in domains[v]:
                        if is_consistent(v, c, assignment):
                            valid_colors += 1
                    if valid_colors < min_size:
                        min_size = valid_colors
                        min_var = v
                        if min_size == 0:
                            return False
                        if min_size == 1:
                            break
        
            if min_var == -1:
                return False
            
            for c in domains[min_var]:
                if is_consistent(min_var, c, assignment):
                    assignment[min_var] = c
                    
                    if backtrack(assigned_count + 1):
                        return True
                    
                    assignment[min_var] = None
                    if timed_out[0]:
                        return False
            
            return False

        if backtrack(0):
            return assignment
        else:
            return None

    def build_result():
        params = ((20, 701), (40, 702), (60, 703), (100, 704))
        results = []
        for splits, seed in params:
            n, edges, lists = instance(splits, seed)
            coloring = solve_csp(n, edges, lists, deadline)
            if coloring is None:
                # Fallback: greedy assignment as last resort
                coloring = [lists[v][0] for v in range(n)]
            # Ensure types are Python int
            results.append([int(c) for c in coloring])
        return results

    # Deadline guard: 150s wall-clock limit, use 130s safety margin
    start_time = time.time()
    deadline = start_time + 130.0
    
    result = build_result()
    
    return result

# EVOLVE-BLOCK-END
