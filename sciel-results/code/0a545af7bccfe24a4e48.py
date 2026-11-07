"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
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

    def is_valid(colour, n, adj, lists):
        for v in range(n):
            c = colour[v]
            if not isinstance(c, int) or c not in lists[v]:
                return False
            for u in adj[v]:
                if colour[u] == c:
                    return False
        return True

    def greedy_color(n, adj, lists):
        """Try greedy coloring with degree-descending order."""
        colour = [-1] * n
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour[u] != -1:
                    taken.add(colour[u])
            for c in lists[v]:
                if c not in taken:
                    colour[v] = c
                    break
            else:
                colour[v] = lists[v][0]
        return colour

    def backtrack_solve(n, adj, lists, order, deadline_local):
        """Backtracking with forward checking and MRV heuristic."""
        sys.setrecursionlimit(max(5000, n + 100))
        
        domains = [set(l) for l in lists]
        assignment = [-1] * n
        
        def check_time():
            return time.time() > deadline_local
        
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
                            break
            
            if min_v == -1:
                return True
            
            # LCV: Order colors by least constraining value
            colors = list(domains[min_v])
            
            def lcv_score(c):
                count = 0
                for u in adj[min_v]:
                    if assignment[u] == -1 and c in domains[u]:
                        count += 1
                return count
            
            colors.sort(key=lcv_score)
            
            for c in colors:
                assignment[min_v] = c
                domains[min_v] = {c}
                
                conflict = False
                removed = []
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
                
                assignment[min_v] = -1
                domains[min_v] = set(lists[min_v])
                for u in removed:
                    domains[u].add(c)
            
            return False

        if backtrack(0):
            return assignment
        return None

    def solve_instance(n, edges, lists, time_budget):
        """Solve a single instance with greedy first, then backtracking."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Try greedy first - very fast
        colour = greedy_color(n, adj, lists)
        if is_valid(colour, n, adj, lists):
            return colour
        
        # Greedy failed, try backtracking with different orderings
        # Ordering 1: degree descending
        order1 = sorted(range(n), key=lambda u: -len(adj[u]))
        result = backtrack_solve(n, adj, lists, order1, time.time() + time_budget * 0.4)
        if result is not None and is_valid(result, n, adj, lists):
            return result
        
        # Ordering 2: most constrained first (smallest list, then highest degree)
        if time.time() < deadline:
            order2 = sorted(range(n), key=lambda u: (len(lists[u]), -len(adj[u])))
            result = backtrack_solve(n, adj, lists, order2, time.time() + time_budget * 0.3)
            if result is not None and is_valid(result, n, adj, lists):
                return result
        
        # Ordering 3: degree ascending
        if time.time() < deadline:
            order3 = sorted(range(n), key=lambda u: len(adj[u]))
            result = backtrack_solve(n, adj, lists, order3, time.time() + time_budget * 0.2)
            if result is not None and is_valid(result, n, adj, lists):
                return result
        
        # Fallback: return greedy result (may be invalid but structurally correct)
        return colour

    out = []
    instances = [(20, 701), (40, 702), (60, 703), (100, 704)]
    
    for idx, (splits, seed) in enumerate(instances):
        remaining_time = deadline - time.time()
        if remaining_time < 1.0:
            # Not enough time; generate minimal valid structure
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue
        
        # Allocate time budget: give more time to harder instances
        # Instance sizes: n = 8 + 4*splits, so 88, 168, 248, 408
        n_est = 8 + 4 * splits
        # Weight by size, but ensure at least 2s each
        time_budget = max(2.0, remaining_time * (n_est / 832.0))  # 832 = sum of all n
        time_budget = min(time_budget, remaining_time - 0.5)
        
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists, time_budget)
        
        # Ensure types are correct: list of Python ints
        out.append([int(c) for c in coloring])
    
    # Ensure we have exactly 4 results
    while len(out) < 4:
        idx = len(out)
        splits, seed = instances[idx]
        n, _, _ = instance(splits, seed)
        out.append([0] * n)
    
    return out
# EVOLVE-BLOCK-END
