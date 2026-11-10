"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    _t_start = time.time()
    _DEADLINE = 140.0  # safety margin before 150s kill

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
        """Solve a list-colouring CSP using backtracking with MRV + forward checking."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # Convert to lists for speed
        adj_list = [sorted(adj[i]) for i in range(n)]
        
        # Initial domains
        domains = [set(lists[i]) for i in range(n)]
        
        # Order vertices by degree descending for initial ordering, but we'll use MRV dynamically
        # Actually, let's use a recursive backtracking with MRV
        
        def backtrack(assignment, domains):
            # Find unassigned vertex with MRV
            unassigned = [v for v in range(n) if assignment[v] is None]
            if not unassigned:
                return True
            
            # MRV: pick vertex with smallest domain
            best_v = min(unassigned, key=lambda v: len(domains[v]))
            if len(domains[best_v]) == 0:
                return False
            
            # Try each colour in the domain
            for c in sorted(domains[best_v]):
                # Check if c is consistent with assigned neighbours
                ok = True
                for u in adj_list[best_v]:
                    if assignment[u] is not None and assignment[u] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Assign and propagate (forward checking)
                assignment[best_v] = c
                # Save old domains for undo
                saved_domains = {}
                conflict = False
                for u in adj_list[best_v]:
                    if assignment[u] is None:
                        if c in domains[u]:
                            saved_domains[u] = domains[u].copy()
                            domains[u].discard(c)
                            if len(domains[u]) == 0:
                                conflict = True
                                break
                
                if not conflict:
                    if backtrack(assignment, domains):
                        return True
                
                # Undo
                assignment[best_v] = None
                for u, old_dom in saved_domains.items():
                    domains[u] = old_dom
            
            return False
        
        assignment = [None] * n
        # We need to pass mutable domains; use a list of sets
        doms = [set(l) for l in lists]
        
        # Use iterative deepening or just recursive
        # For these sizes (up to ~408 vertices), recursive should be fine
        # But Python recursion limit might be an issue, so let's increase it
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 5000))
        
        result = backtrack(assignment, doms)
        
        sys.setrecursionlimit(old_limit)
        
        if result:
            return assignment
        else:
            # Should not happen per Alon-Tarsi, but fallback
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > _DEADLINE:
            # Emergency: return best we can
            break
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists)
        out.append(coloring)
    
    # Ensure we have exactly 4 entries
    while len(out) < 4:
        out.append([])
    
    # Convert to plain Python ints and ensure list type
    result = []
    for coloring in out:
        result.append([int(c) for c in coloring])
    
    return result
# EVOLVE-BLOCK-END
