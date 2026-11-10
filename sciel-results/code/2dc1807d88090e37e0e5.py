"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

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

    def solve_csp(n, edges, lists):
        # Build adjacency lists
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to sets for fast lookup, and keep as sorted list for iteration
        domains = [set(lst) for lst in lists]
        domain_lists = [list(lst) for lst in lists]
        
        # AC-3 arc consistency
        def ac3(domains, adj):
            # Queue of arcs (u, v) meaning we need to check if domain[u] is consistent with domain[v]
            queue = []
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                u, v = queue.pop()
                # Check if any value in domain[u] has no support in domain[v]
                to_remove = []
                for cu in domains[u]:
                    has_support = False
                    for cv in domains[v]:
                        if cu != cv:
                            has_support = True
                            break
                    if not has_support:
                        to_remove.append(cu)
                
                if to_remove:
                    for cu in to_remove:
                        domains[u].discard(cu)
                    # If domain[u] became empty, we have a conflict
                    if not domains[u]:
                        return False
                    # For each neighbor w of u (except v), add arc (w, u) to queue
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
            return True

        # Make a copy of domains for AC-3
        domains_copy = [set(d) for d in domains]
        if not ac3(domains_copy, adj):
            return None  # Should not happen per problem statement, but safety
        
        # Use the pruned domains for backtracking
        # Order vertices by degree (descending) for variable ordering
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        
        # Backtracking search with forward checking
        colour = [None] * n
        assigned = [False] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            # Select next unassigned variable from the order
            v = order[idx]
            
            # Try each colour in the domain
            for c in domain_lists[v]:
                if c not in domains_copy[v]:
                    continue
                # Check if c is compatible with already assigned neighbours
                compatible = True
                for u in adj[v]:
                    if assigned[u] and colour[u] == c:
                        compatible = False
                        break
                
                if not compatible:
                    continue
                
                # Assign
                colour[v] = c
                assigned[v] = True
                
                # Forward checking: check if this assignment causes any unassigned neighbour's domain to become empty
                fc_ok = True
                # Save domains for backtracking
                saved_domains = {}
                for u in adj[v]:
                    if not assigned[u]:
                        saved_domains[u] = set(domains_copy[u])
                        # Remove c from domain of u
                        domains_copy[u].discard(c)
                        if not domains_copy[u]:
                            fc_ok = False
                            break
                
                if fc_ok:
                    if backtrack(idx + 1):
                        return True
                
                # Undo forward checking
                for u in adj[v]:
                    if not assigned[u] and u in saved_domains:
                        domains_copy[u] = saved_domains[u]
                
                # Undo assignment
                colour[v] = None
                assigned[v] = False
            
            return False

        # We need to be careful: the forward checking above modifies domains_copy in place.
        # We should restore domains_copy to its AC-3 pruned state before starting backtracking.
        # Actually, the AC-3 pruned domains are a good starting point.
        # Let's reset domains_copy to the AC-3 result.
        domains_copy = [set(d) for d in domains]
        if not ac3(domains_copy, adj):
            return None

        if backtrack(0):
            # Ensure all colours are Python ints
            return [int(c) for c in colour]
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5.0:
            # Not enough time, break and return what we have (should not happen)
            break
        n, edges, lists = instance(splits, seed)
        colouring = solve_csp(n, edges, lists)
        if colouring is None:
            # Fallback: should not happen, but return something to avoid crash
            # This would score 0 for this instance
            colouring = [0] * n
        out.append(colouring)
    
    return out
# EVOLVE-BLOCK-END
