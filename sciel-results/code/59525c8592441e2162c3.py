"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import deque

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
        """
        Solve the list-coloring problem using forward checking with backtracking.
        Returns a list of colors (integers) of length n, or None if no solution found.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Initialize domains as sets for efficient removal
        domains = [set(lst) for lst in lists]

        # We'll use a recursive backtracking with forward checking
        # Order vertices dynamically using MRV (Minimum Remaining Values)
        
        # For efficiency, we track which vertices are assigned
        assigned = [False] * n
        color = [None] * n

        # To speed up MRV selection, we can maintain a priority structure,
        # but for n up to ~408 (8 + 4*100), a simple scan is fine.
        
        def get_unassigned():
            return [v for v in range(n) if not assigned[v]]

        def mrv_order(unassigned_vertices):
            # Sort by domain size (MRV heuristic)
            return sorted(unassigned_vertices, key=lambda v: len(domains[v]))

        def forward_check(v, c, unassigned_vertices):
            """
            Remove color c from domains of unassigned neighbors of v.
            Returns False if any domain becomes empty (contradiction).
            """
            for u in adj[v]:
                if not assigned[u] and c in domains[u]:
                    domains[u].remove(c)
                    if len(domains[u]) == 0:
                        # Contradiction found
                        return False
            return True

        def backtrack():
            # Select unassigned vertex with smallest domain (MRV)
            unassigned = [v for v in range(n) if not assigned[v]]
            if not unassigned:
                return True  # All vertices assigned

            # MRV: pick vertex with smallest domain
            v = min(unassigned, key=lambda x: len(domains[x]))
            
            # Try each color in the domain
            for c in list(domains[v]):  # Use list to avoid modification issues during iteration
                # Assign color c to v
                assigned[v] = True
                color[v] = c
                
                # Forward check
                if forward_check(v, c, unassigned):
                    if backtrack():
                        return True
                
                # Undo assignment (backtrack)
                assigned[v] = False
                color[v] = None
                # Note: forward_check only removes colors, so we need to restore domains
                # This is a problem with simple forward checking without trail-based undo.
                # We need a proper undo mechanism.

            return False

        # The simple approach above doesn't properly undo domain changes.
        # Let's implement a proper backtracking with domain restoration.
        
        # Reset
        domains = [set(lst) for lst in lists]
        assigned = [False] * n
        color = [None] * n

        def backtrack_proper():
            # Select unassigned vertex with smallest domain (MRV)
            unassigned = [v for v in range(n) if not assigned[v]]
            if not unassigned:
                return True  # All vertices assigned

            # MRV: pick vertex with smallest domain
            v = min(unassigned, key=lambda x: len(domains[x]))
            
            # Save the domain of v before trying colors
            saved_domain_v = set(domains[v])
            
            # Try each color in the domain
            for c in list(saved_domain_v):
                # Assign color c to v
                assigned[v] = True
                color[v] = c
                
                # Remove c from domains of unassigned neighbors
                changes = []
                consistent = True
                for u in adj[v]:
                    if not assigned[u] and c in domains[u]:
                        domains[u].remove(c)
                        changes.append(u)
                        if len(domains[u]) == 0:
                            consistent = False
                            break
                
                if consistent and backtrack_proper():
                    return True
                
                # Undo: restore domains
                for u in changes:
                    domains[u].add(c)
                domains[v] = saved_domain_v
                assigned[v] = False
                color[v] = None

            return False

        if backtrack_proper():
            return color
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_csp(n, edges, lists)
        if coloring is None:
            # Fallback: should not happen per Alon-Tarsi, but just in case
            # Use a simple greedy as fallback
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            coloring = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {coloring[u] for u in adj[v] if coloring[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                coloring[v] = free[0] if free else lists[v][0]
        # Ensure types are Python int
        out.append([int(c) for c in coloring])
    return out
# EVOLVE-BLOCK-END
