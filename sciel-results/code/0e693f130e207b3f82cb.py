"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    DEADLINE = 145.0  # safety margin before 150s kill
    start_time = time.time()

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

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def solve_csp(n, adj, lists, time_limit=10.0):
        """
        Systematic backtracking with MRV + forward checking, time-limited.
        Returns a valid colouring list or a fallback list of length n.
        """
        import time
        import heapq
        start_time = time.time()
        
        # Domain for each vertex: list of allowed colours
        domains = [set(c) for c in lists]
        assigned = [None] * n
        unassigned = set(range(n))
        
        # Precompute degree for tie-breaking
        degree = [len(adj[v]) for v in range(n)]
        
        # Use a heap for efficient MRV selection
        # (domain_size, degree, vertex)
        mrv_heap = [(len(lists[v]), degree[v], v) for v in range(n)]
        heapq.heapify(mrv_heap)
        
        def get_mrv_vertex():
            """Find unassigned vertex with minimum remaining values (MRV)."""
            while mrv_heap:
                cnt, deg, v = mrv_heap[0]
                if v not in unassigned:
                    heapq.heappop(mrv_heap)
                    continue
                if cnt == 0:
                    return v  # fail
                return v
            return -1

        def forward_check(v, c):
            """
            Assign colour c to vertex v, then reduce domains of unassigned neighbors.
            Returns (success, undo_info) where undo_info is list of (neighbor, removed_colors)
            """
            assigned[v] = c
            unassigned.remove(v)
            undo_info = []
            for u in adj[v]:
                if u in unassigned:
                    if c in domains[u]:
                        domains[u].discard(c)
                        undo_info.append((u, c))
                        # Update heap entry for u
                        heapq.heappush(mrv_heap, (len(domains[u]), degree[u], u))
                        if len(domains[u]) == 0:
                            # Backtrack immediately
                            for uu, cc in undo_info:
                                domains[uu].add(cc)
                            assigned[v] = None
                            unassigned.add(v)
                            return False, None
            return True, undo_info

        def backtrack():
            if not unassigned:
                return True
            
            if time.time() - start_time > time_limit:
                return False

            v = get_mrv_vertex()
            if v == -1:
                return True

            if len(domains[v]) == 0:
                return False

            # Try colours in the domain
            for c in sorted(domains[v]):
                ok, undo_info = forward_check(v, c)
                if ok:
                    if backtrack():
                        return True
                    # Undo
                    for u, cc in undo_info:
                        domains[u].add(cc)
                    # Re-push updated domain size for u
                    heapq.heappush(mrv_heap, (len(domains[u]), degree[u], u))
                    assigned[v] = None
                    unassigned.add(v)
                    # Re-push v with its original domain size
                    heapq.heappush(mrv_heap, (len(domains[v]), degree[v], v))

            return False

        if backtrack():
            return assigned
        else:
            # Fallback: Greedy assignment with proper conflict checking
            # Process in order of degree descending (high-degree vertices first)
            color = [None] * n
            order = sorted(range(n), key=lambda u: -len(adj[u]))
            for v in order:
                taken = set()
                for u in adj[v]:
                    if color[u] is not None:
                        taken.add(color[u])
                # Find a color in lists[v] not in taken
                free = [c for c in lists[v] if c not in taken]
                if free:
                    color[v] = free[0]
                else:
                    # No free color; pick any from list (may cause conflict)
                    color[v] = lists[v][0]
            # Ensure all are Python int
            return [int(c) if c is not None else 0 for c in color]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Calculate remaining time
        remaining = DEADLINE - time.time()
        if remaining < 2.0:
            # Not enough time for a full solve, but we must return valid structure
            # We will generate the instance and return a quick greedy solution
            # to ensure correct length and types.
            n, edges, lists = instance(splits, seed)
            adj = build_adj(n, edges)
            # Quick greedy with proper conflict checking
            colouring = solve_csp(n, adj, lists, time_limit=0.5)
            colouring = [int(c) for c in colouring]
            out.append(colouring)
            continue

        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)

        # Try CSP solver with a time limit based on remaining time
        # Allocate proportional time or fixed chunk
        time_limit = min(20.0, remaining / 2.0) 
        colouring = solve_csp(n, adj, lists, time_limit=time_limit)

        # Ensure all values are Python int
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    # Ensure we have exactly 4 results with correct lengths
    # If somehow we missed some (should not happen with the loop above), pad with dummy valid-length lists?
    # The loop runs 4 times always.
    return out
# EVOLVE-BLOCK-END
