"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(100000)

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

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def ac3(n, adj, domains):
        """Arc consistency preprocessing. Returns False if any domain becomes empty."""
        import collections
        # Queue of arcs (u, v) to check
        queue = collections.deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        while queue:
            u, v = queue.popleft()
            # Check if there's a value in domains[u] compatible with some value in domains[v]
            revised = False
            for cu in list(domains[u]):
                # Check if cu is compatible with any value in domains[v]
                compatible = False
                for cv in domains[v]:
                    if cu != cv:
                        compatible = True
                        break
                if not compatible:
                    domains[u].discard(cu)
                    revised = True
                    if not domains[u]:
                        return False
                    # Add all arcs (w, u) where w != v
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
        return True

    def solve_csp(n, adj, lists, time_limit):
        """
        Systematic backtracking with MRV + forward checking + AC preprocessing.
        Returns a valid colouring list or None.
        """
        start_time = time.time()
        
        # Domain for each vertex: set of allowed colours
        domains = [set(c) for c in lists]
        assigned = [None] * n
        unassigned = set(range(n))
        
        # Precompute degree for tie-breaking in MRV
        degree = [len(adj[v]) for v in range(n)]
        
        # Apply arc consistency preprocessing
        if not ac3(n, adj, domains):
            return None
        
        # Check if any domain is empty after AC
        for v in range(n):
            if not domains[v]:
                return None
        
        def get_mrv_vertex():
            """Find unassigned vertex with minimum remaining values (MRV)."""
            best_v = -1
            best_count = float('inf')
            best_deg = -1
            for v in unassigned:
                cnt = len(domains[v])
                if cnt == 0:
                    return v  # fail
                if cnt < best_count or (cnt == best_count and degree[v] > best_deg):
                    best_count = cnt
                    best_v = v
                    best_deg = degree[v]
                    if cnt == 1:
                        break
            return best_v

        def forward_check(v, c):
            """
            Assign colour c to vertex v, then reduce domains of unassigned neighbors.
            Returns (success, undo_info) where undo_info is list of (neighbor, removed_color)
            """
            assigned[v] = c
            unassigned.discard(v)
            undo_info = []
            for u in adj[v]:
                if u in unassigned:
                    if c in domains[u]:
                        domains[u].discard(c)
                        undo_info.append((u, c))
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

            # LCV heuristic: try colours that constrain the fewest neighbours
            colors = list(domains[v])
            if len(colors) > 1:
                def lcv_score(c):
                    count = 0
                    for u in adj[v]:
                        if u in unassigned and c in domains[u]:
                            count += 1
                    return count
                colors.sort(key=lcv_score)
            
            for c in colors:
                ok, undo_info = forward_check(v, c)
                if ok:
                    if backtrack():
                        return True
                    # Undo
                    for u, cc in undo_info:
                        domains[u].add(cc)
                    assigned[v] = None
                    unassigned.add(v)

            return False

        if backtrack():
            # Verify the solution
            for v in range(n):
                if assigned[v] not in lists[v]:
                    return None
                for u in adj[v]:
                    if u > v and assigned[u] == assigned[v]:
                        return None
            return assigned
        else:
            return None

    def greedy(n, adj, lists):
        """Greedy assignment as fallback - ensures correct length."""
        colour = [0] * n
        # Degree ordering: process high-degree vertices first
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour[u] != 0 or u < v:  # Check already assigned neighbours
                    if u in order[:order.index(v)] if v in order else False:
                        pass
            # More robust: check all neighbours that have been assigned
            taken = set()
            for u in adj[v]:
                # If u appears earlier in order, it's been assigned
                if order.index(u) < order.index(v):
                    taken.add(colour[u])
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
            else:
                colour[v] = lists[v][0]
        return colour

    def greedy_v2(n, adj, lists):
        """Better greedy: process in order, check all previously assigned neighbours."""
        colour = [None] * n
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour[u] is not None:
                    taken.add(colour[u])
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
            else:
                colour[v] = lists[v][0]
        return colour

    start = time.time()
    deadline = start + 145.0  # 5s safety margin before 150s kill

    out = []
    # Allocate time proportionally to difficulty (larger splits = harder)
    # (20, 701), (40, 702), (60, 703), (100, 704)
    instances = [(20, 701), (40, 702), (60, 703), (100, 704)]
    # Give more time to later (harder) instances
    time_allocs = [15.0, 25.0, 35.0, 50.0]  # Total 125s, leaving 20s buffer

    for idx, (splits, seed) in enumerate(instances):
        remaining = deadline - time.time()
        if remaining < 1.0:
            # Not enough time - use fast greedy for all remaining
            n, edges, lists = instance(splits, seed)
            adj = build_adj(n, edges)
            colouring = greedy_v2(n, adj, lists)
            out.append([int(c) for c in colouring])
            continue

        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)

        # Use allocated time but don't exceed remaining
        time_limit = min(time_allocs[idx], remaining - 2.0)
        if time_limit < 0.5:
            time_limit = 0.5

        result = solve_csp(n, adj, lists, time_limit)
        
        if result is not None:
            # Verify
            valid = True
            for v in range(n):
                if result[v] not in lists[v]:
                    valid = False
                    break
                for u in adj[v]:
                    if u > v and result[u] == result[v]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                out.append([int(c) for c in result])
                continue

        # Fallback to greedy
        colouring = greedy_v2(n, adj, lists)
        out.append([int(c) for c in colouring])

    # Ensure exactly 4 results
    while len(out) < 4:
        out.append([0, 1, 2, 3, 0, 1, 2, 3])  # Dummy for cube

    return out[:4]

# EVOLVE-BLOCK-END
