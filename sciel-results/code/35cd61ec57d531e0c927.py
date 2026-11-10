import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin for 150s limit

    n = 16
    total_possible = n * (n - 1) // 2  # 120

    # Q3: 3-cube, 8 vertices labeled 0..7, edge if Hamming distance 1
    Q3_EDGES = []
    for u in range(8):
        for v in range(u + 1, 8):
            if bin(u ^ v).count('1') == 1:
                Q3_EDGES.append((u, v))

    q3_adj = [0] * 8
    for u, v in Q3_EDGES:
        q3_adj[u] |= (1 << v)
        q3_adj[v] |= (1 << u)

    # Precompute all 8-vertex subsets
    from itertools import combinations
    all_subsets8 = list(combinations(range(n), 8))
    all_subsets8_set = [set(s) for s in all_subsets8]

    def find_q3_fast(adj):
        """
        Fast Q3 detection. adj: list of 16 bitmasks.
        Returns list of 12 edges [(u,v),...] of a Q3 subgraph, or None.
        """
        for idx, S in enumerate(all_subsets8):
            S_set = all_subsets8_set[idx]
            
            # Filter: each vertex must have degree >= 3 in the induced subgraph
            ok = True
            for v in S:
                cnt = 0
                mask = adj[v]
                for u in S:
                    if u != v and (mask >> u) & 1:
                        cnt += 1
                        if cnt >= 3:
                            break
                if cnt < 3:
                    ok = False
                    break
            if not ok:
                continue

            # Backtrack: assign Q3 vertices 0..7 to vertices in S
            mapping = [-1] * 8
            used = 0  # bitmask of used vertices in S

            # Order: assign Q3 vertices in order 0..7
            # Precompute for each q3 vertex, which earlier q3 vertices are adjacent
            q3_earlier_neighbors = []
            for i in range(8):
                nbrs = []
                for j in range(i):
                    if (q3_adj[i] >> j) & 1:
                        nbrs.append(j)
                q3_earlier_neighbors.append(nbrs)

            def backtrack(qv):
                nonlocal used
                if qv == 8:
                    return True

                # Candidate vertices in S not yet used
                # For each candidate, check adjacency to already-assigned neighbors
                nbr_list = q3_earlier_neighbors[qv]

                for v in S:
                    if (used >> v) & 1:
                        continue
                    # Check if v is adjacent to all assigned neighbors of qv
                    valid = True
                    for nb in nbr_list:
                        if not (adj[mapping[nb]] >> v) & 1:
                            valid = False
                            break
                    if not valid:
                        continue

                    mapping[qv] = v
                    used |= (1 << v)
                    if backtrack(qv + 1):
                        return True
                    used &= ~(1 << v)
                    mapping[qv] = -1

                return False

            if backtrack(0):
                # Found a Q3
                edges = [(mapping[u], mapping[v]) for u, v in Q3_EDGES]
                return edges

        return None

    def count_edges(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2

    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def make_random_graph(p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def make_bipartite(a, b, seed):
        """Complete bipartite with a vertices on one side, b on the other, plus random edges within sides."""
        rng = random.Random(seed)
        adj = [0] * n
        # Side A: 0..a-1, Side B: a..a+b-1
        for u in range(a):
            for v in range(a, a + b):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        # Add some random edges within sides with low probability
        for u in range(a):
            for v in range(u + 1, a):
                if rng.random() < 0.1:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        for u in range(a, a + b):
            for v in range(u + 1, a + b):
                if rng.random() < 0.1:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    # Try multiple strategies
    best_m = 0
    best_adj = [0] * n

    strategies = []
    # Random graphs with various densities
    for p in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        for seed in range(1, 6):
            strategies.append((f"rand_{p}_{seed}", lambda p=p, seed=seed: make_random_graph(p, seed)))
    
    # Bipartite-based
    for a in [2, 3, 4]:
        b = n - a
        for seed in range(1, 4):
            strategies.append((f"bip_{a}_{b}_{seed}", lambda a=a, b=b, seed=seed: make_bipartite(a, b, seed)))

    for strat_name, make_fn in strategies:
        if time.time() > deadline - 10.0:
            break

        adj = make_fn()
        current_edges = count_edges(adj)

        # Greedy removal of Q3s
        max_removals = 300
        for _ in range(max_removals):
            if time.time() > deadline - 5.0:
                break
            q3_edges = find_q3_fast(adj)
            if q3_edges is None:
                break
            # Remove a random edge from the Q3
            rng = random.Random()
            u, v = rng.choice(q3_edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_edges -= 1

        # Check if Q3-free
        q3_edges = find_q3_fast(adj)
        if q3_edges is None and current_edges > best_m:
            best_m = current_edges
            best_adj = adj[:]

        # Local search: try to add edges
        if time.time() < deadline - 5.0:
            # Try adding each missing edge
            missing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        missing.append((u, v))
            random.shuffle(missing)

            for u, v in missing:
                if time.time() > deadline - 3.0:
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                q3_edges = find_q3_fast(adj)
                if q3_edges is None:
                    best_m = current_edges
                    best_adj = adj[:]
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_edges -= 1

    # Final local search on best_adj
    if best_m > 0 and time.time() < deadline - 5.0:
        adj = best_adj[:]
        current_edges = best_m
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    missing.append((u, v))
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 2.0:
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges += 1
            q3_edges = find_q3_fast(adj)
            if q3_edges is None:
                best_m = current_edges
                best_adj = adj[:]
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1

    return get_edge_list(best_adj)
