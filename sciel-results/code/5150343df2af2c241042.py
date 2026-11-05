"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import random
    import itertools

    def has_c5_incremental(adj, n, u, v):
        """Check if adding edge (u,v) creates a C5. 
        A new C5 is created iff there's a path of length 4 from u to v 
        in the current graph (without edge u-v) with all distinct vertices.
        Path: u-a-b-c-v where a,b,c are distinct from u,v and each other.
        """
        for a in range(n):
            if a == u or a == v:
                continue
            if not adj[u][a]:
                continue
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if not adj[a][b]:
                    continue
                for c in range(n):
                    if c == u or c == v or c == a or c == b:
                        continue
                    if not adj[b][c]:
                        continue
                    if adj[c][v]:
                        return True
        return False

    def has_c5_full(adj, n):
        """Full check for C5 in graph. O(n^5) but n<=35."""
        # Check all 5-tuples
        for u in range(n):
            for a in range(n):
                if a == u: continue
                if not adj[u][a]: continue
                for b in range(n):
                    if b == u or b == a: continue
                    if not adj[a][b]: continue
                    for c in range(n):
                        if c == u or c == a or c == b: continue
                        if not adj[b][c]: continue
                        for v in range(n):
                            if v == u or v == a or v == b or v == c: continue
                            if not adj[c][v]: continue
                            if adj[v][u]:
                                return True
        return False

    def has_independent_set_9(adj, n):
        """Check if there's an independent set of size 9."""
        # Independent set = clique in complement
        # Build complement
        comp = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and not adj[i][j]:
                    comp[i][j] = True

        # Backtracking to find clique of size 9 in comp
        def find_clique(comp, n, k, start=0):
            if k == 0:
                return True
            if k == 1:
                return True
            for i in range(start, n):
                # i can be in clique
                # Need k-1 more vertices from i+1..n-1 that are all connected to i and each other
                candidates = [j for j in range(i+1, n) if comp[i][j]]
                if len(candidates) < k - 1:
                    continue
                # Build sub-problem
                sub_comp = [[comp[candidates[a]][candidates[b]] for b in range(len(candidates))] 
                           for a in range(len(candidates))]
                if find_clique(sub_comp, len(candidates), k-1, 0):
                    return True
            return False

        return find_clique(comp, n, 9)

    # Start with 8 K4's: n=32, alpha=8, C5-free
    n = 32
    adj = [[False]*n for _ in range(n)]
    for b in range(8):
        for i in range(4):
            for j in range(i+1, 4):
                u = 4*b + i
                v = 4*b + j
                adj[u][v] = adj[v][u] = True

    # Verify base
    assert not has_c5_full(adj, n)
    assert not has_independent_set_9(adj, n)

    best_n = n
    best_adj = [row[:] for row in adj]

    random.seed(12345)

    # Try to extend to 33, 34, 35
    for target_n in [33, 34, 35]:
        extended = False
        for attempt in range(200):
            trial = [row[:] for row in adj]
            # Extend adjacency matrix to target_n
            for i in range(len(trial)):
                trial[i].extend([False] * (target_n - len(trial[i])))
            for i in range(len(trial), target_n):
                trial.append([False] * target_n)

            valid = True
            for new_v in range(n, target_n):
                # Randomly decide edges from new_v to existing vertices
                # Connect to a random subset
                connect_to = []
                for i in range(new_v):
                    if random.random() < 0.6:
                        connect_to.append(i)

                for i in connect_to:
                    # Check if adding edge (new_v, i) creates a C5
                    if has_c5_incremental(trial, target_n, new_v, i):
                        valid = False
                        break
                    trial[new_v][i] = True
                    trial[i][new_v] = True
                if not valid:
                    break

            if not valid:
                continue

            # Final verification
            if has_c5_full(trial, target_n):
                continue
            if has_independent_set_9(trial, target_n):
                continue

            extended = True
            best_adj = trial
            best_n = target_n
            adj = trial
            n = target_n
            break

        if not extended:
            break

    # Build edge list
    edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if best_adj[i][j]:
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
# EVOLVE-BLOCK-END
