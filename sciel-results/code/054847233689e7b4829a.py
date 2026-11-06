import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def popcount(x):
        return bin(x).count('1')

    def to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def find_k4(adj, n):
        """Find one K4 in the graph. Returns (a,b,c,d) or None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common_ab = adj[a] & adj[b]
                # Need c,d in common_ab with c > b, d > c, and edge (c,d)
                c = common_ab
                while c:
                    l = c & -c
                    c_idx = l.bit_length() - 1
                    c &= c - 1
                    if c_idx <= b:
                        continue
                    common_abc = common_ab & adj[c_idx]
                    d = common_abc
                    while d:
                        m = d & -d
                        d_idx = m.bit_length() - 1
                        d &= d - 1
                        if d_idx > c_idx:
                            return (a, b, c_idx, d_idx)
        return None

    def find_ind6(adj, n):
        """Find one independent set of 6 in the graph. Returns tuple of 6 vertices or None."""
        mask = (1 << n) - 1
        for a in range(n):
            for b in range(a + 1, n):
                if (adj[a] >> b) & 1:
                    continue
                non_common_ab = (~adj[a]) & (~adj[b]) & mask
                for c in range(b + 1, n):
                    if (adj[a] >> c) & 1 or (adj[b] >> c) & 1:
                        continue
                    non_common_abc = non_common_ab & (~adj[c]) & mask
                    for d in range(c + 1, n):
                        if (adj[a] >> d) & 1 or (adj[b] >> d) & 1 or (adj[c] >> d) & 1:
                            continue
                        non_common_abcd = non_common_abc & (~adj[d]) & mask
                        for e in range(d + 1, n):
                            if (adj[a] >> e) & 1 or (adj[b] >> e) & 1 or (adj[c] >> e) & 1 or (adj[d] >> e) & 1:
                                continue
                            non_common_abcde = non_common_abcd & (~adj[e]) & mask
                            for f in range(e + 1, n):
                                if (adj[a] >> f) & 1 or (adj[b] >> f) & 1 or (adj[c] >> f) & 1 or (adj[d] >> f) & 1 or (adj[e] >> f) & 1:
                                    continue
                                return (a, b, c, d, e, f)
        return None

    def is_valid(adj, n):
        return find_k4(adj, n) is None and find_ind6(adj, n) is None

    def greedy_repair(adj, n, max_iters=50000):
        """
        Greedy edge-flip local search that targets violating K4s and I6s.
        When a K4 is found, try flipping edges to break it.
        When an I6 is found, try flipping edges to break it.
        """
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return False, adj

            # Check for K4
            k4 = find_k4(adj, n)
            if k4 is not None:
                a, b, c, d = k4
                # Try flipping each edge in the K4 to break it
                # The K4 has 6 edges. Flipping any one breaks this K4.
                # But we need to be careful not to create new violations.
                # Strategy: try flipping each edge, check if it reduces total violations
                best_flip = None
                best_score = float('inf')

                edges_in_k4 = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
                random.shuffle(edges_in_k4)

                for u, v in edges_in_k4:
                    # Try removing edge (u,v)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

                    # Quick check: does this still have a K4 containing u,v?
                    # Count remaining K4s involving this set
                    k4_after = find_k4(adj, n)
                    i6_after = find_ind6(adj, n)

                    # Score: prefer no K4, then no I6
                    score = (1 if k4_after else 0) * 10 + (1 if i6_after else 0)

                    if score < best_score:
                        best_score = score
                        best_flip = (u, v, 'remove')

                    # Restore
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

                    if best_score == 0:
                        break

                if best_flip is not None and best_score < 10:
                    u, v, action = best_flip
                    if action == 'remove':
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    # No good flip found for this K4, try random flip elsewhere
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        if (adj[u] >> v) & 1:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                continue

            # Check for I6
            i6 = find_ind6(adj, n)
            if i6 is not None:
                # I6 found: 6 vertices with no edges between them
                # To break it, add an edge between any two of them
                verts = list(i6)
                random.shuffle(verts)
                best_flip = None
                best_score = float('inf')

                for i in range(len(verts)):
                    for j in range(i + 1, len(verts)):
                        u, v = verts[i], verts[j]
                        # Add edge (u,v)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

                        k4_after = find_k4(adj, n)
                        i6_after = find_ind6(adj, n)

                        score = (1 if k4_after else 0) * 10 + (1 if i6_after else 0)

                        if score < best_score:
                            best_score = score
                            best_flip = (u, v, 'add')

                        # Restore
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)

                        if best_score == 0:
                            break
                    if best_score == 0:
                        break

                if best_flip is not None and best_score < 10:
                    u, v, action = best_flip
                    if action == 'add':
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                else:
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        if (adj[u] >> v) & 1:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                continue

            # No K4 and no I6 found - we're valid!
            return True, adj

        return False, adj

    # Start with a reasonable density for the target n
    # For K4-free + no I6, we need balance.
    # Paley graph on prime p gives good structure.
    # Try to build up from smaller valid graphs.

    best_n = 0
    best_edges = None

    # Try target sizes from large to small
    for target_n in [30, 25, 20, 17, 15, 12]:
        if time.time() > deadline - 10:
            break

        # Initialize with random graph at appropriate density
        # For n vertices, to avoid K4 and I6, density should be around 0.5
        # but slightly biased based on n
        p = 0.45

        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

        # Run greedy repair
        valid, adj = greedy_repair(adj, target_n, max_iters=30000)

        if valid:
            if target_n > best_n:
                best_n = target_n
                best_edges = to_edges(adj, target_n)
                if best_n >= 30:
                    break  # Good enough, stop searching
        else:
            # Check if we at least have a valid subgraph
            # Try shrinking: remove vertices until valid
            for try_n in range(target_n, 10, -1):
                if time.time() > deadline - 10:
                    break
                sub_adj = [0] * try_n
                for i in range(try_n):
                    for j in range(i + 1, try_n):
                        if (adj[i] >> j) & 1:
                            sub_adj[i] |= (1 << j)
                            sub_adj[j] |= (1 << i)
                valid_sub, _ = greedy_repair(sub_adj, try_n, max_iters=10000)
                if valid_sub:
                    if try_n > best_n:
                        best_n = try_n
                        best_edges = to_edges(sub_adj, try_n)
                    break

    # Fallback: Paley graph on 17 vertices (known valid)
    if best_n < 17:
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        best_n = 17
        best_edges = to_edges(adj, 17)

    return {'n': best_n, 'edges': best_edges}
