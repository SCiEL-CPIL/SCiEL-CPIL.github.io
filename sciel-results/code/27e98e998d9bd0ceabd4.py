import time
import random
from math import log, exp

def construct():
    n = 15
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill

    # Precompute all potential edges
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_pairs)  # 105

    # Precompute for each edge (u,v) the list of (s, t) pairs that would be
    # endpoints of a 3-edge path s-u-v-t (or t-v-u-s) if edge (u,v) is present.
    # A conflict between edges e1 and e2 exists if adding both creates a 3-path
    # with equal-degree endpoints.
    #
    # For edge (u, v): paths s-u-v-t where s in N(u)\{v}, t in N(v)\{u,s}
    # The "forbidden" condition is deg[s] == deg[t].
    #
    # Two edges e1=(u,v) and e2=(s,t) conflict if there's a 3-edge path using both
    # where the endpoints have equal degree.
    #
    # Actually, let's think about this differently. The constraint is:
    # For any 3-edge path s-a-b-t (s!=t), deg[s] != deg[t].
    # A 3-edge path uses 3 edges. If we have edges (s,a), (a,b), (b,t),
    # then we need deg[s] != deg[t].
    #
    # So the conflict is: for any 3 consecutive edges forming a path, the two
    # endpoint vertices must have different degrees.
    #
    # Let's precompute: for each pair of vertices (s, t) with s != t,
    # find all length-3 paths from s to t. If any such path exists in our
    # edge set, then deg[s] must != deg[t].
    #
    # A length-3 path from s to t: s-a-b-t where (s,a), (a,b), (b,t) are edges,
    # and s, a, b, t are all distinct.
    #
    # So the constraint is: for all distinct s, a, b, t with edges (s,a), (a,b), (b,t),
    # we need deg[s] != deg[t].
    #
    # This means: if there's a path of length 3 from s to t, then deg[s] != deg[t].
    #
    # The degree of a vertex depends on which edges are present, which makes this
    # circular. We need to find a set of edges and degrees such that:
    # 1. deg[v] = number of edges incident to v
    # 2. For any length-3 path s-a-b-t, deg[s] != deg[t]
    #
    # Let's use a different approach: iterate over possible degree sequences.
    # With 15 vertices, degrees range from 0 to 14.
    #
    # Actually, let me use the conflict graph approach more carefully.
    #
    # Key insight: The constraint "no 3-edge path has equal-degree endpoints"
    # can be checked given a fixed edge set. For the search, we can:
    # 1. Use bitmask for fast adjacency
    # 2. Fast validity check
    # 3. Greedy + local search (edge swaps)

    # Bitmask adjacency representation
    # adj_mask[v] is a bitmask of neighbors of v

    def check_valid(adj_mask, n):
        """Check if no simple 3-edge path has endpoints with equal degree."""
        deg = [bin(adj_mask[v]).count('1') for v in range(n)]
        # For each edge (u, v) with u < v:
        #   For each s in N(u) \ {v}:
        #     For each t in N(v) \ {u, s}:
        #       if s != t and deg[s] == deg[t]: return False
        for u in range(n):
            mask_u = adj_mask[u]
            for v in range(u + 1, n):
                if not (mask_u >> v) & 1:
                    continue
                mask_v = adj_mask[v]
                # s in N(u), s != v
                s_mask = mask_u & ~(1 << v)
                # t in N(v), t != u, t != s
                t_mask_base = mask_v & ~(1 << u)
                # For each s in s_mask:
                s = 0
                while s_mask:
                    # Get lowest set bit
                    lb = s_mask & (-s_mask)
                    s = lb.bit_length() - 1
                    s_mask ^= lb
                    # t in t_mask_base, t != s
                    t_mask = t_mask_base & ~(1 << s)
                    # Check if any t in t_mask has deg[t] == deg[s]
                    t = 0
                    while t_mask:
                        lb_t = t_mask & (-t_mask)
                        t = lb_t.bit_length() - 1
                        t_mask ^= lb_t
                        if deg[s] == deg[t]:
                            return False
        return True

    def get_degrees(adj_mask, n):
        return [bin(adj_mask[v]).count('1') for v in range(n)]

    # Start with K_{7,8} bipartite graph
    def make_bipartite(adj_mask, n, split):
        for v in range(n):
            adj_mask[v] = 0
        for i in range(split):
            for j in range(split, n):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

    def try_add_edges_greedy(adj_mask, n, deadline):
        """Greedy: try to add each missing edge if it maintains validity."""
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                if (adj_mask[i] >> j) & 1:
                    continue
                # Try adding edge (i, j)
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                if check_valid(adj_mask, n):
                    pass  # keep it
                else:
                    adj_mask[i] ^= (1 << j)
                    adj_mask[j] ^= (1 << i)
        return adj_mask

    def count_edges(adj_mask, n):
        total = 0
        for v in range(n):
            total += bin(adj_mask[v]).count('1')
        return total // 2

    # Initial: try different bipartite splits
    best_adj = [0] * n
    best_count = 0

    for split in range(3, 13):
        if time.time() > deadline - 10:
            break
        adj_mask = [0] * n
        make_bipartite(adj_mask, n, split)
        adj_mask = try_add_edges_greedy(adj_mask, n, deadline)
        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Local search: edge swaps (remove one edge, add another)
    def local_search(adj_mask, n, deadline):
        """Try edge swaps to improve edge count."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Get current edge set
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))

            random.shuffle(edges)
            random.shuffle(non_edges)

            for e1 in edges:
                if time.time() > deadline - 5:
                    break
                if improved:
                    break
                for e2 in non_edges:
                    if time.time() > deadline - 5:
                        break
                    if improved:
                        break
                    # Try swap: remove e1, add e2
                    i1, j1 = e1
                    i2, j2 = e2
                    adj_mask[i1] ^= (1 << j1)
                    adj_mask[j1] ^= (1 << i1)
                    adj_mask[i2] |= (1 << j2)
                    adj_mask[j2] |= (1 << i2)
                    if check_valid(adj_mask, n):
                        improved = True
                        # Keep the swap
                        break
                    else:
                        # Revert
                        adj_mask[i1] |= (1 << j1)
                        adj_mask[j1] |= (1 << i1)
                        adj_mask[i2] ^= (1 << j2)
                        adj_mask[j2] ^= (1 << i2)

            # Also try adding a single edge (no removal)
            if not improved and time.time() < deadline - 5:
                for e2 in non_edges:
                    if time.time() > deadline - 5:
                        break
                    i2, j2 = e2
                    adj_mask[i2] |= (1 << j2)
                    adj_mask[j2] |= (1 << i2)
                    if check_valid(adj_mask, n):
                        improved = True
                        break
                    else:
                        adj_mask[i2] ^= (1 << j2)
                        adj_mask[j2] ^= (1 << i2)

        return adj_mask

    # Run local search from best
    best_adj = local_search(best_adj, n, deadline)
    best_count = count_edges(best_adj, n)

    # Try 2-edge swaps (remove 2, add 2) for more exploration
    if time.time() < deadline - 10:
        adj_mask = list(best_adj)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))

            random.shuffle(edges)
            random.shuffle(non_edges)

            # Try removing 2 edges and adding 2 edges
            for idx1 in range(min(len(edges), 50)):
                if time.time() > deadline - 5:
                    break
                if improved:
                    break
                for idx2 in range(idx1 + 1, min(len(edges), 50)):
                    if time.time() > deadline - 5:
                        break
                    if improved:
                        break
                    e1 = edges[idx1]
                    e2 = edges[idx2]
                    # Try adding 2 non-edges
                    for idx3 in range(min(len(non_edges), 50)):
                        if time.time() > deadline - 5:
                            break
                        if improved:
                            break
                        for idx4 in range(idx3 + 1, min(len(non_edges), 50)):
                            if time.time() > deadline - 5:
                                break
                            if improved:
                                break
                            f1 = non_edges[idx3]
                            f2 = non_edges[idx4]
                            # Remove e1, e2; add f1, f2
                            for (a, b) in [e1, e2]:
                                adj_mask[a] ^= (1 << b)
                                adj_mask[b] ^= (1 << a)
                            for (a, b) in [f1, f2]:
                                adj_mask[a] |= (1 << b)
                                adj_mask[b] |= (1 << a)
                            if check_valid(adj_mask, n):
                                improved = True
                                break
                            else:
                                # Revert
                                for (a, b) in [f1, f2]:
                                    adj_mask[a] ^= (1 << b)
                                    adj_mask[b] ^= (1 << a)
                                for (a, b) in [e1, e2]:
                                    adj_mask[a] |= (1 << b)
                                    adj_mask[b] |= (1 << a)

        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Final: try to add any remaining edges greedily
    if time.time() < deadline - 2:
        adj_mask = list(best_adj)
        adj_mask = try_add_edges_greedy(adj_mask, n, deadline)
        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Convert to edge list
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([int(i), int(j)])

    return result
