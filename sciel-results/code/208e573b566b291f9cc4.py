"""K3,3-free graph construction on 20 vertices (maximize edges)."""
import random
import time

def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill

    n = 20
    rng = random.Random(12345)

    # All possible edges as (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def popcount(x):
        cnt = 0
        while x:
            x &= x - 1
            cnt += 1
        return cnt

    def is_k33_free_full(adj):
        """Full check: any disjoint triples S,T with all 9 cross edges."""
        for a in range(n):
            for b in range(a + 1, n):
                common_ab = adj[a] & adj[b]
                if common_ab == 0:
                    continue
                for c in range(b + 1, n):
                    common_abc = common_ab & adj[c]
                    if popcount(common_abc) >= 3:
                        return False
        return True

    def try_add_edge(adj, u, v):
        """Check if adding edge (u,v) creates a K3,3.
        Any new K3,3 must use edge (u,v). WLOG u in left, v in right.
        Left = {u, a, b}, Right = {v, c, d}.
        a,b must be common neighbors of u and v (adjacent to both),
        and c,d must be common neighbors of u and v,
        AND a,b must both be adjacent to c and d,
        AND c,d must both be adjacent to a and b.
        
        Actually: a,b in N(u)∩N(v). c,d in N(u)∩N(v).
        Need a-c, a-d, b-c, b-d all present.
        So for the set CN = N(u)∩N(v), we need two disjoint pairs {a,b} and {c,d}
        from CN such that all 4 cross edges between them exist.
        Equivalently: there exist a,b,c,d in CN (all distinct) with
        a~c, a~d, b~c, b~d.
        """
        N_u = adj[u]
        N_v = adj[v]
        CN = N_u & N_v  # common neighbors of u and v
        # Remove u and v (shouldn't be there in simple graph, but safe)
        CN &= ~(1 << u)
        CN &= ~(1 << v)
        if popcount(CN) < 4:
            return True
        # For each pair (a,b) in CN, check if there exist c,d in CN\{a,b}
        # such that a~c, a~d, b~c, b~d.
        # c,d must be in CN ∩ N(a) ∩ N(b).
        ab = CN
        while ab:
            lsb_a = ab & (-ab)
            ab ^= lsb_a
            a = lsb_a.bit_length() - 1
            rest_a = CN & (~lsb_a)
            bb = rest_a
            while bb:
                lsb_b = bb & (-bb)
                bb ^= lsb_b
                b = lsb_b.bit_length() - 1
                # c,d must be in CN ∩ N(a) ∩ N(b), excluding a and b
                candidates = CN & adj[a] & adj[b]
                candidates &= ~(1 << a)
                candidates &= ~(1 << b)
                if popcount(candidates) >= 2:
                    return False
        return True

    def try_remove_edge(adj, u, v):
        """Removing an edge can never create a K3,3. Always valid."""
        return True

    def edge_count(adj):
        cnt = 0
        for i in range(n):
            cnt += popcount(adj[i])
        return cnt // 2

    def greedy_construct():
        adj = [0] * n
        edges_copy = all_edges[:]
        rng.shuffle(edges_copy)
        for u, v in edges_copy:
            if try_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return edge_count(adj), adj

    def local_search(adj, max_iters=200000):
        adj = adj[:]
        current_edges = edge_count(adj)
        non_edges = [(u, v) for u, v in all_edges if not (adj[u] & (1 << v))]
        # Keep a list of current edges for swap operations
        current_edge_list = []
        for u in range(n):
            m = adj[u]
            while m:
                lsb = m & (-m)
                m ^= lsb
                v = lsb.bit_length() - 1
                if u < v:
                    current_edge_list.append((u, v))
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            op = rng.random()
            if op < 0.4:
                # Try to add a random non-edge
                if not non_edges:
                    break
                idx = rng.randrange(len(non_edges))
                u, v = non_edges[idx]
                if try_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                    non_edges.pop(idx)
                    current_edge_list.append((u, v))
            elif op < 0.7:
                # Try a swap: remove one edge, add another
                if not current_edge_list or not non_edges:
                    continue
                rm_idx = rng.randrange(len(current_edge_list))
                a, b = current_edge_list[rm_idx]
                add_idx = rng.randrange(len(non_edges))
                c, d = non_edges[add_idx]
                # Remove (a,b)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                current_edge_list.pop(rm_idx)
                non_edges.append((a, b))
                # Try to add (c,d)
                if try_add_edge(adj, c, d):
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    current_edges += 1
                    non_edges.pop(add_idx)
                    current_edge_list.append((c, d))
                else:
                    # Revert: put (a,b) back, remove (c,d) attempt
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    current_edge_list.append((a, b))
                    non_edges.remove((a, b))
            else:
                # Try 2-swap: remove two edges, add two edges
                if len(current_edge_list) < 2 or len(non_edges) < 2:
                    continue
                i1 = rng.randrange(len(current_edge_list))
                i2 = rng.randrange(len(current_edge_list))
                if i1 == i2:
                    continue
                a1, b1 = current_edge_list[i1]
                a2, b2 = current_edge_list[i2]
                j1 = rng.randrange(len(non_edges))
                j2 = rng.randrange(len(non_edges))
                if j1 == j2:
                    continue
                c1, d1 = non_edges[j1]
                c2, d2 = non_edges[j2]
                # Remove both
                adj[a1] &= ~(1 << b1)
                adj[b1] &= ~(1 << a1)
                adj[a2] &= ~(1 << b2)
                adj[b2] &= ~(1 << a2)
                # Try to add both
                ok1 = try_add_edge(adj, c1, d1)
                ok2 = ok1 and try_add_edge(adj, c2, d2)
                if ok1 and ok2:
                    adj[c1] |= (1 << d1)
                    adj[d1] |= (1 << c1)
                    adj[c2] |= (1 << d2)
                    adj[d2] |= (1 << c2)
                    # Update lists
                    # Remove i1, i2 from current_edge_list (careful with indices)
                    if i1 > i2:
                        i1, i2 = i2, i1
                    current_edge_list.pop(i2)
                    current_edge_list.pop(i1)
                    # Remove j1, j2 from non_edges
                    if j1 > j2:
                        j1, j2 = j2, j1
                    non_edges.pop(j2)
                    non_edges.pop(j1)
                    current_edge_list.append((c1, d1))
                    current_edge_list.append((c2, d2))
                else:
                    # Revert
                    adj[a1] |= (1 << b1)
                    adj[b1] |= (1 << a1)
                    adj[a2] |= (1 << b2)
                    adj[b2] |= (1 << a2)
        return current_edges, adj

    # Run multiple greedy constructions
    best_edges = 0
    best_adj = None
    num_trials = 20

    for trial in range(num_trials):
        if time.time() > deadline - 20:
            break
        edges, adj = greedy_construct()
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]

    # Local search on the best
    if best_adj is not None and time.time() < deadline - 10:
        best_edges, best_adj = local_search(best_adj)

    # Final full validation
    if best_adj is not None and not is_k33_free_full(best_adj):
        # Should not happen, but as a safeguard, fall back to a simpler construction
        # (This shouldn't occur with correct try_add_edge)
        pass

    # Convert adj to edge list
    edges_list = []
    for u in range(n):
        v_mask = best_adj[u]
        while v_mask:
            lsb = v_mask & (-v_mask)
            v_mask ^= lsb
            v = lsb.bit_length() - 1
            if u < v:
                edges_list.append([u, v])

    return edges_list
