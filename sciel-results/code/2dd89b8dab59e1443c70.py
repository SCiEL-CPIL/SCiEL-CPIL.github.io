"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    def recompute_n1_n2_n4(adj):
        """Compute N1, N2, N4 bitsets for C6 checking."""
        N1 = adj[:]
        N2 = [0] * n
        for i in range(n):
            mask = 0
            x = N1[i]
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                mask |= N1[j]
                x ^= lsb
            N2[i] = mask

        N4 = [0] * n
        for i in range(n):
            mask = 0
            x = N2[i]
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                mask |= N2[j]
                x ^= lsb
            N4[i] = mask
        return N1, N2, N4

    def is_safe(u, v, N1, N4):
        """Check if adding edge (u,v) creates no C6."""
        return (N1[u] & N4[v]) == 0 and (N1[v] & N4[u]) == 0

    def get_edges(adj):
        edges = []
        for u in range(n):
            x = adj[u]
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                x ^= lsb
        return edges

    def count_edges(adj):
        cnt = 0
        for u in range(n):
            cnt += bin(adj[u]).count('1')
        return cnt // 2

    def make_paley_adj():
        """Create adjacency from Paley graph on 41 vertices (quadratic residues mod 41),
        then drop vertex 40 to get 40 vertices. Paley(41) is C6-free and has
        (41*20)/2 = 410 edges; dropping one vertex removes at most 20 edges,
        leaving at least 390 edges... wait, that's too many.
        
        Actually Paley(41) has 41 vertices and degree 20, so 410 edges.
        Removing vertex 40 removes its 20 incident edges, leaving 390 edges on 40 vertices.
        But we need to verify it's C6-free. Paley graphs of prime order p ≡ 1 (mod 4)
        are not necessarily C6-free. Let me check: the eigenvalues of Paley(p) are
        (p-1)/2 and (-1 ± sqrt(-p))/2 ... for p=41, the smallest eigenvalue is
        (-1 - sqrt(-41))/2 which is complex... no, for p ≡ 1 mod 4, eigenvalues are
        (p-1)/2 with multiplicity 1, and (-1 ± sqrt(p-1))/2 ... 
        
        Actually for Paley(p) with p ≡ 1 (mod 4): eigenvalues are (p-1)/2 (mult 1),
        (-1+sqrt(p-1))/2 (mult (p-1)/2), (-1-sqrt(p-1))/2 (mult (p-1)/2).
        For p=41: 20, (-1+sqrt(40))/2 ≈ (-1+6.32)/2 ≈ 2.66, (-1-sqrt(40))/2 ≈ -3.66.
        The number of C6s relates to eigenvalues. A graph with min eigenvalue >= -k
        can still have C6s. Let me just verify computationally.
        
        Actually, let me just build it and verify. If it has C6s, I'll use a different
        construction.
        """
        # Quadratic residues mod 41
        qr = set()
        for a in range(1, 41):
            qr.add((a * a) % 41)
        
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if (v - u) % 41 in qr or (u - v) % 41 in qr:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return adj

    def has_c6(adj):
        """Check if graph contains a C6 using N4 bitsets."""
        N1, N2, N4 = recompute_n1_n2_n4(adj)
        for u in range(n):
            if N1[u] & N4[u]:
                return True
        return False

    def make_seed_adj():
        """Create adjacency for 8 disjoint K5s."""
        adj = [0] * n
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return adj

    def greedy_add(adj, N1, N2, N4, order, deadline):
        """Try to add edges in given order, updating N1/N2/N4 incrementally."""
        added = 0
        for u, v in order:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if is_safe(u, v, N1, N4):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                added += 1
                N1, N2, N4 = recompute_n1_n2_n4(adj)
        return adj, N1, N2, N4, added

    def local_search_swap(adj, N1, N2, N4, deadline):
        """Try to improve by removing one edge and adding another (1-for-1 swap)."""
        edges = get_edges(adj)
        non_edges = [(u, v) for u in range(n) for v in range(u + 1, n)
                     if not (adj[u] & (1 << v))]
        best_adj = adj[:]
        best_N1, best_N2, best_N4 = N1[:], N2[:], N4[:]
        best_count = count_edges(adj)
        improved = False

        random.shuffle(edges)
        random.shuffle(non_edges)

        # Try removing each edge and see if we can add a non-edge
        for u, v in edges:
            if time.time() > deadline:
                break

            test_adj = adj[:]
            test_adj[u] &= ~(1 << v)
            test_adj[v] &= ~(1 << u)
            tN1, tN2, tN4 = recompute_n1_n2_n4(test_adj)

            # Try to add a non-edge
            for a, b in non_edges:
                if a == u and b == v:
                    continue
                if test_adj[a] & (1 << b):
                    continue
                if is_safe(a, b, tN1, tN4):
                    test_adj[a] |= 1 << b
                    test_adj[b] |= 1 << a
                    new_count = count_edges(test_adj)
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = test_adj[:]
                        best_N1, best_N2, best_N4 = tN1[:], tN2[:], tN4[:]
                        improved = True
                    break

        return best_adj, best_N1, best_N2, best_N4, best_count, improved

    def local_search_remove_add(adj, N1, N2, N4, deadline):
        """Remove one edge, then greedily add as many non-edges as possible."""
        edges = get_edges(adj)
        best_adj = adj[:]
        best_N1, best_N2, best_N4 = N1[:], N2[:], N4[:]
        best_count = count_edges(adj)

        random.shuffle(edges)
        max_removals = min(len(edges), 15)

        for idx in range(max_removals):
            if time.time() > deadline:
                break
            u, v = edges[idx]

            test_adj = adj[:]
            test_adj[u] &= ~(1 << v)
            test_adj[v] &= ~(1 << u)
            tN1, tN2, tN4 = recompute_n1_n2_n4(test_adj)

            candidates = [(a, b) for a in range(n) for b in range(a + 1, n)
                         if not (test_adj[a] & (1 << b))]
            random.shuffle(candidates)

            test_adj, tN1, tN2, tN4, _ = greedy_add(test_adj, tN1, tN2, tN4, candidates, deadline)
            new_count = count_edges(test_adj)

            if new_count > best_count:
                best_count = new_count
                best_adj = test_adj
                best_N1, best_N2, best_N4 = tN1, tN2, tN4

        return best_adj, best_N1, best_N2, best_N4, best_count

    best_edges = []
    best_count = 0

    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

    # First: try Paley construction
    paley_adj = make_paley_adj()
    paley_count = count_edges(paley_adj)
    if not has_c6(paley_adj):
        paley_N1, paley_N2, paley_N4 = recompute_n1_n2_n4(paley_adj)
        # Greedy add more edges
        candidates = [(u, v) for u, v in all_pairs if not (paley_adj[u] & (1 << v))]
        random.shuffle(candidates)
        p_adj, p_N1, p_N2, p_N4, _ = greedy_add(paley_adj, paley_N1, paley_N2, paley_N4, candidates, deadline)
        p_count = count_edges(p_adj)
        if p_count > best_count:
            best_count = p_count
            best_edges = get_edges(p_adj)

        # Local search on Paley result
        p_adj, p_N1, p_N2, p_N4, p_count = local_search_remove_add(p_adj, p_N1, p_N2, p_N4, deadline)
        if p_count > best_count:
            best_count = p_count
            best_edges = get_edges(p_adj)

    # Also try K5 seed
    k5_adj = make_seed_adj()
    k5_N1, k5_N2, k5_N4 = recompute_n1_n2_n4(k5_adj)
    candidates = [(u, v) for u, v in all_pairs if not (k5_adj[u] & (1 << v))]
    random.shuffle(candidates)
    k5_adj, k5_N1, k5_N2, k5_N4, _ = greedy_add(k5_adj, k5_N1, k5_N2, k5_N4, candidates, deadline)
    k5_count = count_edges(k5_adj)
    if k5_count > best_count:
        best_count = k5_count
        best_edges = get_edges(k5_adj)

    # Multiple improvement strategies
    while time.time() < deadline:
        strategy = random.random()

        if strategy < 0.40:
            # Strategy 1: Start from best found, do local search remove+add
            if best_edges:
                adj = [0] * n
                for u, v in best_edges:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                N1, N2, N4 = recompute_n1_n2_n4(adj)
                adj, N1, N2, N4, cnt = local_search_remove_add(adj, N1, N2, N4, deadline)
                if cnt > best_count:
                    best_count = cnt
                    best_edges = get_edges(adj)

        elif strategy < 0.65:
            # Strategy 2: Start from Paley (if C6-free) or K5, greedy with random order
            if not has_c6(make_paley_adj()):
                adj = make_paley_adj()
            else:
                adj = make_seed_adj()
            N1, N2, N4 = recompute_n1_n2_n4(adj)
            candidates = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
            random.shuffle(candidates)
            adj, N1, N2, N4, _ = greedy_add(adj, N1, N2, N4, candidates, deadline)
            cnt = count_edges(adj)
            if cnt > best_count:
                best_count = cnt
                best_edges = get_edges(adj)

        elif strategy < 0.85:
            # Strategy 3: Start from best, try 1-for-1 swaps
            if best_edges:
                adj = [0] * n
                for u, v in best_edges:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                N1, N2, N4 = recompute_n1_n2_n4(adj)
                adj, N1, N2, N4, cnt, improved = local_search_swap(adj, N1, N2, N4, deadline)
                if cnt > best_count:
                    best_count = cnt
                    best_edges = get_edges(adj)

        else:
            # Strategy 4: Perturbed Paley - remove some edges from Paley, then greedy
            if not has_c6(make_paley_adj()):
                adj = make_paley_adj()
            else:
                adj = make_seed_adj()
            # Remove 3-6 random edges
            edge_list = get_edges(adj)
            for _ in range(random.randint(3, 6)):
                if edge_list:
                    u, v = random.choice(edge_list)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edge_list = [e for e in edge_list if e[0] != u or e[1] != v]
            N1, N2, N4 = recompute_n1_n2_n4(adj)
            candidates = [(u, v) for u, v in all_pairs if not (adj[u] & (1 << v))]
            random.shuffle(candidates)
            adj, N1, N2, N4, _ = greedy_add(adj, N1, N2, N4, candidates, deadline)
            cnt = count_edges(adj)
            if cnt > best_count:
                best_count = cnt
                best_edges = get_edges(adj)

    return best_edges
# EVOLVE-BLOCK-END
