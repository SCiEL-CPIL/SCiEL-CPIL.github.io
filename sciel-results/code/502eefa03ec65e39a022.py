import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 118  # safety margin under 120s
    rng = random.Random(7331)

    def bit_count(x):
        return bin(x).count("1")

    def make_adj(n, edge_list=None):
        """Return adjacency bitmasks list of length n."""
        A = [0] * n
        if edge_list:
            for u, v in edge_list:
                A[u] |= (1 << v)
                A[v] |= (1 << u)
        return A

    def count_k4(n, A):
        """Count all K4s in the graph using bitmasks. Fast for n<=40."""
        cnt = 0
        for a in range(n):
            for b in range(a + 1, n):
                if not (A[b] >> a) & 1:
                    continue
                # Common neighbors of a and b, both > b to avoid double count
                common = A[a] & A[b] & ~((1 << (b + 1)) - 1)
                # For each pair (c,d) in common with c < d, check if c-d edge exists
                c = common
                while c:
                    low = c & -c
                    ci = low.bit_length() - 1
                    c &= ~low
                    # d must be > ci and in common
                    d_mask = common & ~((1 << (ci + 1)) - 1)
                    # Check edges from ci to each d in d_mask
                    adj_ci = A[ci]
                    while d_mask:
                        dl = d_mask & -d_mask
                        di = dl.bit_length() - 1
                        d_mask &= ~dl
                        if (adj_ci >> di) & 1:
                            cnt += 1
        return cnt

    def count_ind6(n, A):
        """Count independent 6-sets by counting K6 in complement."""
        comp = [0] * n
        full = (1 << n) - 1
        for i in range(n):
            comp[i] = full & ~(A[i] | (1 << i))
        # Count K6 in complement graph
        cnt = 0
        for a in range(n):
            for b in range(a + 1, n):
                if not (comp[b] >> a) & 1:
                    continue
                common = comp[a] & comp[b] & ~((1 << (b + 1)) - 1)
                c = common
                while c:
                    low = c & -c
                    ci = low.bit_length() - 1
                    c &= ~low
                    d_mask = common & ~((1 << (ci + 1)) - 1)
                    adj_ci = comp[ci]
                    e_mask = d_mask
                    while e_mask:
                        el = e_mask & -e_mask
                        ei = el.bit_length() - 1
                        e_mask &= ~el
                        if not (adj_ci >> ei) & 1:
                            continue
                        # Now need f > ei in common, adjacent to both ci and ei in comp
                        f_mask = common & ~((1 << (ei + 1)) - 1)
                        f_mask &= comp[ci] & comp[ei]
                        cnt += bit_count(f_mask)
        return cnt

    def energy(n, A):
        """Total violations: K4 count + independent 6-set count."""
        return count_k4(n, A) + count_ind6(n, A)

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    def random_graph(n, p=0.5):
        A = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    A[i] |= (1 << j)
                    A[j] |= (1 << i)
        return A

    def flip_edges(A, n, k):
        """Flip k random edges. Returns new A."""
        new_A = list(A)
        for _ in range(k):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            if (new_A[v] >> u) & 1:
                new_A[u] &= ~(1 << v)
                new_A[v] &= ~(1 << u)
            else:
                new_A[u] |= (1 << v)
                new_A[v] |= (1 << u)
        return new_A

    def add_vertex(A, n, p=0.5):
        """Add a new vertex with random connections. Returns (new_A, n+1)."""
        new_n = n + 1
        new_A = list(A) + [0]
        for i in range(n):
            if rng.random() < p:
                new_A[i] |= (1 << new_n - 1)
                new_A[new_n - 1] |= (1 << i)
        return new_A, new_n

    # Simulated annealing
    def simulated_annealing(start_n, start_A, max_iter, t_start, t_end, rng):
        cur_A = list(start_A)
        cur_n = start_n
        cur_e = energy(cur_n, cur_A)
        best_A = list(cur_A)
        best_n = cur_n
        best_e = cur_e

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            t = t_start * (t_end / t_start) ** (it / max(1, max_iter - 1))

            # Choose move type
            r = rng.random()
            if r < 0.05 and cur_n < 40:
                # Try adding a vertex
                cand_A, cand_n = add_vertex(cur_A, cur_n, p=0.5)
            elif r < 0.10 and cur_n > 10:
                # Try removing a vertex (rare)
                rem = rng.randrange(cur_n)
                cand_n = cur_n - 1
                cand_A = []
                for i in range(cur_n):
                    if i == rem:
                        continue
                    new_mask = cur_A[i] & ~((1 << rem) | (1 << cur_n))
                    # Remap bits above rem
                    new_mask = (new_mask & ((1 << rem) - 1)) | ((new_mask >> 1) & ~((1 << rem) - 1))
                    cand_A.append(new_mask)
            else:
                # Flip 1-4 random edges
                k = rng.randint(1, 4)
                cand_A = flip_edges(cur_A, cur_n, k)
                cand_n = cur_n

            cand_e = energy(cand_n, cand_A)

            if cand_e <= cur_e:
                cur_A = cand_A
                cur_n = cand_n
                cur_e = cand_e
                if cur_e < best_e:
                    best_e = cur_e
                    best_A = list(cur_A)
                    best_n = cur_n
            else:
                # Accept with probability exp(-(delta)/t)
                delta = cand_e - cur_e
                if t > 0 and rng.random() < pow(2.718281828, -delta / max(t, 1e-10)):
                    cur_A = cand_A
                    cur_n = cand_n
                    cur_e = cand_e

        return best_n, best_A, best_e

    # Multi-restart SL
    best_n = 1
    best_A = [0]
    best_e = float('inf')

    # Initial seeds: Paley(17), random graphs
    seeds = []
    # Paley(17)
    q = 17
    squares = {x * x % q for x in range(1, q)}
    A_paley = [0] * q
    for i in range(q):
        for j in range(q):
            if i != j and (j - i) % q in squares:
                A_paley[i] |= (1 << j)
    seeds.append((q, A_paley))

    # Random graphs of various sizes
    for n_try in [20, 25, 30, 35]:
        for _ in range(2):
            A_rand = random_graph(n_try, p=0.5)
            seeds.append((n_try, A_rand))

    # Run SA on each seed
    for seed_n, seed_A in seeds:
        if time.time() > deadline - 20:
            break
        max_iter = 3000
        t_start = 5.0
        t_end = 0.01
        bn, bA, be = simulated_annealing(seed_n, seed_A, max_iter, t_start, t_end, rng)
        if be < best_e:
            best_e = be
            best_n = bn
            best_A = bA

    # If we found a valid graph (energy 0), try to grow it
    if best_e == 0 and best_n < 40:
        # Try adding vertices one at a time with SA refinement
        cur_n = best_n
        cur_A = list(best_A)
        while cur_n < 40 and time.time() < deadline - 10:
            # Try many random additions
            improved = False
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                cand_A, cand_n = add_vertex(cur_A, cur_n, p=rng.uniform(0.3, 0.7))
                # Quick energy check
                ce = energy(cand_n, cand_A)
                if ce == 0:
                    cur_A = cand_A
                    cur_n = cand_n
                    improved = True
                    break
                elif ce <= 2:
                    # Refine with short SA
                    rn, rA, re = simulated_annealing(cand_n, cand_A, 500, 2.0, 0.01, rng)
                    if re == 0:
                        cur_A = rA
                        cur_n = rn
                        improved = True
                        break
            if not improved:
                # Try edge flips on current graph then re-add
                for _ in range(20):
                    if time.time() > deadline - 10:
                        break
                    perturbed = flip_edges(cur_A, cur_n, rng.randint(1, 3))
                    pn, pA, pe = simulated_annealing(cur_n, perturbed, 300, 1.0, 0.01, rng)
                    if pe == 0:
                        cur_A = pA
                        cur_n = pn
                        break
                    # Try adding vertex to perturbed
                    cand_A, cand_n = add_vertex(pA, pn, p=0.5)
                    ce = energy(cand_n, cand_A)
                    if ce <= 1:
                        rn, rA, re = simulated_annealing(cand_n, cand_A, 500, 2.0, 0.01, rng)
                        if re == 0:
                            cur_A = rA
                            cur_n = rn
                            improved = True
                            break
                if not improved:
                    break
        if cur_n > best_n:
            best_n = cur_n
            best_A = cur_A

    # Final: ensure validity
    final_e = energy(best_n, best_A)
    if final_e != 0:
        # Try one more SA round on the best
        if time.time() < deadline - 5:
            bn, bA, be = simulated_annealing(best_n, best_A, 2000, 3.0, 0.005, rng)
            if be == 0:
                best_n = bn
                best_A = bA
            elif be < final_e:
                best_n = bn
                best_A = bA
            # If still invalid, fall back to a known valid small graph
            if energy(best_n, best_A) != 0:
                # Use Paley(17) as fallback
                best_n = 17
                best_A = A_paley

    return {'n': best_n, 'edges': to_edges(best_n, best_A)}
# EVOLVE-BLOCK-END
