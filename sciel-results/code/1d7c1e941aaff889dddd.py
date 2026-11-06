import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def exact_check(n, adj):
        """Exact check: no K4 and no independent 6-set."""
        # Check K4
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    m = common
                    while m:
                        lsb = m & -m
                        a = lsb.bit_length() - 1
                        m ^= lsb
                        rest = common & ~(1 << a)
                        if adj[a] & rest:
                            return False
        # Check independent 6-set using bitset backtracking
        all_mask = (1 << n) - 1
        def find_indep_6(remaining, size):
            if size == 6:
                return True
            if remaining.bit_count() < size:
                return False
            m = remaining
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                non_n = remaining & ~(adj[u] | (1 << u))
                if non_n.bit_count() >= size - 1:
                    if find_indep_6(non_n, size - 1):
                        return True
            return False
        if find_indep_6(all_mask, 6):
            return False
        return True

    def count_indep_sets(remaining, size, adj, n):
        """Count independent sets of given size in the induced subgraph on 'remaining'."""
        if size == 0:
            return 1
        if remaining.bit_count() < size:
            return 0
        # Pick the lowest vertex
        lsb = remaining & -remaining
        u = lsb.bit_length() - 1
        rest = remaining ^ lsb
        # Don't include u
        count = count_indep_sets(rest, size, adj, n)
        # Include u: neighbors of u in remaining are excluded
        non_n = rest & ~adj[u]
        count += count_indep_sets(non_n, size - 1, adj, n)
        return count

    def delta_energy_add_edge(n, adj, u, v):
        """Energy change when ADDING edge (u,v)."""
        # Count new K4s: triangles in common neighborhood of u,v
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        tri_count = 0
        m = common
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            m ^= lsb
            rest = common & ~(1 << a)
            tri_count += (adj[a] & rest).bit_count()
        tri_count //= 3

        # Count destroyed independent 6-sets: independent 5-sets in non-neighborhood of {u,v}
        all_mask = (1 << n) - 1
        non_n = all_mask & ~(adj[u] | adj[v] | (1 << u) | (1 << v))
        i5_count = count_indep_sets(non_n, 5, adj, n)
        return tri_count - i5_count

    def delta_energy_remove_edge(n, adj, u, v):
        """Energy change when REMOVING edge (u,v)."""
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        tri_count = 0
        m = common
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            m ^= lsb
            rest = common & ~(1 << a)
            tri_count += (adj[a] & rest).bit_count()
        tri_count //= 3

        all_mask = (1 << n) - 1
        non_n = all_mask & ~(adj[u] | adj[v] | (1 << u) | (1 << v))
        i5_count = count_indep_sets(non_n, 5, adj, n)
        return -tri_count + i5_count

    def random_init(n, p=0.5):
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def simulated_annealing(n, adj, T_start, T_end, steps, deadline):
        """Run SA on the energy landscape."""
        best_energy = float('inf')
        best_adj = None

        for s in range(steps):
            if time.time() > deadline:
                break

            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u

            is_edge = bool((adj[u] >> v) & 1)

            if is_edge:
                d = delta_energy_remove_edge(n, adj, u, v)
            else:
                d = delta_energy_add_edge(n, adj, u, v)

            T = T_start * (T_end / T_start) ** (s / max(steps - 1, 1))

            if d < 0 or (T > 0 and random.random() < math.exp(-d / T)):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)

                # Quick check if this is a valid solution
                # We'll do a full check only when we think we're close
                # For now, track via energy
                # Actually let's just track best and check at the end
                pass

        # Check the final state
        if exact_check(n, adj):
            return adj, 0, True
        return adj, -1, False

    def try_sa_for_n(n, deadline):
        """Try multiple SA restarts to find a valid graph on n vertices."""
        best_result = None

        # Try different densities
        for restart in range(30):
            if time.time() > deadline - 5:
                break

            p = 0.25 + 0.5 * random.random()
            adj = random_init(n, p)

            # Run SA
            steps = 5000
            result_adj, energy, success = simulated_annealing(
                n, adj, T_start=15.0, T_end=0.01,
                steps=steps, deadline=deadline - 5
            )

            if success:
                return result_adj

        return None

    def try_backtracking(n, deadline):
        """Backtracking construction for n vertices."""
        if n > 38:
            return None

        adj = [0] * n

        def has_indep_in_set(T, adj, k, size):
            """Check if set T contains independent set of given size."""
            if T.bit_count() < size:
                return False
            def find(remaining, sz):
                if sz == 0:
                    return True
                if remaining.bit_count() < sz:
                    return False
                m = remaining
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    m ^= lsb
                    non_n = remaining & ~(adj[u] | (1 << u))
                    if non_n.bit_count() >= sz - 1:
                        if find(non_n, sz - 1):
                            return True
                return False
            return find(T, size)

        def check_K4_with_vertex(k, S, adj):
            """Check if adding vertex k with neighbor set S creates a K4."""
            for a in range(k):
                if not ((S >> a) & 1):
                    continue
                for b in range(a + 1, k):
                    if not ((S >> b) & 1):
                        continue
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b] & S
                    common &= ~(1 << a)
                    common &= ~(1 << b)
                    m = common
                    while m:
                        lsb = m & -m
                        c = lsb.bit_length() - 1
                        m ^= lsb
                        rest = common & ~(1 << c)
                        if adj[c] & rest:
                            return False
            return True

        def search(k, adj, deadline):
            if time.time() > deadline:
                return None
            if k == n:
                return adj[:]

            # Try random neighbor sets
            attempts = 100
            for _ in range(attempts):
                if time.time() > deadline:
                    return None
                p = 0.3 + 0.4 * random.random()
                S = 0
                for v in range(k):
                    if random.random() < p:
                        S |= (1 << v)

                # Check K4 constraint
                if not check_K4_with_vertex(k, S, adj):
                    continue

                # Check I6 constraint: non-neighbors must not have independent 5-set
                T = ((1 << k) - 1) & ~S
                if has_indep_in_set(T, adj, k, 5):
                    continue

                # Accept
                adj[k] = S
                for v in range(k):
                    if (S >> v) & 1:
                        adj[v] |= (1 << k)
                    else:
                        adj[v] &= ~(1 << k)

                result = search(k + 1, adj, deadline)
                if result is not None:
                    return result

                # Backtrack
                adj[k] = 0
                for v in range(k):
                    if (S >> v) & 1:
                        adj[v] &= ~(1 << k)
                    else:
                        adj[v] |= (1 << k)

            return None

        # Multiple restarts
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            random.seed(restart * 104729 + 1)
            adj = [0] * n
            result = search(0, adj, deadline - 10)
            if result is not None:
                return result
        return None

    best_graph = None
    best_n = 0

    # Strategy 1: Try backtracking for moderate n
    for target_n in [20, 25, 30]:
        if time.time() > deadline - 20:
            break
        bt_deadline = min(deadline - 15, time.time() + 10)
        result = try_backtracking(target_n, bt_deadline)
        if result is not None and exact_check(target_n, result):
            best_graph = result
            best_n = target_n

    # Strategy 2: Try SA for larger n
    if best_n < 33:
        for target_n in [35, 34, 33, 32, 31, 30]:
            if time.time() > deadline - 15:
                break
            result = try_sa_for_n(target_n, deadline - 5)
            if result is not None and exact_check(target_n, result):
                best_graph = result
                best_n = target_n
                break

    # Fallback: Paley graph of order 17 (known to be K4-free and has no independent 6-set? Let's check)
    if best_graph is None:
        # Use a smaller known valid graph as fallback
        # Paley graph of order 17
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Verify
        if exact_check(n, adj):
            return {'n': n, 'edges': get_edges(adj, n)}
        # If even that fails (shouldn't), return trivial
        return {'n': 1, 'edges': []}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
