import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def popcount(x):
        return bin(x).count('1')

    def build_adj_bitset(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_triangle_free(adj, n):
        for i in range(n):
            neighbors = adj[i]
            while neighbors:
                lsb = neighbors & (-neighbors)
                j = lsb.bit_length() - 1
                neighbors ^= lsb
                common = adj[j] & adj[i] & ~(1 << i) & ~(1 << j)
                if common:
                    return False
        return True

    def has_independent_set_size(adj, n, k, time_limit=3.0):
        """Returns True if there exists an independent set of size k."""
        start_t = time.time()
        found = [False]

        # Use a recursive branch and bound with greedy coloring bound
        # We'll use a simple approach: try to find independent set of size k
        # using backtracking with pruning

        # Precompute non-neighborhoods for faster access
        non_adj = [0] * n
        for i in range(n):
            non_adj[i] = ((1 << n) - 1) & ~(1 << i) & ~adj[i]

        def dfs(candidates, current_size, path):
            if found[0]:
                return
            if time.time() - start_t > time_limit:
                return
            if current_size >= k:
                found[0] = True
                return
            if not candidates:
                return

            # Upper bound: if we can't reach k even with all remaining
            if current_size + popcount(candidates) < k:
                return

            # Greedy coloring bound for pruning
            # Simple bound: pick vertices greedily to form color classes
            remaining = candidates
            color_count = 0
            while remaining:
                color_count += 1
                # Pick a vertex with minimum degree in remaining
                best_v = -1
                best_deg = float('inf')
                temp = remaining
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    deg = popcount(adj[v] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v

                # Greedy maximal independent set
                color_class = 0
                rem = remaining
                while rem:
                    lsb = rem & (-rem)
                    v = lsb.bit_length() - 1
                    rem ^= lsb
                    color_class |= (1 << v)
                    rem &= ~adj[v] & ~(1 << v)

                remaining &= ~color_class

            if current_size + color_count < k:
                return

            # Pick vertex: use minimum degree heuristic
            best_v = -1
            best_deg = float('inf')
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = popcount(adj[v] & candidates)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v

            v = best_v
            bit_v = 1 << v
            rest = candidates & ~bit_v

            # Branch include v
            non_neighbors = rest & ~adj[v]
            dfs(non_neighbors, current_size + 1, path + [v])
            if found[0]:
                return

            # Branch exclude v
            dfs(rest, current_size, path)

        dfs((1 << n) - 1, 0, [])
        return found[0]

    def verify_graph(n, edges, time_limit=5.0):
        adj = build_adj_bitset(n, edges)
        if not is_triangle_free(adj, n):
            return False
        if has_independent_set_size(adj, n, 10, time_limit):
            return False
        return True

    # --- Known constructions ---
    def andrasfai_graph(k):
        """Andrásfai graph A_k on 3k vertices."""
        n = 3 * k
        edges = []
        for a in range(3):
            for i in range(k):
                for b in range(3):
                    if (b - a) % 3 != 1:
                        continue
                    for j in range(k):
                        if (j - i) % k in (0, 1):
                            u = a * k + i
                            v = b * k + j
                            if u < v:
                                edges.append([u, v])
        return {'n': n, 'edges': edges}

    def random_triangle_free(n, seed_val, density=0.4):
        """Generate a random triangle-free graph."""
        random.seed(seed_val)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Remove triangles
        for _ in range(200):
            if time.time() - start_time > deadline - 3:
                break
            found = False
            for i in range(n):
                ni = adj[i]
                while ni:
                    lsb1 = ni & (-ni)
                    j = lsb1.bit_length() - 1
                    ni ^= lsb1
                    common = adj[j] & adj[i] & ~(1 << i) & ~(1 << j)
                    if common:
                        lsb2 = common & (-common)
                        k = lsb2.bit_length() - 1
                        choice = random.randint(0, 2)
                        if choice == 0:
                            adj[i] &= ~(1 << j); adj[j] &= ~(1 << i)
                        elif choice == 1:
                            adj[i] &= ~(1 << k); adj[k] &= ~(1 << i)
                        else:
                            adj[j] &= ~(1 << k); adj[k] &= ~(1 << j)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return adj

    def add_edges_to_reduce_alpha(adj, n, max_iter=50):
        """Add edges to reduce independence number while maintaining triangle-free."""
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            if not has_independent_set_size(adj, n, 10, time_limit=1.5):
                break
            safe_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        if not (adj[i] & adj[j]):
                            safe_edges.append((i, j))
            random.shuffle(safe_edges)
            for i, j in safe_edges[:10]:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                break
        return adj

    # Baseline: Andrásfai A_9 has n=27, alpha=9
    best = {'n': 27, 'edges': []}
    ag = andrasfai_graph(9)
    if ag and verify_graph(ag['n'], ag['edges']):
        best = ag

    # Try to find larger valid graphs
    for target_n in range(28, 41):
        if time.time() - start_time > deadline - 10:
            break

        found = False
        for attempt in range(15):
            if time.time() - start_time > deadline - 5:
                break

            adj = random_triangle_free(target_n, attempt * 100003 + target_n * 7)
            if not is_triangle_free(adj, target_n):
                continue

            adj = add_edges_to_reduce_alpha(adj, target_n)

            if is_triangle_free(adj, target_n) and not has_independent_set_size(adj, target_n, 10, time_limit=2.0):
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                best = {'n': target_n, 'edges': edges}
                found = True
                break

        if found:
            continue
        else:
            for attempt in range(10, 25):
                if time.time() - start_time > deadline - 5:
                    break
                adj = random_triangle_free(target_n, attempt * 7919 + target_n * 31 + 42, density=0.3)
                if not is_triangle_free(adj, target_n):
                    continue
                adj = add_edges_to_reduce_alpha(adj, target_n)
                if is_triangle_free(adj, target_n) and not has_independent_set_size(adj, target_n, 10, time_limit=2.0):
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    best = {'n': target_n, 'edges': edges}
                    found = True
                    break

    return best
