"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        """Check if graph contains a K4 using bitmask operations."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        return True
        return False

    def max_triangle_free_size(adj, time_limit=5.0):
        """Find largest induced triangle-free subgraph using branch and bound with pruning."""
        best = [0]
        t0 = time.time()
        timeout = [False]

        # Precompute neighbor masks
        neigh = adj[:]

        # For each vertex v, precompute conflict info:
        # conflict_lists[v] = list of (u, mask) where u is a neighbor of v,
        # and mask is the set of other neighbors w of v such that u-w is an edge.
        # If we include v in the set, we cannot have both u and w in the set.
        conflict_lists = [[] for _ in range(n)]
        for v in range(n):
            neigh_v = neigh[v]
            neighbors_v = []
            tmp = neigh_v
            while tmp:
                bit = tmp & (-tmp)
                u = bit.bit_length() - 1
                neighbors_v.append(u)
                tmp ^= bit

            for u in neighbors_v:
                # Find neighbors w of v such that u-w is an edge and w != u
                conflict_mask = neigh_v & neigh[u] & ~(1 << u) & ~(1 << v)
                if conflict_mask:
                    conflict_lists[v].append((u, conflict_mask))

        # Order vertices by degree (descending) for better pruning
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])

        # Precompute for each position in order, the mask of remaining vertices
        remaining_mask = [0] * (n + 1)
        for idx in range(n, -1, -1):
            if idx == n:
                remaining_mask[idx] = 0
            else:
                remaining_mask[idx] = remaining_mask[idx + 1] | (1 << order[idx])

        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return

            # Upper bound: current + remaining
            if current_size + (n - idx) <= best[0]:
                return

            v = order[idx]

            # Option 1: exclude v
            backtrack(idx + 1, current_mask, current_size)

            # Option 2: include v
            if not timeout[0]:
                # Check if adding v creates a triangle
                can_include = True
                for u, conflict_mask in conflict_lists[v]:
                    if (current_mask & (1 << u)) and (current_mask & conflict_mask):
                        can_include = False
                        break

                if can_include:
                    backtrack(idx + 1, current_mask | (1 << v), current_size + 1)

        backtrack(0, 0, 0)
        return best[0]

    def remove_k4_fast(adj):
        """Remove edges to make graph K4-free. Returns True if successful."""
        for _ in range(300):
            found = False
            for a in range(n):
                if found:
                    break
                for b in range(a + 1, n):
                    if found:
                        break
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b]
                    for c in range(b + 1, n):
                        if found:
                            break
                        if not ((common >> c) & 1):
                            continue
                        d_candidates = common & adj[c]
                        d_candidates &= ~((1 << (c + 1)) - 1)
                        if d_candidates:
                            d_bit = d_candidates & (-d_candidates)
                            d = d_bit.bit_length() - 1
                            # Remove a random edge from K4
                            pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
                            x, y = pair
                            adj[x] &= ~(1 << y)
                            adj[y] &= ~(1 << x)
                            found = True
            if not found:
                break
        return not has_k4(adj)

    random.seed(42)
    candidates = []

    # 1. Complete 3-partite graph K_{6,6,6}
    adj1 = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj1[a] |= (1 << b)
                    adj1[b] |= (1 << a)
    candidates.append(adj1)

    # 2. Turan graph T(18,3) with perturbations
    for trial in range(50):
        if time.time() > deadline - 40:
            break
        density = 0.5 + random.random() * 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4_fast(adj):
            candidates.append(adj)

    # 3. Structured: random 3-partite with cross edges
    for trial in range(40):
        if time.time() > deadline - 40:
            break
        vertices = list(range(n))
        random.shuffle(vertices)
        part_sizes = [6, 6, 6]
        parts = []
        idx = 0
        for size in part_sizes:
            parts.append(vertices[idx:idx + size])
            idx += size

        adj = [0] * n
        cross_density = 0.6 + random.random() * 0.4
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        if random.random() < cross_density:
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
        # Add some within-part edges
        within_density = random.random() * 0.4
        for i in range(3):
            for a in parts[i]:
                for b in parts[i]:
                    if a < b and random.random() < within_density:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        if remove_k4_fast(adj):
            candidates.append(adj)

    # 4. Complement of matching-based constructions
    for trial in range(30):
        if time.time() > deadline - 40:
            break
        adj = [0] * n
        # Start with random graph, ensure K4-free
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.55:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4_fast(adj):
            candidates.append(adj)

    # 5. More structured: graphs based on circulant patterns
    for trial in range(20):
        if time.time() > deadline - 40:
            break
        adj = [0] * n
        # Use a base circulant pattern with random modifications
        base_step = random.choice([2, 3, 4, 5])
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff in (base_step, n - base_step) or random.random() < 0.3:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4_fast(adj):
            candidates.append(adj)

    # 6. Blow-up of smaller K4-free graphs
    for trial in range(20):
        if time.time() > deadline - 40:
            break
        # Random K4-free graph on 6 vertices
        small_adj = [0] * 6
        for i in range(6):
            for j in range(i + 1, 6):
                if random.random() < 0.5:
                    small_adj[i] |= (1 << j)
                    small_adj[j] |= (1 << i)
        # Check if K4-free
        is_k4_free = True
        for a in range(6):
            for b in range(a + 1, 6):
                if not ((small_adj[a] >> b) & 1):
                    continue
                common = small_adj[a] & small_adj[b]
                for c in range(b + 1, 6):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & small_adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        is_k4_free = False
                        break
                if not is_k4_free:
                    break
            if not is_k4_free:
                break
        if not is_k4_free:
            continue

        # Blow up: each vertex i becomes vertices 3*i, 3*i+1, 3*i+2
        adj = [0] * n
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                if (small_adj[i] >> j) & 1:
                    for a in range(3):
                        for b in range(3):
                            u = 3 * i + a
                            v = 3 * j + b
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        # Add random within-blowup edges
        for i in range(6):
            for a in range(3):
                for b in range(a + 1, 3):
                    if random.random() < 0.3:
                        u = 3 * i + a
                        v = 3 * i + b
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        if remove_k4_fast(adj):
            candidates.append(adj)

    # Evaluate all candidates
    best_adj = None
    best_score = 18
    for adj in candidates:
        if time.time() > deadline - 25:
            break
        score = max_triangle_free_size(adj, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj, time_limit=2.0)

    # Local search with simulated annealing
    adj = best_adj[:]
    current_score = best_score
    best_adj_sa = adj[:]
    best_score_sa = current_score

    temperature = 1.0
    cooling = 0.995
    max_iter = 5000

    def find_k4(adj):
        """Find a K4 in the graph, return tuple (a,b,c,d) or None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        d_bit = d_candidates & (-d_candidates)
                        d = d_bit.bit_length() - 1
                        return (a, b, c, d)
        return None

    for iteration in range(max_iter):
        if time.time() > deadline - 15:
            break
        if best_score_sa <= 6:
            break

        # Random edge flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i

        old_adj = adj[:]
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)

        k4 = find_k4(adj)
        if k4 is not None:
            adj = old_adj
            continue

        score = max_triangle_free_size(adj, time_limit=1.5)
        delta = score - current_score

        accept = False
        if delta <= 0:
            accept = True
        else:
            exp_val = math.exp(-delta / max(temperature, 0.01))
            if random.random() < exp_val:
                accept = True

        if accept:
            current_score = score
            if score < best_score_sa:
                best_score_sa = score
                best_adj_sa = adj[:]
        else:
            adj = old_adj

        temperature *= cooling

    # Final evaluation with more time
    final_score = max_triangle_free_size(best_adj_sa, time_limit=20.0)

    return edges_from_adj(best_adj_sa)
