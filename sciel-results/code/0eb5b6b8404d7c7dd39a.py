import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave margin for evaluation overhead

    # Fallback: Andrasfai graph And(9) on 26 vertices
    # Triangle-free, alpha = 9
    best_result = {
        'n': 26,
        'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    }

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def check_valid(adj, n, time_budget=0.5):
        """Check if graph is triangle-free (assumed) and has no independent set of size 10.
        Returns True if valid (alpha < 10)."""
        all_mask = (1 << n) - 1
        non_neighbors = [0] * n
        for i in range(n):
            non_neighbors[i] = (all_mask ^ (1 << i)) & ~adj[i]

        # Branch and bound for independent set of size 10
        def dfs(mask, size):
            if size == 10:
                return True
            if mask == 0:
                return False
            if mask.bit_count() < 10 - size:
                return False

            # Heuristic: pick vertex with minimum non-neighbor count in mask
            min_count = n + 1
            best_v = -1
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                cnt = (non_neighbors[v] & mask).bit_count()
                if cnt < min_count:
                    min_count = cnt
                    best_v = v
                    if cnt == 0:
                        break
                temp_mask ^= lsb

            if best_v == -1:
                return False

            v_bit = 1 << best_v

            # Include best_v: new candidates are its non-neighbors in mask
            new_mask_include = non_neighbors[best_v] & mask
            if dfs(new_mask_include, size + 1):
                return True

            # Exclude best_v
            new_mask_exclude = mask & ~v_bit
            if dfs(new_mask_exclude, size):
                return True

            return False

        t0 = time.time()
        result = dfs(all_mask, 0)
        return not result

    # Strategy 1: Try deterministic polarity graph construction
    def construct_polarity_graph(q):
        """Construct the polarity graph of PG(2,q). Returns (n, adj) or None if n > 40."""
        n = q * q + q + 1
        if n > 40 or n < 2:
            return None, None

        points = []
        seen = set()
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        rep = (1, y * pow(x, q - 2, q) % q, z * pow(x, q - 2, q) % q)
                    elif y != 0:
                        rep = (0, 1, z * pow(y, q - 2, q) % q)
                    else:
                        rep = (0, 0, 1)
                    if rep not in seen:
                        seen.add(rep)
                        points.append(rep)

        n_actual = len(points)
        if n_actual != n:
            n = n_actual

        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                dot = (points[i][0] * points[j][0] +
                       points[i][1] * points[j][1] +
                       points[i][2] * points[j][2]) % q
                if dot == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

        return n, adj

    for q in [5, 4, 3, 2]:
        if time.time() > deadline - 5:
            break
        n_pol, adj_pol = construct_polarity_graph(q)
        if n_pol is not None and n_pol > best_result['n']:
            if check_valid(adj_pol, n_pol, time_budget=2.0):
                best_result = {'n': n_pol, 'edges': edges_from_adj(adj_pol, n_pol)}

    # Strategy 2: Random search for larger graphs
    # Try n from 40 down to best_result['n'] + 1
    for n in range(40, best_result['n'], -1):
        if time.time() > deadline - 10:
            break

        restarts = 0
        while time.time() < deadline - 5:
            restarts += 1
            if restarts > 5000:
                break

            # Generate a random maximal triangle-free graph
            adj = [0] * n
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)

            for u, v in pairs:
                common = adj[u] & adj[v]
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            if check_valid(adj, n, time_budget=0.3):
                best_result = {'n': n, 'edges': edges_from_adj(adj, n)}
                break  # Found valid graph for this n, move to next (smaller) n

    return best_result
