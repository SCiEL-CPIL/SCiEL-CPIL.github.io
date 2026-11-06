"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                    if remaining:
                        return True
        return False

    def max_triangle_free_size(adj, time_limit=3.0):
        best = [0]
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        t0 = time.time()
        timeout = [False]

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
            if current_size + (n - idx) <= best[0]:
                return

            v = order[idx]

            # Try including v first (often finds large solutions faster for pruning)
            neighbors_in_S = current_mask & adj[v]
            has_edge = False
            tmp = neighbors_in_S
            while tmp:
                bit = tmp & (-tmp)
                u = bit.bit_length() - 1
                rest = tmp & ~bit
                if (adj[u] & rest) != 0:
                    has_edge = True
                    break
                tmp = rest
            if not has_edge:
                backtrack(idx + 1, current_mask | (1 << v), current_size + 1)

            if not timeout[0]:
                backtrack(idx + 1, current_mask, current_size)

        backtrack(0, 0, 0)
        return best[0]

    def remove_k4(adj):
        for _ in range(300):
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b]
                    for c in range(b + 1, n):
                        if not ((common >> c) & 1):
                            continue
                        remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                        if remaining:
                            d_bit = remaining & (-remaining)
                            d = d_bit.bit_length() - 1
                            pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
                            x, y = pair
                            adj[x] &= ~(1 << y)
                            adj[y] &= ~(1 << x)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return not has_k4(adj)

    def make_partition_graph(parts, cross_density, within_density):
        adj = [0] * n
        num_parts = len(parts)
        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                for a in parts[i]:
                    for b in parts[j]:
                        if random.random() < cross_density:
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
        for i in range(num_parts):
            for a in parts[i]:
                for b in parts[i]:
                    if a < b and random.random() < within_density:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    random.seed(42)
    candidates = []

    # Strategy 1: Complete tripartite K_{6,6,6}
    adj1 = [0] * n
    parts3 = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts3[i]:
                for b in parts3[j]:
                    adj1[a] |= (1 << b)
                    adj1[b] |= (1 << a)
    candidates.append(adj1)

    # Strategy 2: Turan T(18,3) with internal edges
    for trial in range(30):
        if time.time() > deadline - 35:
            break
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        adj = [0] * n
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        for i in range(3):
            wd = 0.2 + random.random() * 0.5
            for a in parts[i]:
                for b in parts[i]:
                    if a < b and random.random() < wd:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 3: Random 4-partition
    for trial in range(25):
        if time.time() > deadline - 35:
            break
        vertices = list(range(n))
        random.shuffle(vertices)
        sizes = [5, 5, 4, 4]
        parts = []
        idx = 0
        for size in sizes:
            parts.append(vertices[idx:idx + size])
            idx += size
        adj = make_partition_graph(parts, 0.5 + random.random() * 0.5, random.random() * 0.35)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 4: 5-partition
    for trial in range(20):
        if time.time() > deadline - 35:
            break
        vertices = list(range(n))
        random.shuffle(vertices)
        sizes = [4, 4, 4, 3, 3]
        parts = []
        idx = 0
        for size in sizes:
            parts.append(vertices[idx:idx + size])
            idx += size
        adj = make_partition_graph(parts, 0.4 + random.random() * 0.5, random.random() * 0.25)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 5: Dense random graphs (complement of sparse)
    for trial in range(40):
        if time.time() > deadline - 35:
            break
        density = 0.35 + random.random() * 0.45
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 6: Very dense random (high density, more K4s to remove)
    for trial in range(20):
        if time.time() > deadline - 35:
            break
        density = 0.55 + random.random() * 0.3
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)

    # Evaluate candidates with short time limit
    best_adj = None
    best_score = 18
    for adj in candidates:
        if time.time() > deadline - 25:
            break
        score = max_triangle_free_size(adj, time_limit=1.5)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj, time_limit=1.5)

    # Simulated annealing with restarts
    adj = best_adj[:]
    current_score = best_score
    best_adj_sa = adj[:]
    best_score_sa = current_score

    temperature = 2.0
    cooling = 0.997
    max_iter = 12000
    restart_threshold = 800  # restart if no improvement for this many steps
    steps_since_improve = 0

    for iteration in range(max_iter):
        if time.time() > deadline - 3:
            break
        if best_score_sa <= 6:
            break

        # Restart mechanism: if stuck, perturb more aggressively
        if steps_since_improve > restart_threshold and random.random() < 0.1:
            # Flip 3-5 random edges
            for _ in range(random.randint(3, 5)):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                if (adj[i] >> j) & 1:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            if has_k4(adj):
                adj = best_adj_sa[:]
                current_score = best_score_sa
            else:
                current_score = max_triangle_free_size(adj, time_limit=1.0)
                if current_score < best_score_sa:
                    best_score_sa = current_score
                    best_adj_sa = adj[:]
                    steps_since_improve = 0
            temperature = max(temperature, 1.5)
            continue

        # Single edge flip
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

        if has_k4(adj):
            adj = old_adj
            continue

        score = max_triangle_free_size(adj, time_limit=1.0)
        delta = score - current_score

        if delta <= 0 or random.random() < min(1.0, math.exp(-delta / max(temperature, 0.01))):
            current_score = score
            if score < best_score_sa:
                best_score_sa = score
                best_adj_sa = adj[:]
                steps_since_improve = 0
            else:
                steps_since_improve += 1
        else:
            adj = old_adj
            steps_since_improve += 1

        temperature *= cooling

    return edges_from_adj(best_adj_sa)
# EVOLVE-BLOCK-END
