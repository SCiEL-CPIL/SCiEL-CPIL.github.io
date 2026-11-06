import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 18

    # 1. Construct Paley(17) + Isolated vertex
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    adj = [0] * n
    for u in range(17):
        for v in range(u + 1, 17):
            if (v - u) % 17 in residues:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    def is_k4_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        w = (c & -c).bit_length() - 1
                        c &= c - 1
                        if adj[w] & common:
                            return False
        return True

    def max_triangle_free_size(adj, timeout=1.0):
        """Find largest induced triangle-free subgraph using branch and bound with degree ordering."""
        t0 = time.time()
        best = 0
        all_vertices = (1 << n) - 1

        # Order vertices by degree descending for better pruning
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        order_idx = [0] * n
        for i, v in enumerate(order):
            order_idx[v] = i

        def bt(idx, current_mask, current_size):
            nonlocal best
            if time.time() - t0 > timeout:
                return
            if idx == n:
                if current_size > best:
                    best = current_size
                return
            # Pruning: even if we add all remaining, can't beat best
            if current_size + (n - idx) <= best:
                return

            v = order[idx]

            # Option 1: Exclude v
            bt(idx + 1, current_mask, current_size)

            # Option 2: Include v
            if time.time() - t0 > timeout:
                return
            Nv_in_C = adj[v] & current_mask
            can_add = True
            c = Nv_in_C
            while c:
                u = (c & -c).bit_length() - 1
                c &= c - 1
                # Check if u has any neighbor in Nv_in_C (other than itself)
                if adj[u] & Nv_in_C:
                    can_add = False
                    break

            if can_add:
                bt(idx + 1, current_mask | (1 << v), current_size + 1)

        bt(0, 0, 0)
        return best

    def get_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            for v in range(u + 1, n):
                if (mask >> v) & 1:
                    edges.append([u, v])
        return edges

    # Initial evaluation
    current_adj = adj[:]
    current_score = max_triangle_free_size(current_adj, timeout=2.0)
    best_adj = current_adj[:]
    best_score = current_score

    # 2. Local Search
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    T = 5.0
    T_min = 0.05
    cooling = 0.995
    random.seed(42)

    while time.time() < deadline - 10.0:
        if T < T_min:
            break

        u, v = random.choice(all_edges)
        new_adj = current_adj[:]

        if (new_adj[u] >> v) & 1:
            # Remove edge
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            # Add edge - check K4-freedom
            common = new_adj[u] & new_adj[v]
            valid = True
            c = common
            while c:
                w = (c & -c).bit_length() - 1
                c &= c - 1
                if new_adj[w] & common:
                    valid = False
                    break
            if not valid:
                T *= cooling
                continue
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)

        # Evaluate new score
        eval_timeout = 0.08 if time.time() < deadline - 30 else 0.05
        new_score = max_triangle_free_size(new_adj, timeout=eval_timeout)

        if new_score < current_score:
            current_adj = new_adj
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj[:]
        else:
            delta = new_score - current_score
            if delta == 0:
                if random.random() < 0.3:
                    current_adj = new_adj
                    current_score = new_score
            else:
                prob = pow(2.71828, -delta / T)
                if random.random() < prob:
                    current_adj = new_adj
                    current_score = new_score

        T *= cooling

    # Final polish: try all single-edge modifications on best_adj
    if time.time() < deadline - 5.0:
        for u, v in all_edges:
            if time.time() > deadline - 5.0:
                break
            new_adj = best_adj[:]
            if (new_adj[u] >> v) & 1:
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
            else:
                common = new_adj[u] & new_adj[v]
                valid = True
                c = common
                while c:
                    w = (c & -c).bit_length() - 1
                    c &= c - 1
                    if new_adj[w] & common:
                        valid = False
                        break
                if not valid:
                    continue
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)

            new_score = max_triangle_free_size(new_adj, timeout=0.1)
            if new_score < best_score:
                best_score = new_score
                best_adj = new_adj

    return get_edges(best_adj)
