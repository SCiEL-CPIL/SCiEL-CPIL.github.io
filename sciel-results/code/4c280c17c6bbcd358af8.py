import time
import random
import math

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 145

    def get_state_stats(adj, edges):
        max_b = 0
        sum_b = 0
        for u, v in edges:
            common = adj[u] & adj[v]
            b = common.bit_count()
            if b > max_b:
                max_b = b
            sum_b += b
        return max_b, sum_b

    def solve_single_seed(seed):
        random.seed(seed)
        adj = [0] * n
        edges = set()

        # Strategy: start with K20,20 + 1 intra-partition edge
        for u in range(20):
            for v in range(20, 40):
                edges.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        u1, v1 = 0, 1
        edges.add((u1, v1))
        adj[u1] |= (1 << v1)
        adj[v1] |= (1 << u1)

        max_b, sum_b = get_state_stats(adj, edges)

        best_max = max_b
        best_sum = sum_b
        best_adj = adj[:]
        best_edges = edges.copy()

        T = 5.0
        cooling = 0.9995
        min_T = 0.01

        # Precompute all non-edges for faster sampling
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

        while time.time() < deadline - 15 and T > min_T:
            # Pick random edge to remove
            e_rem = random.choice(tuple(edges))
            u1, v1 = e_rem

            adj_backup = adj[:]
            edges_backup = edges.copy()

            # Remove
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            edges.discard(e_rem)

            # Sample non-edges to add
            best_add = None
            best_potential = 999
            # Try up to 50 random non-edges
            for _ in range(50):
                u2, v2 = random.choice(all_pairs)
                if (u2, v2) in edges:
                    continue
                potential = (adj[u2] & adj[v2]).bit_count()
                if potential < best_potential:
                    best_potential = potential
                    best_add = (u2, v2)

            if best_add is None:
                adj = adj_backup
                edges = edges_backup
                T *= cooling
                continue

            # Add
            u2, v2 = best_add
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            edges.add(best_add)

            # Evaluate
            new_max_b, new_sum_b = get_state_stats(adj, edges)

            # Accept/Reject
            accept = False
            if new_max_b < max_b:
                accept = True
            elif new_max_b == max_b and new_sum_b < sum_b:
                accept = True
            else:
                delta = (new_max_b - max_b) * 10 + (new_sum_b - sum_b)
                if delta < 0:
                    accept = True
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        accept = True

            if accept:
                max_b = new_max_b
                sum_b = new_sum_b
                if max_b < best_max or (max_b == best_max and sum_b < best_sum):
                    best_max = max_b
                    best_sum = sum_b
                    best_adj = adj[:]
                    best_edges = edges.copy()
            else:
                adj = adj_backup
                edges = edges_backup

            T *= cooling

            if best_max <= 7:
                break

        return best_max, best_sum, best_edges

    global_best_max = 999
    global_best_sum = 999
    global_best_edges = None

    for seed in range(200):
        if time.time() > deadline - 5:
            break

        max_b, sum_b, edges = solve_single_seed(seed)

        if max_b < global_best_max or (max_b == global_best_max and sum_b < global_best_sum):
            global_best_max = max_b
            global_best_sum = sum_b
            global_best_edges = edges

        if global_best_max <= 7:
            break

    if global_best_edges is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]

    return [[u, v] for (u, v) in sorted(global_best_edges)]
