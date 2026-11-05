import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    if n == 0:
        return np.array([])

    degree = adj.sum(axis=1)

    # Initial ordering: low degree first, then high avg-neighbor-degree (hub-blocking)
    neighbor_degree_sum = adj @ degree
    avg_neighbor_degree = np.where(degree > 0, neighbor_degree_sum / np.maximum(degree, 1), 0.0)
    init_score = avg_neighbor_degree - degree
    order = np.argsort(-init_score)  # descending score

    # Helper: simulate greedy MIS given an order, return (set_size, selected_mask)
    def simulate(order_arr):
        selected = np.zeros(n, dtype=bool)
        in_set = np.zeros(n, dtype=bool)
        size = 0
        for v in order_arr:
            if selected[v]:
                continue
            in_set[v] = True
            size += 1
            # Mark neighbors as selected (blocked)
            neighbors = np.where(adj[v] > 0)[0]
            selected[neighbors] = True
        return size, in_set

    best_size, best_in_set = simulate(order)
    best_order = order.copy()

    # Local search: try swapping adjacent pairs in the order
    # If swapping v_i and v_{i+1} improves the greedy result, accept it
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            # Try swapping positions i and i+1
            order_new = best_order.copy()
            order_new[i], order_new[i + 1] = order_new[i + 1], order_new[i]
            size_new, _ = simulate(order_new)
            if size_new > best_size:
                best_order = order_new
                best_size = size_new
                improved = True
                # Re-scan from start to catch cascading improvements
                # But to save time, continue scanning from current position
                # Actually, let's just restart the pass to be safe but bounded
                # For efficiency, continue from i+1 (we already checked i-1,i)
                # Let's just continue; next outer loop will catch more

    # If we still have time, try non-adjacent swaps (larger jumps)
    if time.time() < deadline - 10:
        # Try swapping vertices at random positions (a few iterations)
        rng = np.random.default_rng(42)
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            i = rng.integers(0, n - 1)
            j = rng.integers(0, n)
            if i == j:
                continue
            order_new = best_order.copy()
            order_new[i], order_new[j] = order_new[j], order_new[i]
            size_new, _ = simulate(order_new)
            if size_new > best_size:
                best_order = order_new
                best_size = size_new

    # Final: convert the best ordering to priority scores
    # Higher score = earlier in order = higher priority
    # Assign rank-based scores: position 0 gets highest score
    ranks = np.zeros(n, dtype=np.float64)
    for pos, v in enumerate(best_order):
        ranks[v] = float(n - pos)  # earlier position -> higher rank/score

    return ranks
