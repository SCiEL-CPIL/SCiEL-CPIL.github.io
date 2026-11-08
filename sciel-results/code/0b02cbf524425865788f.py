"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit

    n = adjacency.shape[0]
    adjacency = adjacency.astype(bool)

    # Precompute neighbor lists for faster simulation
    neighbor_lists = [np.where(adjacency[i])[0] for i in range(n)]

    def simulate_greedy(order):
        """Simulate greedy decoder given an explicit vertex order (descending priority)."""
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                selected[neighbor_lists[v]] = True
        return count

    degrees = adjacency.sum(axis=1).astype(float)
    rng = np.random.RandomState(42)

    # Generate several initial orderings to start local search from
    initial_orders = []
    # 1. Ascending degree (low degree first)
    initial_orders.append(np.argsort(degrees))
    # 2. Descending degree (high degree first)
    initial_orders.append(np.argsort(-degrees))
    # 3. Random
    initial_orders.append(rng.permutation(n))
    # 4. Degree with small random noise for tie-breaking
    deg_noise = degrees + rng.uniform(0, 1, n) * 0.01
    initial_orders.append(np.argsort(deg_noise))
    # 5. Another random
    initial_orders.append(rng.permutation(n))

    best_order = initial_orders[0].copy()
    best_size = simulate_greedy(best_order)

    for init_order in initial_orders:
        if time.time() > deadline - 15:
            break
        curr_order = init_order.copy()
        curr_size = simulate_greedy(curr_order)

        # Phase 1: Adjacent swap local search (multiple passes until no improvement)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(n - 1):
                if time.time() > deadline - 10:
                    break
                new_order = curr_order.copy()
                new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                new_size = simulate_greedy(new_order)
                if new_size > curr_size:
                    curr_size = new_size
                    curr_order = new_order
                    improved = True

        # Phase 2: Random swap local search to escape local optima
        no_improve = 0
        while time.time() < deadline - 5 and no_improve < 300:
            i, j = rng.randint(0, n, 2)
            if i == j:
                continue
            new_order = curr_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_size = simulate_greedy(new_order)
            if new_size > curr_size:
                curr_size = new_size
                curr_order = new_order
                no_improve = 0
            else:
                no_improve += 1

        if curr_size > best_size:
            best_size = curr_size
            best_order = curr_order

    # Convert order to scores: earlier in order = higher score
    scores = np.zeros(n)
    for pos, v in enumerate(best_order):
        scores[v] = -pos

    return scores
# EVOLVE-BLOCK-END
