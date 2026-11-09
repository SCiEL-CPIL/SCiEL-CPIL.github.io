import numpy as np
import time
import heapq

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    n = adjacency.shape[0]
    if n == 0:
        return np.array([])

    # Precompute neighbor lists
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = adjacency.sum(axis=1).astype(float)

    # Compute Core Numbers (k-core decomposition)
    def compute_core_numbers(adj, degs):
        n = len(degs)
        core = np.zeros(n, dtype=int)
        remaining_deg = degs.astype(int).copy()
        removed = np.zeros(n, dtype=bool)
        heap = [(int(remaining_deg[i]), i) for i in range(n)]
        heapq.heapify(heap)

        while heap:
            d, v = heapq.heappop(heap)
            if removed[v]:
                continue
            core[v] = d
            removed[v] = True
            for u in neighbors[v]:
                if not removed[u]:
                    remaining_deg[u] -= 1
                    heapq.heappush(heap, (int(remaining_deg[u]), u))

        return core

    core_numbers = compute_core_numbers(adjacency, degrees)

    # Fast greedy MIS evaluation with delta computation for local search
    def greedy_mis_size(order):
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                if neighbors[v].size > 0:
                    selected[neighbors[v]] = True
        return count

    def greedy_mis_size_delta(order, i, j):
        """Compute the change in MIS size when swapping positions i and j in order."""
        # Quick check: if both vertices have same neighbors or are isolated, delta is 0
        # Full recomputation is expensive, so we do a targeted check
        v1, v2 = order[i], order[j]
        
        # If they're in the same "layer" of the greedy process, swap doesn't change outcome
        # Simple heuristic: if both are selected or both are skipped in current order, delta=0
        selected = np.zeros(n, dtype=bool)
        count = 0
        for k, v in enumerate(order):
            if k == i:
                v = v2
            elif k == j:
                v = v1
            if not selected[v]:
                selected[v] = True
                count += 1
                if neighbors[v].size > 0:
                    selected[neighbors[v]] = True
        return count

    rng = np.random.default_rng(42)

    best_size = 0
    best_order = None

    def update_best(order, size):
        nonlocal best_size, best_order
        if size > best_size:
            best_size = size
            best_order = order[:]

    # Strategy 1: Min-Degree Greedy Simulation
    # This simulates the greedy MIS process (picking min degree available vertex)
    # and assigns higher priority to vertices picked earlier.
    def compute_min_deg_layer_scores():
        remaining = np.ones(n, dtype=bool)
        remaining_deg = degrees.copy()
        scores = np.zeros(n)
        layer = 0
        max_layer = n

        while np.sum(remaining) > 0 and layer < max_layer:
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break

            rem_deg = remaining_deg[rem_indices]
            best_local = np.argmin(rem_deg)
            best_v = rem_indices[best_local]

            to_remove = set([best_v])
            for u in neighbors[best_v]:
                if remaining[u]:
                    to_remove.add(u)

            for v in to_remove:
                if remaining[v]:
                    scores[v] = max_layer - layer
                    remaining[v] = False
                    for u in neighbors[v]:
                        if remaining[u]:
                            remaining_deg[u] -= 1

            layer += 1

        return scores

    # Strategy 2: Max-Degree Greedy Simulation
    def compute_max_deg_layer_scores():
        remaining = np.ones(n, dtype=bool)
        remaining_deg = degrees.copy()
        scores = np.zeros(n)
        layer = 0
        max_layer = n

        while np.sum(remaining) > 0 and layer < max_layer:
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break

            rem_deg = remaining_deg[rem_indices]
            best_local = np.argmax(rem_deg)
            best_v = rem_indices[best_local]

            to_remove = set([best_v])
            for u in neighbors[best_v]:
                if remaining[u]:
                    to_remove.add(u)

            for v in to_remove:
                if remaining[v]:
                    scores[v] = max_layer - layer
                    remaining[v] = False
                    for u in neighbors[v]:
                        if remaining[u]:
                            remaining_deg[u] -= 1

            layer += 1

        return scores

    # Strategy 3: Independent Potential (max degree in remaining graph)
    def compute_independent_potential_scores():
        remaining = np.ones(n, dtype=bool)
        remaining_deg = degrees.copy().astype(int)
        scores = np.zeros(n)
        layer = 0
        max_layer = n

        while np.sum(remaining) > 0 and layer < max_layer:
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break

            # Select vertex with max degree in remaining graph
            rem_deg = remaining_deg[rem_indices]
            best_local = np.argmax(rem_deg)
            best_v = rem_indices[best_local]

            # Remove best_v and all its remaining neighbors
            to_remove = [best_v]
            for u in neighbors[best_v]:
                if remaining[u]:
                    to_remove.append(u)

            # Assign scores: the selected vertex gets high score, removed neighbors get lower
            scores[best_v] = max_layer - layer
            for u in to_remove:
                if u != best_v and remaining[u]:
                    scores[u] = (max_layer - layer) * 0.5

            for v in to_remove:
                if remaining[v]:
                    remaining[v] = False
                    for u in neighbors[v]:
                        if remaining[u]:
                            remaining_deg[u] -= 1

            layer += 1

        return scores

    # Generate initial orderings
    initial_orders = []

    # 1. Min-Degree Layer Scores (Primary)
    min_layer_scores = compute_min_deg_layer_scores()
    initial_orders.append(np.argsort(-min_layer_scores))

    # 2. Max-Degree Layer Scores
    max_layer_scores = compute_max_deg_layer_scores()
    initial_orders.append(np.argsort(-max_layer_scores))

    # 3. Min Degree (Static)
    initial_orders.append(np.argsort(degrees))

    # 4. Min Core Number
    initial_orders.append(np.argsort(core_numbers))

    # 5. Min (Core Number, Degree)
    score_cd = core_numbers * n + degrees
    initial_orders.append(np.argsort(score_cd))

    # 6. Random
    initial_orders.append(rng.permutation(n))

    # 7. Noisy Min Degree
    noisy_deg = degrees + rng.normal(0, 1, n)
    initial_orders.append(np.argsort(noisy_deg))

    # 8. Noisy Min Core
    noisy_core = core_numbers + rng.normal(0, 1, n)
    initial_orders.append(np.argsort(noisy_core))

    # 9. Combination: Min Layer + Core + Degree
    combo_score = min_layer_scores * 10 + core_numbers * 5 + (n - degrees)
    initial_orders.append(np.argsort(-combo_score))

    # 10. Noisy Min Layer Scores
    noisy_min_layer = min_layer_scores + rng.normal(0, 10, n)
    initial_orders.append(np.argsort(-noisy_min_layer))

    # 11. Noisy Max Degree (from inspiration)
    noisy_max_deg = degrees + rng.normal(0, 1, n)
    initial_orders.append(np.argsort(-noisy_max_deg))

    # 12. Noisy Min Degree (from inspiration)
    noisy_min_deg = -degrees + rng.normal(0, 1, n)
    initial_orders.append(np.argsort(noisy_min_deg))

    # 13. Another random
    initial_orders.append(rng.permutation(n))

    # 14. Max Degree (Static)
    initial_orders.append(np.argsort(-degrees))

    # 15. Independent Potential Scores
    indep_pot_scores = compute_independent_potential_scores()
    initial_orders.append(np.argsort(-indep_pot_scores))

    # 16. Combination of Min Layer and Independent Potential
    combo_ip = min_layer_scores * 8 + indep_pot_scores * 6 + (n - degrees) * 0.5
    initial_orders.append(np.argsort(-combo_ip))

    # Evaluate initial orders
    for order in initial_orders:
        size = greedy_mis_size(order)
        update_best(order, size)

    if time.time() > deadline - 5:
        # Not enough time for SA, return best found
        if best_order is None:
            scores = np.zeros(n)
            order = np.argsort(degrees)
            for rank, v in enumerate(order):
                scores[v] = n - rank
            return scores
        scores = np.zeros(n)
        for rank, v in enumerate(best_order):
            scores[v] = n - rank
        return scores

    # Local Search: Simulated Annealing
    # Pick top 4 distinct-ish initial orders for SA
    evaluated = [(greedy_mis_size(o), o) for o in initial_orders]
    evaluated.sort(key=lambda x: -x[0])

    sa_candidates = []
    seen_hashes = set()
    for size, order in evaluated:
        h = tuple(order[:10])
        if h not in seen_hashes:
            sa_candidates.append(order)
            seen_hashes.add(h)
        if len(sa_candidates) >= 4:
            break

    if not sa_candidates:
        sa_candidates = [initial_orders[0]]

    steps_per_restart = 12000  # Adjusted for safety

    for r, start_order in enumerate(sa_candidates):
        if time.time() > deadline - 10:
            break

        order = start_order.copy()
        current_size = greedy_mis_size(order)
        update_best(order, current_size)

        # Simulated Annealing with adaptive neighborhood
        initial_temp = 2.0
        cooling_rate = 0.9997  # Slower cooling for more exploration

        temperature = initial_temp
        no_improve = 0
        max_no_improve = 500

        for step in range(steps_per_restart):
            if time.time() > deadline - 2:
                break

            temperature *= cooling_rate

            # Adaptive step size: smaller steps as we cool
            if temperature > 1.0:
                # Large moves: swap distant positions
                i, j = rng.integers(0, n, 2)
            else:
                # Small moves: swap nearby positions
                i = rng.integers(0, n)
                offset = rng.integers(1, max(2, n // 10))
                j = (i + offset) % n

            if i == j:
                continue

            order[i], order[j] = order[j], order[i]

            new_size = greedy_mis_size(order)

            delta = new_size - current_size

            if delta >= 0:
                current_size = new_size
                update_best(order, current_size)
                no_improve = 0
            else:
                if temperature > 1e-6:
                    prob = np.exp(delta / temperature)
                    if rng.random() < prob:
                        current_size = new_size
                        no_improve += 1
                    else:
                        order[i], order[j] = order[j], order[i]
                        no_improve += 1
                else:
                    order[i], order[j] = order[j], order[i]
                    no_improve += 1

            # Restart if stuck
            if no_improve > max_no_improve:
                no_improve = 0
                temperature = max(temperature, 0.5)
                # Try a larger perturbation
                for _ in range(5):
                    i, j = rng.integers(0, n, 2)
                    if i != j:
                        order[i], order[j] = order[j], order[i]
                current_size = greedy_mis_size(order)

    # Convert best_order to scores
    if best_order is None:
        scores = np.zeros(n)
        order = np.argsort(degrees)
        for rank, v in enumerate(order):
            scores[v] = n - rank
        return scores

    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank

    return scores
