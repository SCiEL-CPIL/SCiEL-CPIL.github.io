import numpy as np
import time

def priority(adjacency):
    """
    Assign priority scores to vertices for a greedy Maximum Independent Set decoder.
    Structural change: optimize the vertex PERMUTATION directly via simulated annealing
    with an efficient greedy-decoder evaluation. Scores are derived from the permutation.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]

    # Build adjacency as list of sets for fast membership, plus bitmask-free neighbor arrays
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    # Also a boolean adjacency matrix for vectorized ops if needed
    adj_bool = adj

    def greedy_count(order):
        """Simulate the greedy decoder on a given vertex order (list/array of vertex ids)."""
        active = [True] * n
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                for nb in neighbors_list[v]:
                    if active[nb]:
                        active[nb] = False
        return count

    # Initial ordering: sort by a good heuristic (low degree, high neighbor degree = pick first)
    deg = adj_bool.sum(axis=1).astype(float)
    neighbor_deg_sum = adj_bool @ deg
    safe_deg = deg + 1.0
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    init_score = -deg + 0.5 * avg_neighbor_deg
    init_order = np.argsort(-init_score, kind='stable').tolist()

    rng = np.random.default_rng(7)

    best_order = init_order[:]
    best_val = greedy_count(best_order)
    cur_order = init_order[:]
    cur_val = best_val

    # Simulated annealing on the permutation
    T = 1.0
    T_min = 1e-4
    alpha = 0.9995  # cooling rate per accepted/attempted move

    def rand_move(order, val):
        """Perform a random mutation: swap or 2-opt segment reversal. Return new order, new val."""
        move_type = rng.random()
        if move_type < 0.5:
            # Swap two positions
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                return order, val
            new_order = order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif move_type < 0.8:
            # Insert a vertex at a different position
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                return order, val
            new_order = order[:]
            elem = new_order.pop(i)
            new_order.insert(j, elem)
        else:
            # 2-opt: reverse a segment
            i = rng.integers(0, n - 1)
            j = rng.integers(0, n - 1)
            if i > j:
                i, j = j, i
            if i == j:
                return order, val
            new_order = order[:]
            new_order[i:j+1] = new_order[i:j+1][::-1]
        new_val = greedy_count(new_order)
        return new_order, new_val

    step = 0
    while time.time() < deadline:
        # Batch: do a few moves before checking time to reduce overhead
        batch_end = min(step + 50, 200000)
        for s in range(step, batch_end):
            if time.time() > deadline:
                break
            new_order, new_val = rand_move(cur_order, cur_val)
            delta = new_val - cur_val
            if delta >= 0 or rng.random() < np.exp(delta / max(T, T_min)):
                cur_order = new_order
                cur_val = new_val
                if cur_val > best_val:
                    best_val = cur_val
                    best_order = cur_order[:]
            T = max(T * alpha, T_min)
            step = s + 1
        # Occasionally reheat if stuck
        if step % 5000 == 0 and T < 0.05:
            T = 0.5

    # Derive scores from best permutation: rank-based scores (earlier = higher score)
    # The decoder picks in descending score order, so vertex at position p gets score (n - p)
    rank_scores = np.zeros(n, dtype=float)
    for pos, v in enumerate(best_order):
        rank_scores[v] = n - pos

    return rank_scores
