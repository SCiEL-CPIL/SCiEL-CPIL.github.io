"""Combined approach: fast exhaustive search with NumPy vectorization + local search polish."""
import time
import itertools
import random
import numpy as np

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def _build_adj_matrix():
    """Build adjacency matrix as a numpy boolean array."""
    n = P['n']
    adj = np.zeros((n, n), dtype=bool)
    for a, b in P['edges']:
        adj[a, b] = True
        adj[b, a] = True
    return adj


def _distinct_degree_count_fast(subset, adj_np):
    """Count distinct degrees in induced subgraph using numpy."""
    idx = np.array(subset, dtype=np.intp)
    sub_adj = adj_np[np.ix_(idx, idx)]
    degrees = sub_adj.sum(axis=1)
    return len(set(degrees.tolist()))


def _local_search_polish(combo, adj_np, n, minv, deadline, rng):
    """Hill-climbing local search: try single swaps to improve distinct degree count."""
    combo_set = set(combo)
    current_score = _distinct_degree_count_fast(combo, adj_np)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v_in in list(combo):
            for v_out in range(n):
                if v_out in combo_set:
                    continue
                new_combo = list(combo)
                new_combo.remove(v_in)
                new_combo.append(v_out)
                new_score = _distinct_degree_count_fast(new_combo, adj_np)
                if new_score > current_score:
                    combo = new_combo
                    combo_set = set(combo)
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
    return combo, current_score


def construct():
    deadline = time.time() + 82.0  # safety margin under 90s wall-clock
    n = P['n']
    minv = P['minimum_vertices']
    adj_np = _build_adj_matrix()
    vertices = list(range(n))

    best_score = 0
    best_subset = list(range(minv))

    # Phase 1: Exhaustive search over all subsets of size >= minv
    # Enumerate by size from largest to smallest
    for size in range(n, minv - 1, -1):
        if time.time() > deadline:
            break
        # Theoretical max for this size is 'size'
        if best_score >= size:
            break
        for subset in itertools.combinations(vertices, size):
            if time.time() > deadline:
                break
            score = _distinct_degree_count_fast(subset, adj_np)
            if score > best_score:
                best_score = score
                best_subset = list(subset)
                # If we achieved theoretical max for this size, stop
                if best_score == size:
                    return [int(v) for v in best_subset]

    # Phase 2: Local search polish on the best subset found
    rng = random.Random(12345)
    polish_deadline = time.time() + 78.0  # leave 12s margin

    if time.time() < polish_deadline:
        # Try local search from the best subset
        polished, polished_score = _local_search_polish(list(best_subset), adj_np, n, minv, polish_deadline, rng)
        if polished_score > best_score:
            best_score = polished_score
            best_subset = polished

        # Try a few random restarts with local search
        for _ in range(20):
            if time.time() > polish_deadline:
                break
            random_combo = rng.sample(range(n), len(best_subset))
            polished, polished_score = _local_search_polish(random_combo, adj_np, n, minv, polish_deadline, rng)
            if polished_score > best_score:
                best_score = polished_score
                best_subset = polished

    # Ensure exact Python int types
    return [int(v) for v in best_subset]
