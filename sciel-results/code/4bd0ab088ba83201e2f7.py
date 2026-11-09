"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque


def _simulate_station_count(instance, scores):
    """Simulate the greedy station filling and return number of stations used."""
    times = instance["times"]
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    # Build successor lists
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1

    remaining = [True] * n
    stations = 0
    current_load = 0
    remaining_count = n

    while remaining_count > 0:
        # Find eligible tasks (all predecessors completed)
        eligible = [i for i in range(n) if remaining[i] and in_degree[i] == 0]

        if not eligible:
            break

        # Sort by priority (descending)
        eligible.sort(key=lambda x: scores[x], reverse=True)

        # Greedily place tasks in current station
        placed_any = False
        for i in eligible:
            if times[i] <= cycle_time - current_load + 1e-9:
                # Place task i
                remaining[i] = False
                remaining_count -= 1
                current_load += times[i]
                placed_any = True
                # Update in-degrees of successors
                for s in successors[i]:
                    in_degree[s] -= 1

        if not placed_any:
            # Start a new station
            stations += 1
            current_load = 0

    # If we placed any tasks in the last station, count it
    if remaining_count == 0 and current_load > 0:
        stations += 1
    elif remaining_count > 0:
        # Shouldn't happen in valid instances, but handle gracefully
        stations += 1

    return stations


def _compute_features(instance):
    """Compute various features for tasks."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    # Build successor lists
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological order
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    indeg_copy = in_degree[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            indeg_copy[s] -= 1
            if indeg_copy[s] == 0:
                queue.append(s)

    # Longest path from task i to any terminal (including i's own time)
    longest_path = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        max_succ = 0
        for s in successors[i]:
            max_succ = max(max_succ, longest_path[s])
        longest_path[i] = times[i] + max_succ

    # Depth from start (longest path from any source to i)
    depth_from_start = np.zeros(n, dtype=float)
    for i in topo_order:
        max_pred = 0
        for p in preds[i]:
            max_pred = max(max_pred, depth_from_start[p])
        depth_from_start[i] = max_pred + times[i]

    # Slack: cycle_time - (depth_from_start[i] + longest_path[i] - times[i])
    slack = cycle_time - (depth_from_start + longest_path - times)

    # Immediate successor count
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Immediate predecessor count
    num_preds = np.array([len(preds[i]) for i in range(n)], dtype=float)

    # Successor depth: longest path starting from any immediate successor
    succ_depth = np.zeros(n, dtype=float)
    for i in range(n):
        if successors[i]:
            succ_depth[i] = max(longest_path[s] for s in successors[i])

    # Fit score: how well the task time fits into common remaining capacities
    # We approximate by checking how many "standard" remaining slots it fits well into
    # A task with time t fits well if t is close to cycle_time or a common divisor
    # Simple heuristic: reward times that are a good fraction of cycle_time
    fit_score = np.zeros(n, dtype=float)
    for i in range(n):
        t = times[i]
        if t <= 0:
            fit_score[i] = 0
        else:
            # Score based on how "tight" the fit is to cycle_time
            # Tasks that use a large fraction of cycle_time are good
            ratio = t / cycle_time
            fit_score[i] = ratio

    # RPW (Rank Position Weight): based on position in topological order
    # Earlier tasks in topo order get higher RPW
    rpw = np.zeros(n, dtype=float)
    for rank, i in enumerate(reversed(topo_order)):
        rpw[i] = rank

    return {
        'time': times.copy(),
        'longest_path': longest_path,
        'depth_from_start': depth_from_start,
        'slack': slack,
        'num_successors': num_successors,
        'num_preds': num_preds,
        'succ_depth': succ_depth,
        'fit_score': fit_score,
        'rpw': rpw,
    }


def _normalize_features(features_dict):
    """Normalize each feature to [0, 1]."""
    norm = {}
    for name, f in features_dict.items():
        f = np.asarray(f, dtype=float)
        f_min = f.min()
        f_max = f.max()
        if f_max > f_min + 1e-12:
            norm[name] = (f - f_min) / (f_max - f_min)
        else:
            norm[name] = np.ones(len(f)) * 0.5
    return norm


def priority(instance):
    """
    Score each assembly task using a feature-based priority with simulated annealing local search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)

    # Compute features
    features_dict = _compute_features(instance)
    feature_names = list(features_dict.keys())
    norm_features = _normalize_features(features_dict)

    # Feature matrix: n x len(feature_names)
    F = np.column_stack([norm_features[name] for name in feature_names])
    n_features = len(feature_names)

    # Initial: use longest_path (critical path) as primary
    init_scores = features_dict['longest_path'].copy()
    best_scores = init_scores.copy()
    best_stations = _simulate_station_count(instance, best_scores)

    if time.time() > deadline - 20:
        return best_scores

    # Simulated annealing over weight vectors
    random.seed(42)
    np.random.seed(42)

    # Start with weights emphasizing longest_path, time, and fit_score
    w0 = np.zeros(n_features)
    idx_lp = feature_names.index('longest_path')
    idx_time = feature_names.index('time')
    idx_fit = feature_names.index('fit_score')
    idx_num_succ = feature_names.index('num_successors')
    idx_depth = feature_names.index('depth_from_start')
    idx_slack = feature_names.index('slack')
    idx_rpw = feature_names.index('rpw')

    w0[idx_lp] = 0.30
    w0[idx_time] = 0.20
    w0[idx_fit] = 0.20
    w0[idx_num_succ] = 0.10
    w0[idx_depth] = 0.10
    w0[idx_slack] = -0.10  # Lower slack = higher priority
    w0[idx_rpw] = 0.10

    w_sum = w0.sum()
    if abs(w_sum) > 1e-12:
        w0 = w0 / w_sum
    else:
        w0 = np.ones(n_features) / n_features

    weights = w0.copy()
    scores = F @ weights

    current_stations = _simulate_station_count(instance, scores)
    if current_stations < best_stations:
        best_stations = current_stations
        best_scores = scores.copy()

    # Simulated annealing parameters
    T_start = 0.5
    T_end = 0.01
    max_iter = 8000

    # Check time budget
    elapsed = time.time() - start_time
    time_per_iter = min((deadline - 5 - elapsed) / max_iter, 0.01)

    for it in range(max_iter):
        if time.time() > deadline - 5:
            break

        # Temperature schedule
        progress = it / max_iter
        T = T_start * (T_end / T_start) ** progress

        # Perturb weights
        new_weights = weights.copy()
        idx = random.randint(0, n_features - 1)
        delta = random.gauss(0, 0.15)
        new_weights[idx] += delta

        # Optionally perturb another weight
        if random.random() < 0.3:
            idx2 = random.randint(0, n_features - 1)
            new_weights[idx2] += random.gauss(0, 0.1)

        # Clip weights to reasonable range
        new_weights = np.clip(new_weights, -1.0, 2.0)

        # Compute new scores
        new_scores = F @ new_weights

        # Evaluate
        new_stations = _simulate_station_count(instance, new_scores)

        # Accept/reject
        if new_stations < current_stations:
            # Better: always accept
            weights = new_weights
            scores = new_scores
            current_stations = new_stations
            if new_stations < best_stations:
                best_stations = new_stations
                best_scores = scores.copy()
        else:
            # Worse or equal: accept with probability exp(-delta/T)
            delta_obj = new_stations - current_stations
            if delta_obj > 0 and T > 0:
                prob = np.exp(-delta_obj / T)
                if random.random() < prob:
                    weights = new_weights
                    scores = new_scores
                    current_stations = new_stations

    # Final local polish: fine-tune around best
    if time.time() < deadline - 10:
        # Use best_scores as starting point for fine search
        # Convert best_scores back to approximate weights via least squares
        try:
            weights_fine, _, _, _ = np.linalg.lstsq(F, best_scores, rcond=None)
            weights_fine = np.clip(weights_fine, -1.0, 2.0)
        except:
            weights_fine = weights.copy()

        current_stations = best_stations
        scores = best_scores.copy()

        T_fine_start = 0.05
        T_fine_end = 0.001
        max_fine_iter = 3000

        for it in range(max_fine_iter):
            if time.time() > deadline - 3:
                break

            progress = it / max_fine_iter
            T = T_fine_start * (T_fine_end / T_fine_start) ** progress

            new_weights = weights_fine.copy()
            idx = random.randint(0, n_features - 1)
            new_weights[idx] += random.gauss(0, 0.05)

            if random.random() < 0.2:
                idx2 = random.randint(0, n_features - 1)
                new_weights[idx2] += random.gauss(0, 0.03)

            new_weights = np.clip(new_weights, -1.0, 2.0)
            new_scores = F @ new_weights
            new_stations = _simulate_station_count(instance, new_scores)

            if new_stations < current_stations:
                weights_fine = new_weights
                scores = new_scores
                current_stations = new_stations
                if new_stations < best_stations:
                    best_stations = new_stations
                    best_scores = scores.copy()
            else:
                delta_obj = new_stations - current_stations
                if delta_obj > 0 and T > 0:
                    prob = np.exp(-delta_obj / T)
                    if random.random() < prob:
                        weights_fine = new_weights
                        scores = new_scores
                        current_stations = new_stations

    return best_scores
# EVOLVE-BLOCK-END
