import numpy as np
import time
import random

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    Strategy: Tabu Search with diversified neighborhood on priority vectors,
    initialized with CPM slack + resource-pressure weighting.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Extract instance data
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)  # (N, R)
    capacities = np.array(instance["capacities"], dtype=float)  # (R,)

    # Handle predecessors
    preds = instance.get("predecessors", [[] for _ in range(N)])
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
            preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
            preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
        normalized_preds = []
        for p in preds:
            if isinstance(p, (list, tuple, np.ndarray)):
                normalized_preds.append(list(map(int, p)))
            else:
                normalized_preds.append([])
        preds = normalized_preds
    else:
        preds = [[] for _ in range(N)]

    # Build successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)

    # Compute CPM slack
    in_degree = [len(p) for p in preds]
    q = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    indeg_temp = in_degree[:]
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            indeg_temp[v] -= 1
            if indeg_temp[v] == 0:
                q.append(v)

    ES = np.zeros(N)
    EF = np.zeros(N)
    for u in topo_order:
        max_pred_ef = 0
        for p in preds[u]:
            if EF[p] > max_pred_ef:
                max_pred_ef = EF[p]
        ES[u] = max_pred_ef
        EF[u] = ES[u] + durations[u]

    project_end = max(EF)
    LF = np.zeros(N)
    LS = np.zeros(N)
    for u in reversed(topo_order):
        min_succ_lf = project_end
        for v in succs[u]:
            if LF[v] < min_succ_lf:
                min_succ_lf = LF[v]
        LF[u] = min_succ_lf
        LS[u] = LF[u] - durations[u]

    slack = LS - ES

    # Resource pressure: how much of the capacity is consumed on average
    # Higher pressure = more constrained = higher priority
    resource_pressure = np.zeros(N)
    for r in range(demands.shape[1]):
        if capacities[r] > 0:
            resource_pressure += demands[:, r] / capacities[r]

    # Initial priorities: low slack + high resource pressure = high priority
    initial_priorities = -slack + 0.5 * resource_pressure + np.random.normal(0, 0.1, N)

    def simulate_sgs(priorities):
        """Fast Serial SGS simulation. Returns makespan."""
        scheduled = np.zeros(N, dtype=bool)
        active = []  # (finish_time, activity_index)
        remaining_res = capacities.copy()
        time_current = 0.0
        makespan = 0.0
        unsched_preds = np.array([len(p) for p in preds])
        eligible_set = set(i for i in range(N) if unsched_preds[i] == 0)

        while not np.all(scheduled):
            if not eligible_set:
                if not active:
                    break
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                still_active = []
                for f, idx in active:
                    if f > time_current:
                        still_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                        for v in succs[idx]:
                            if not scheduled[v]:
                                unsched_preds[v] -= 1
                                if unsched_preds[v] == 0:
                                    eligible_set.add(v)
                active = still_active
                continue

            elig_list = list(eligible_set)
            elig_list.sort(key=lambda x: priorities[x], reverse=True)

            to_remove = []
            any_scheduled = False
            for i in elig_list:
                if np.all(remaining_res >= demands[i]):
                    scheduled[i] = True
                    remaining_res -= demands[i]
                    finish_t = time_current + durations[i]
                    active.append((finish_t, i))
                    makespan = max(makespan, finish_t)
                    any_scheduled = True
                    to_remove.append(i)

            for i in to_remove:
                eligible_set.remove(i)

            if not any_scheduled:
                if not active:
                    break
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                still_active = []
                for f, idx in active:
                    if f > time_current:
                        still_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                        for v in succs[idx]:
                            if not scheduled[v]:
                                unsched_preds[v] -= 1
                                if unsched_preds[v] == 0:
                                    eligible_set.add(v)
                active = still_active

        return makespan

    # Tabu Search
    best_priorities = initial_priorities.copy()
    best_score = simulate_sgs(best_priorities)
    current_priorities = initial_priorities.copy()
    current_score = best_score

    # Tabu list: store (activity_index, direction) with expiry
    tabu_list = []  # (activity, direction, expiry_iter)
    tabu_tenure = 7  # Number of iterations to keep in tabu
    iteration = 0
    no_improve_count = 0
    max_no_improve = 50

    # Diversification: every X iterations without improvement, do a larger perturbation
    diversify_interval = 25

    while time.time() < deadline:
        if no_improve_count >= max_no_improve:
            # Diversification: random restart around best
            current_priorities = best_priorities + np.random.normal(0, 1.0, N)
            current_score = simulate_sgs(current_priorities)
            no_improve_count = 0
            tabu_list.clear()

        # Generate neighbor: pick 1-3 activities and adjust their priority
        num_change = random.choice([1, 2, 3])
        candidates = list(range(N))
        # Filter out tabu moves
        valid_candidates = []
        for c in candidates:
            is_tabu = any(t[0] == c and t[1] == 1 and t[2] > iteration for t in tabu_list)
            is_tabu_down = any(t[0] == c and t[1] == -1 and t[2] > iteration for t in tabu_list)
            if not (is_tabu or is_tabu_down):
                valid_candidates.append(c)
        
        if not valid_candidates:
            # All tabu, allow one
            valid_candidates = candidates

        indices = random.sample(valid_candidates, min(num_change, len(valid_candidates)))

        neighbor = current_priorities.copy()
        for idx in indices:
            # Random perturbation magnitude
            step = random.uniform(0.5, 2.0)
            direction = random.choice([-1, 1])
            neighbor[idx] += step * direction
            # Add to tabu
            tabu_list.append((idx, direction, iteration + tabu_tenure))

        neighbor_score = simulate_sgs(neighbor)

        # Accept if better, or if aspiration (beats best)
        if neighbor_score < current_score or neighbor_score < best_score:
            current_priorities = neighbor
            current_score = neighbor_score
            if current_score < best_score:
                best_score = current_score
                best_priorities = neighbor.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        iteration += 1

        # Clean expired tabu entries occasionally
        if iteration % 10 == 0:
            tabu_list = [(a, d, e) for a, d, e in tabu_list if e > iteration]

    return best_priorities
