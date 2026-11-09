import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    n = 16

    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Generate all unique configurations (sets of 3 disjoint pairs on 6 vertices)
    # and compute the constraint matrix rows for each config.
    # Each constraint: sum of 8 transversal triple variables <= 7

    # Build constraint matrix as a sparse-like structure.
    # We'll collect for each config the indices of its 8 transversals.
    config_transversal_indices = []

    for v6 in itertools.combinations(range(n), 6):
        # All 15 perfect matchings on these 6 vertices
        v0, v1, v2, v3, v4, v5 = v6
        matchings = [
            ((v0, v1), (v2, v3), (v4, v5)),
            ((v0, v1), (v2, v4), (v3, v5)),
            ((v0, v1), (v2, v5), (v3, v4)),
            ((v0, v2), (v1, v3), (v4, v5)),
            ((v0, v2), (v1, v4), (v3, v5)),
            ((v0, v2), (v1, v5), (v3, v4)),
            ((v0, v3), (v1, v2), (v4, v5)),
            ((v0, v3), (v1, v4), (v2, v5)),
            ((v0, v3), (v1, v5), (v2, v4)),
            ((v0, v4), (v1, v2), (v3, v5)),
            ((v0, v4), (v1, v3), (v2, v5)),
            ((v0, v4), (v1, v5), (v2, v3)),
            ((v0, v5), (v1, v2), (v3, v4)),
            ((v0, v5), (v1, v3), (v2, v4)),
            ((v0, v5), (v1, v4), (v2, v3)),
        ]

        for m in matchings:
            p1, p2, p3 = m
            a, b = p1
            c, d = p2
            e, f = p3

            transversals = [
                (a, c, e), (a, c, f), (a, d, e), (a, d, f),
                (b, c, e), (b, c, f), (b, d, e), (b, d, f)
            ]

            indices = []
            for t in transversals:
                t_sorted = tuple(sorted(t))
                if t_sorted in triple_to_idx:
                    indices.append(triple_to_idx[t_sorted])
                else:
                    # Should not happen, but safety
                    indices.append(-1)

            if -1 not in indices:
                config_transversal_indices.append(indices)

    num_configs = len(config_transversal_indices)

    # Build the constraint matrix A (num_configs x num_triples) with 0/1 entries
    # and the RHS vector b (all 7s)
    A = np.zeros((num_configs, num_triples), dtype=np.float64)
    for cfg_idx, indices in enumerate(config_transversal_indices):
        for t_idx in indices:
            A[cfg_idx, t_idx] = 1.0

    b = np.full(num_configs, 7.0)

    # Objective: maximize sum of x_i  =>  minimize -sum(x_i)
    c = np.full(num_triples, -1.0)

    # Bounds: 0 <= x_i <= 1
    lb = np.zeros(num_triples)
    ub = np.ones(num_triples)

    # Integrate bounds into the LinearConstraint
    # A @ x <= b  and  0 <= x <= 1
    # Combine: [A; I; -I] @ x <= [b; 1; 0]
    # Actually scipy milp handles bounds separately.
    # Use: LinearConstraint(A, -np.inf, b) for the config constraints
    # Bounds(lb, ub) for the variable bounds

    constraint = LinearConstraint(A, -np.inf, b)
    bounds = Bounds(lb, ub)
    integrality = np.ones(num_triples)  # all variables are integer (binary)

    # Solve
    try:
        res = milp(
            c=c,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 130, "mip_gap": 0.01}
        )

        if res.status == 0 or res.status == 2:  # optimal or best feasible within time
            x = np.round(res.x).astype(int)
            # Filter: only include triples with x_i == 1
            selected_triples = [all_triples[i] for i in range(num_triples) if x[i] == 1]

            # Verify validity (safety check)
            # Build config counts
            config_counts = np.zeros(num_configs, dtype=int)
            for cfg_idx, indices in enumerate(config_transversal_indices):
                count = 0
                for t_idx in indices:
                    if x[t_idx] == 1:
                        count += 1
                if count > 7:
                    # Invalid! Fall back to heuristic
                    selected_triples = _fallback_heuristic()
                    return selected_triples

            if len(selected_triples) > 0:
                return selected_triples
        else:
            # No solution found, fall back
            pass

    except Exception:
        pass

    # Fallback: simple heuristic if MILP fails or times out
    return _fallback_heuristic()


def _fallback_heuristic():
    """Randomized greedy + local search fallback."""
    start_time = time.time()
    deadline = start_time + 140
    n = 16

    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute config structures
    config_transversal_indices = []
    for v6 in itertools.combinations(range(n), 6):
        v0, v1, v2, v3, v4, v5 = v6
        matchings = [
            ((v0, v1), (v2, v3), (v4, v5)),
            ((v0, v1), (v2, v4), (v3, v5)),
            ((v0, v1), (v2, v5), (v3, v4)),
            ((v0, v2), (v1, v3), (v4, v5)),
            ((v0, v2), (v1, v4), (v3, v5)),
            ((v0, v2), (v1, v5), (v3, v4)),
            ((v0, v3), (v1, v2), (v4, v5)),
            ((v0, v3), (v1, v4), (v2, v5)),
            ((v0, v3), (v1, v5), (v2, v4)),
            ((v0, v4), (v1, v2), (v3, v5)),
            ((v0, v4), (v1, v3), (v2, v5)),
            ((v0, v4), (v1, v5), (v2, v3)),
            ((v0, v5), (v1, v2), (v3, v4)),
            ((v0, v5), (v1, v3), (v2, v4)),
            ((v0, v5), (v1, v4), (v2, v3)),
        ]
        for m in matchings:
            p1, p2, p3 = m
            a, b = p1
            c, d = p2
            e, f = p3
            transversals = [
                (a, c, e), (a, c, f), (a, d, e), (a, d, f),
                (b, c, e), (b, c, f), (b, d, e), (b, d, f)
            ]
            indices = []
            for t in transversals:
                t_sorted = tuple(sorted(t))
                if t_sorted in triple_to_idx:
                    indices.append(triple_to_idx[t_sorted])
            if len(indices) == 8:
                config_transversal_indices.append(indices)

    num_configs = len(config_transversal_indices)

    # Build triple_to_configs
    triple_to_configs = [[] for _ in range(num_triples)]
    for cfg_idx, indices in enumerate(config_transversal_indices):
        for t_idx in indices:
            triple_to_configs[t_idx].append(cfg_idx)

    selected = [False] * num_triples
    config_counts = [0] * num_configs

    def is_valid_add(t_idx):
        for cfg_idx in triple_to_configs[t_idx]:
            if config_counts[cfg_idx] == 7:
                return False
        return True

    def add_triple(t_idx):
        selected[t_idx] = True
        for cfg_idx in triple_to_configs[t_idx]:
            config_counts[cfg_idx] += 1

    def remove_triple(t_idx):
        selected[t_idx] = False
        for cfg_idx in triple_to_configs[t_idx]:
            config_counts[cfg_idx] -= 1

    def try_add(t_idx):
        if is_valid_add(t_idx):
            add_triple(t_idx)
            return True
        return False

    best_system = None
    best_count = 0
    random.seed(123)

    max_runs = 50
    for run in range(max_runs):
        if time.time() > deadline - 3:
            break

        # Reset
        for i in range(num_triples):
            if selected[i]:
                remove_triple(i)
        for i in range(num_configs):
            config_counts[i] = 0

        # Randomized greedy
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            try_add(t_idx)

        current_count = sum(selected)

        # Local search: sweeps + swaps
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for t_idx in range(num_triples):
                if not selected[t_idx] and try_add(t_idx):
                    current_count += 1
                    improved = True

            if not improved:
                unselected = [i for i in range(num_triples) if not selected[i]]
                if not unselected:
                    break
                swap_found = False
                for _ in range(30):
                    if time.time() > deadline - 3:
                        break
                    t_add = random.choice(unselected)
                    if is_valid_add(t_add):
                        continue
                    blocking_configs = [cfg for cfg in triple_to_configs[t_add] if config_counts[cfg] == 7]
                    if not blocking_configs:
                        continue
                    cfg_block = random.choice(blocking_configs)
                    indices = config_transversal_indices[cfg_block]
                    selected_in_cfg = [t for t in indices if selected[t]]
                    if not selected_in_cfg:
                        continue
                    t_remove = random.choice(selected_in_cfg)
                    remove_triple(t_remove)
                    if is_valid_add(t_add):
                        add_triple(t_add)
                        improved = True
                        swap_found = True
                        break
                    else:
                        add_triple(t_remove)
                if swap_found:
                    continue
                else:
                    break

        if current_count > best_count:
            best_count = current_count
            best_system = [all_triples[i] for i in range(num_triples) if selected[i]]

    if best_system is None:
        return [[0, a, b] for a in range(1, n) for b in range(a + 1, n)]
    return best_system
