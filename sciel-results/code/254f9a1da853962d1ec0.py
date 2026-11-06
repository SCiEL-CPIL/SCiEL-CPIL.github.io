import time
import numpy as np

_START = time.perf_counter()
# Leave a safety margin below the 150s wall-clock limit.
_DEADLINE = _START + 130.0


def _remaining_time():
    return _DEADLINE - time.perf_counter()


def _greedy_density_value(capacity, weights, values):
    """Total value from the greedy value/weight-density policy.

    Vectorized with NumPy: at each step pick the feasible item with the
    maximum density (ties broken toward larger value, then larger index).
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0

    w = weights
    v = values
    n = len(w)
    # Precompute density once; it is fixed for the whole simulation.
    density = v / w
    total = 0.0
    cap = capacity
    avail = np.ones(n, dtype=bool)

    for _ in range(n):
        if cap <= 0:
            break
        # Feasible = still available AND fits in remaining capacity.
        feasible = avail & (w <= cap)
        if not feasible.any():
            break
        # Among feasible items, maximize density; break ties by value, then index.
        d = np.where(feasible, density, -1.0)
        best = int(np.argmax(d))
        if not feasible[best]:
            break
        total += float(v[best])
        cap -= float(w[best])
        avail[best] = False

    return total


def _two_step_value(capacity, weights, values):
    """Best achievable value with a 2-step lookahead.

    For each feasible candidate i, simulate the greedy continuation over the
    remaining items after taking i, and pick the i with the highest
    (value_i + continuation). This is a deeper search than 1-step lookahead.
    """
    n = len(weights)
    if n == 0:
        return -1.0

    w = weights
    v = values
    cap = capacity

    best_total = -1.0
    best_idx = -1

    for i in range(n):
        wi = float(w[i])
        if wi > cap:
            continue
        # Build the arrays for the items other than i (vectorized exclusion).
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w_rest = w[mask]
        v_rest = v[mask]
        if len(w_rest) == 0:
            future = 0.0
        else:
            future = _greedy_density_value(cap - wi, w_rest, v_rest)
        total = float(v[i]) + future
        if total > best_total:
            best_total = total
            best_idx = i

    if best_idx == -1:
        return -1.0
    return best_total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    # Convert to float arrays once for fast vectorized work.
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    cap = float(remaining_capacity)

    # Feasible items (fit in remaining capacity).
    feasible = w <= cap
    if not feasible.any():
        return -1

    # If we are running low on time, fall back to plain greedy density.
    if _remaining_time() < 0.5:
        d = np.where(feasible, v / w, -1.0)
        best = int(np.argmax(d))
        if not feasible[best]:
            return -1
        return int(best)

    # Structural core: 2-step lookahead (deeper than the 1-step baseline).
    best_total = -1.0
    best_idx = -1

    # Iterate candidates in a stable order; vectorized simulation inside.
    for i in np.where(feasible)[0]:
        if _remaining_time() < 1.0:
            # Time guard: return best found so far, else greedy fallback.
            if best_idx != -1:
                return int(best_idx)
            d = np.where(feasible, v / w, -1.0)
            cand = int(np.argmax(d))
            return int(cand) if feasible[cand] else -1

        wi = float(w[i])
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w_rest = w[mask]
        v_rest = v[mask]
        future = _greedy_density_value(cap - wi, w_rest, v_rest)
        total = float(v[i]) + future
        if total > best_total:
            best_total = total
            best_idx = int(i)

    if best_idx == -1:
        # No feasible item found (shouldn't happen given feasible.any()).
        return -1
    return int(best_idx)
