"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 150.0
_START = _time.monotonic()
_MARGIN = 2.0


def _elapsed():
    return _time.monotonic() - _START


def _safe_return(idx):
    if _elapsed() > _DEADLINE - _MARGIN:
        return int(0)
    return int(idx)


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Check deadline
    if _elapsed() > _DEADLINE - _MARGIN:
        return 0

    # Gather machine options and durations for each ready op
    # For each op, we consider each machine option.
    # We want to pick (op, machine) that minimizes the resulting max machine load increase.

    # Since we don't have current machine loads directly, we approximate:
    # For each op i and machine m, the "load" added is duration[i][m].
    # A good heuristic: pick the (i, m) pair that minimizes duration[i][m] + remaining_work[i] / n_machines
    # But the strategy says: minimize the increase in maximum machine load.
    # Since we don't know current loads, we use a proxy: for each op, pick the machine
    # that gives the smallest duration, and then among ops, pick the one with the
    # smallest "cost" = min_duration + small * remaining_work.

    # Actually, let's think more carefully. The strategy is load-balancing:
    # For each ready operation i, for each machine m in its options, the cost of
    # assigning i to m is the resulting load on m. Since we don't track loads,
    # we use: cost(i, m) = duration(i, m). We want to minimize the max load increase.
    # A simple greedy: for each op, find the best (min duration) machine.
    # Then among ops, pick the one with the smallest best-duration, breaking ties
    # by largest remaining_work (to keep critical jobs moving).

    # But we can do better: consider all (i, m) pairs and pick the one that
    # minimizes duration, with a tie-breaker that favors ops with more remaining work.

    best_idx = 0
    best_score = float('inf')

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]

        # Find the minimum duration across machine options for this op
        min_dur = float('inf')
        for m, dur in machine_options.items():
            if dur < min_dur:
                min_dur = dur

        # Score: primarily minimize the duration (load increase),
        # secondarily maximize remaining_work (to prioritize critical jobs),
        # tertiarily minimize remaining_ops (to finish jobs soon)
        # Use a weighted combination:
        score = min_dur - 0.01 * remaining_work + 0.001 * remaining_ops

        if score < best_score:
            best_score = score
            best_idx = i

    return _safe_return(best_idx)
# EVOLVE-BLOCK-END
