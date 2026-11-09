"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# --- deadline guard ---
_START = None
_DEADLINE = 150.0
_MARGIN = 5.0


def _elapsed():
    global _START
    if _START is None:
        _START = _time.time()
    return _time.time() - _START


def _fast_fallback(ready_operations):
    # O(n) greedy: most remaining work, then shortest duration.
    best_i = 0
    best_key = None
    for i, op in enumerate(ready_operations):
        mw = op["remaining_work"]
        opts = op["machine_options"]
        min_d = min(opts.values())
        key = (mw, -min_d)
        if best_key is None or key > best_key:
            best_key = key
            best_i = i
    return best_i


def _pick_machine(op):
    # Choose the eligible machine with the shortest duration for this op.
    # Tie-break: prefer machines that appear in fewer other ops' options
    # (less contended) -> lower expected contention.
    opts = op["machine_options"]
    best_m = None
    best_key = None
    for m, d in opts.items():
        # primary: shortest duration; secondary: least contention
        key = (-d,)
        if best_key is None or key > best_key:
            best_key = key
            best_m = m
    return best_m


def _full_heuristic(ready_operations, time):
    """
    Multi-criteria dispatch:
      - Primary: maximize remaining_work (long jobs first, FWS-like).
      - Secondary: minimize the best (shortest) duration across machines
        (so we don't idle a machine on a long op when a short one is ready).
      - Tertiary: minimize remaining_ops (finish jobs to free their successors).
    Returns (index, machine).
    """
    n = len(ready_operations)
    # Precompute a contention proxy: for each machine, how many ready ops can use it.
    machine_load = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_load[m] = machine_load.get(m, 0) + 1

    best_i = 0
    best_score = None
    for i, op in enumerate(ready_operations):
        mw = op["remaining_work"]
        ro = op["remaining_ops"]
        opts = op["machine_options"]
        # best (shortest) duration and its machine
        min_d = None
        min_m = None
        for m, d in opts.items():
            if min_d is None or d < min_d:
                min_d = d
                min_m = m
        # contention of the chosen machine
        cont = machine_load.get(min_m, 0)
        # score: higher is better
        # weight remaining_work heavily, then prefer shorter duration,
        # then fewer remaining ops, then lower contention.
        score = (mw, -min_d, -ro, -cont)
        if best_score is None or score > best_score:
            best_score = score
            best_i = i
    best_op = ready_operations[best_i]
    # choose machine: shortest duration; tie-break by lower contention
    opts = best_op["machine_options"]
    best_m = None
    best_key = None
    for m, d in opts.items():
        cont = machine_load.get(m, 0)
        key = (-d, -cont)
        if best_key is None or key > best_key:
            best_key = key
            best_m = m
    return (best_i, best_m)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    # deadline guard: if we're close to the limit, use the fast path
    if _elapsed() > (_DEADLINE - _MARGIN):
        return _fast_fallback(ready_operations)
    # full heuristic (still O(n * opts), very fast for these sizes)
    return _full_heuristic(ready_operations, time)
# EVOLVE-BLOCK-END
