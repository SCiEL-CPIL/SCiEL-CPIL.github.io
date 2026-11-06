"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        # pick machine with smallest duration
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Pre-compute best machine and min duration for each operation
    best_machines = []
    min_durs = []
    rem_works = []
    rem_ops = []
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])

    # Deadline guard: fall back to simple deterministic choice
    if not _elapsed_ok():
        idx = int(np.argmax(rem_works))
        return (idx, best_machines[idx])

    # Load-Balancing Water-Filling heuristic:
    # For each candidate operation, assign it to the machine that would result
    # in the smallest increase in maximum machine load (L-infinity norm).
    # We approximate machine loads using the sum of durations assigned so far.
    
    # Since we don't have direct access to current machine loads, we use a proxy:
    # The "load" of a machine is estimated by the total remaining work that must
    # be processed on it. We approximate this by tracking which machines are
    # heavily loaded based on the operations' machine options.
    
    # For each operation i, compute the cost of assigning it to each possible machine.
    # Cost = resulting max load increase.
    
    # To estimate machine loads, we use the fact that in open-shop scheduling,
    # each job must be processed on each machine exactly once. We can estimate
    # the load on each machine by summing up the durations of all operations
    # that must use that machine.
    
    # However, since we only have ready operations, we'll use a simpler heuristic:
    # For each operation, compute the "balance score" based on how well it fits
    # into the current load distribution.
    
    # Simple approximation: For each operation, compute the minimum possible
    # maximum load increase if we assign it to the least-loaded compatible machine.
    
    # We'll use the remaining_work as a proxy for the "size" of the operation,
    # and the min_dur as the actual processing time.
    
    # For each operation, compute the cost of assigning it to its best machine:
    # cost = max(0, min_dur - average_machine_load)
    
    # Since we don't have direct load info, we use a heuristic based on
    # the distribution of remaining work across machines.
    
    # Compute the total remaining work
    total_rem_work = sum(rem_works)
    
    # For each operation, compute a load-balancing score:
    # We want to minimize the skew in machine loads.
    
    # For each operation, compute the "imbalance" that would be caused by
    # assigning it to its best machine.
    
    # Heuristic: Operations with smaller min_dur relative to their remaining_work
    # are more "balanced" and should be prioritized.
    
    # Score = min_dur / (remaining_work + epsilon)
    # Lower score means more balanced (smaller processing time relative to total work)
    
    epsilon = 1e-6
    scores = []
    for i in range(n):
        # Ratio of current processing time to total remaining work
        ratio = min_durs[i] / (rem_works[i] + epsilon)
        # We want to minimize this ratio (more balanced operations first)
        scores.append(ratio)
    
    # Sort by score (ascending - most balanced first)
    # Tie-break: smaller min_dur, then larger remaining_work
    order = sorted(
        range(n),
        key=lambda i: (
            scores[i],
            min_durs[i],
            -rem_works[i],
            rem_ops[i],
        ),
    )
    
    idx = order[0]
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
