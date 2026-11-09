"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: evaluator kills after 150s wall-clock. We track elapsed since module load.
_T0 = _time.time()
_DEADLINE = 135.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _T0) < _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Model as a bipartite assignment where we want to minimize the
    maximum completion time (makespan proxy). For each ready operation, we
    consider assigning it to its best (shortest) machine, and we want to pick
    the operation+machine pair that minimizes the resulting peak machine load,
    while also considering remaining work balance.
    
    We use a cost matrix approach: for each ready op i and each machine m in its
    machine_options, the cost is (current_load[m] + duration[i][m]) which is the
    completion time on that machine. We want to pick the op that, when assigned
    to its best machine, leads to the best overall balance.
    
    Since we can only pick ONE operation to start, we score each ready operation
    by:
      - The minimum completion time it would achieve (min over its machines of 
        load[m] + dur)
      - Weighted by remaining work (to prioritize jobs with more work)
      - And a penalty for creating an imbalanced load
    
    We use a smooth surrogate: soft-min over machines for the completion time,
    combined with a load-balance penalty.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Gather data
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # For each op, get machine options and durations
    # We need to know current machine loads. The 'time' parameter is current time.
    # But we don't have direct access to machine loads. We need to infer from context.
    # In an open-shop simulation, the dispatch function is called when a machine becomes
    # free. The ready_operations are those that can be started. We don't have explicit
    # machine load info, so we use the durations and remaining work as proxies.
    
    # Since we don't have machine loads directly, let's use a different approach:
    # Score each operation based on:
    # 1. Its minimum processing time across available machines (shorter = better to start)
    # 2. Its remaining work (more = more urgent)
    # 3. Its remaining ops (more = more complex, prioritize)
    
    # Compute for each op: min duration, mean duration, max duration
    min_durs = np.zeros(n)
    mean_durs = np.zeros(n)
    max_durs = np.zeros(n)
    num_machines = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        durs = list(op["machine_options"].values())
        if durs:
            min_durs[i] = min(durs)
            mean_durs[i] = np.mean(durs)
            max_durs[i] = max(durs)
            num_machines[i] = len(durs)
    
    # Normalize
    max_rw = np.max(remaining_works) if np.max(remaining_works) > 0 else 1
    max_ro = np.max(remaining_ops) if np.max(remaining_ops) > 0 else 1
    max_md = np.max(min_durs) if np.max(min_durs) > 0 else 1
    max_mmd = np.max(mean_durs) if np.max(mean_durs) > 0 else 1
    
    # Score: we want to prioritize ops with:
    # - High remaining work (urgent)
    # - High remaining ops (complex)
    # - Low min duration (can finish quickly, freeing machine)
    # - Many machine options (flexible, less critical to sequence)
    
    # Weighted combination
    w_rw = 0.35   # weight for remaining work
    w_ro = 0.25   # weight for remaining ops
    w_md = 0.25   # weight for min duration (inverted: lower is better)
    w_flex = 0.15 # weight for flexibility (more machines = less urgent)
    
    score = (
        w_rw * (remaining_works / max_rw) +
        w_ro * (remaining_ops / max_ro) +
        w_md * (1.0 - min_durs / max_md) +
        w_flex * (1.0 - num_machines / (np.max(num_machines) if np.max(num_machines) > 0 else 1))
    )
    
    # Use soft-argmax style: pick the op with highest score
    # But add a small random tie-breaker to avoid pathological cycles
    idx = int(np.argmax(score))
    
    return idx
# EVOLVE-BLOCK-END
