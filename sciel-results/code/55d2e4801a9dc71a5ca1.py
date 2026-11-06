"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache
import time


def priority(instance):
    t_start = time.time()
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    preds = instance.get("predecessors", [set() for _ in range(n)])
    cycle_time = instance.get("cycle_time", 40)
    
    # Compute successors
    succs = [set() for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].add(i)
    
    # Count transitive successors
    @lru_cache(maxsize=None)
    def count_succs(i):
        if not succs[i]:
            return 0
        return 1 + sum(count_succs(j) for j in succs[i])
    
    # Total downstream time (including self)
    @lru_cache(maxsize=None)
    def total_time_succ(i):
        return times[i] + sum(total_time_succ(j) for j in succs[i])
    
    # Longest path from any source to task i (earliest start time in a single-machine timeline)
    # This measures "positional criticality": how deep into the dependency chain the task sits
    @lru_cache(maxsize=None)
    def longest_path_from_source(i):
        if not preds[i]:
            return 0.0
        return max(longest_path_from_source(p) + times[p] for p in preds[i])
    
    # Safety check: if computation is taking too long, fall back to simpler metrics
    elapsed = time.time() - t_start
    if elapsed > 100:
        # Fallback: just use processing time
        return times
    
    succ_counts = np.array([count_succs(i) for i in range(n)])
    total_times = np.array([total_time_succ(i) for i in range(n)])
    longest_paths = np.array([longest_path_from_source(i) for i in range(n)])
    
    # Normalize components
    max_time = times.max() if times.max() > 0 else 1
    max_succ = succ_counts.max() if succ_counts.max() > 0 else 1
    max_total = total_times.max() if total_times.max() > 0 else 1
    max_lp = longest_paths.max() if longest_paths.max() > 0 else 1
    
    # Positional criticality: ratio of earliest start to cycle time
    # Tasks with high positional criticality are late in the dependency chain
    # and constrain the timeline's end phase; placing them early in station filling
    # helps the greedy packer respect their tight scheduling windows
    positional_crit = longest_paths / max(max_lp, 1e-9)
    
    # Weighted combination:
    # - Processing time: long tasks should be placed first to avoid fragmentation
    # - Successor count: tasks that unlock many downstream tasks should go early
    # - Total downstream time: tasks on critical paths should go early
    # - Positional criticality: tasks deep in the dependency chain constrain the timeline
    priority_scores = (
        0.35 * (times / max_time) +
        0.25 * (succ_counts / max_succ) +
        0.25 * (total_times / max_total) +
        0.15 * positional_crit
    )
    
    return priority_scores
# EVOLVE-BLOCK-END
