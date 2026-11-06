"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache
import time


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    preds = instance.get("predecessors", [set() for _ in range(n)])
    cycle_time = instance.get("cycle_time", 40)
    
    # Build successor lists
    succs = [set() for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].add(i)
    
    # Compute earliest start time (longest path from sources) for each task
    # This represents the positional criticality - how late a task can start
    # based on its predecessors' processing times
    @lru_cache(maxsize=None)
    def earliest_start(i):
        if not preds[i]:
            return 0.0
        return max(earliest_start(p) + times[p] for p in preds[i])
    
    # Compute latest finish time (longest path to sinks) for each task
    @lru_cache(maxsize=None)
    def latest_finish(i):
        if not succs[i]:
            return times[i]
        return times[i] + max(latest_finish(j) for j in succs[i])
    
    # Compute total downstream time (including self)
    @lru_cache(maxsize=None)
    def total_time_succ(i):
        return times[i] + sum(total_time_succ(j) for j in succs[i])
    
    # Count successors
    @lru_cache(maxsize=None)
    def count_succs(i):
        if not succs[i]:
            return 0
        return 1 + sum(count_succs(j) for j in succs[i])
    
    # Compute metrics
    earliest_starts = np.array([earliest_start(i) for i in range(n)])
    latest_finishes = np.array([latest_finish(i) for i in range(n)])
    total_times = np.array([total_time_succ(i) for i in range(n)])
    succ_counts = np.array([count_succs(i) for i in range(n)])
    
    # Normalize components
    max_time = times.max() if times.max() > 0 else 1
    max_earliest = earliest_starts.max() if earliest_starts.max() > 0 else 1
    max_latest = latest_finishes.max() if latest_finishes.max() > 0 else 1
    max_total = total_times.max() if total_times.max() > 0 else 1
    max_succ = succ_counts.max() if succ_counts.max() > 0 else 1
    
    # Positional criticality: ratio of earliest start to cycle time
    # Tasks with high earliest start are constrained by long predecessor chains
    positional_crit = earliest_starts / cycle_time if cycle_time > 0 else earliest_starts / max_earliest
    
    # Flexibility: how much slack a task has (lower is more critical)
    # A task with low flexibility should be placed early to avoid blocking others
    flexibility = np.zeros(n)
    for i in range(n):
        # Slack = cycle_time - (earliest_start + processing_time)
        # But we need to consider the full chain
        slack_i = cycle_time - (earliest_starts[i] + times[i])
        flexibility[i] = max(0, slack_i)
    
    max_flex = flexibility.max() if flexibility.max() > 0 else 1
    inv_flex = 1.0 - (flexibility / max_flex)
    
    # Weighted combination:
    # - Positional criticality: tasks with long predecessor chains (high earliest start)
    # - Processing time: long tasks
    # - Downstream impact: tasks that block many successors
    # - Inverse flexibility: tasks with little slack
    priority_scores = (
        0.3 * (positional_crit / max(positional_crit.max(), 1e-10)) +
        0.25 * (times / max_time) +
        0.25 * (total_times / max_total) +
        0.2 * inv_flex
    )
    
    return priority_scores
# EVOLVE-BLOCK-END
