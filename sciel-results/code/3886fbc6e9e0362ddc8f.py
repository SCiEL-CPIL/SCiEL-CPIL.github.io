"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Score each assembly task by its positional criticality.
    
    The idea: tasks that are on the longest path from the start (sources) 
    are the ones that constrain the timeline's start phase. By placing these
    early, we reduce the number of stations needed because we avoid idle time
    waiting for predecessors.
    
    We compute the earliest possible start time (ES) for each task, which is
    the longest path from any source to that task. We normalize this by the
    cycle time to get a ratio. Higher ES means the task is deeper in the
    precedence graph and thus more critical for the timeline.
    
    However, we also want to consider the task duration itself. A task with
    high ES but very short duration might be less critical than a task with
    slightly lower ES but much longer duration. So we combine ES and duration.
    
    Actually, let's think more carefully. The goal is to minimize the number
    of stations. A common heuristic is to prioritize tasks with high "rank"
    where rank is the longest path from the task to any sink (successor rank).
    But the directive says to use the ratio of earliest start time to cycle time.
    
    Let's follow the directive: prioritize by ES / cycle_time. But we should
    also consider that tasks with no predecessors (ES=0) should be placed early
    too, because they can start immediately. Wait, if ES=0, they are sources.
    Placing sources early makes sense because their successors can then be
    placed sooner.
    
    Actually, the directive says "identify and place tasks that strictly constrain
    the timeline's start phase first." Tasks with high ES constrain the middle/end
    of the timeline, not the start. Tasks with low ES (especially 0) constrain
    the start because they must be placed before their successors.
    
    Let me re-read: "the ratio of their earliest possible start time (longest path
    from sources) to the cycle time, to identify and place tasks that strictly
    constrain the timeline's start phase first."
    
    This is a bit confusing. If ES is the longest path from sources, then tasks
    with high ES are deep in the graph. Placing them first doesn't make sense
    because their predecessors haven't been placed yet.
    
    Wait, in assembly line balancing, the priority is used to select which eligible
    task to place next. Eligible tasks are those whose predecessors have all been
    placed. So among eligible tasks, we pick the one with highest priority.
    
    If we use ES as priority, then among eligible tasks, the one with the highest
    ES would be picked. But eligible tasks have all their predecessors placed, so
    their ES is determined by the longest path from sources to them.
    
    Actually, I think the intent is to use a rank that reflects how "critical"
    the task is. Let's use the longest path from the task to any sink (successor
    rank) combined with the task duration. This is a well-known heuristic.
    
    But the directive specifically says to use ES / cycle_time. Let's try that.
    
    Actually, let's compute both ES and the longest path to sinks (LS), and use
    a combination. But to follow the directive, let's primarily use ES.
    
    Let me just implement ES-based priority as directed, and see if it helps.
    """
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build adjacency list from predecessors
    # predecessors[i] is a list of task indices that must come before i
    predecessors = instance["predecessors"]
    
    # Compute in-degree and build successor list
    in_degree = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if isinstance(predecessors[i], (list, tuple, np.ndarray)) else [predecessors[i]]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Compute ES (earliest start) using topological order
    # ES[i] = max(ES[p] + times[p] for p in predecessors[i]), or 0 if no predecessors
    ES = np.zeros(n, dtype=float)
    # Use Kahn's algorithm for topological sort
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_deg_copy[succ] -= 1
            if in_deg_copy[succ] == 0:
                queue.append(succ)
    
    # Process in topological order
    for node in topo_order:
        preds = predecessors[node] if isinstance(predecessors[node], (list, tuple, np.ndarray)) else [predecessors[node]]
        if preds:
            ES[node] = max(ES[p] + times[p] for p in preds)
    
    # Normalize by cycle time
    priority_scores = ES / cycle_time if cycle_time > 0 else ES
    
    return priority_scores
# EVOLVE-BLOCK-END
