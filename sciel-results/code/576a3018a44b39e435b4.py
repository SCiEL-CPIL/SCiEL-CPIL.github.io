"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_SECONDS = 145.0  # safety margin before 150s kill


def _compute_priority(instance):
    """
    Compute a priority score for each activity to minimize makespan.
    
    Strategy: Lexicographic priority using:
    1. Minimum Total Slack (Urgency)
    2. Maximum Tail (Critical Path Length)
    3. Maximum Resource Demand Intensity (Resource Pressure)
    4. Maximum Duration (LPT)
    
    This approach avoids the dilution effect of weighted sums by creating
    a hierarchical scoring system where primary criteria dominate.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance["demands"]  # shape: (n_activities, n_resources)
    capacities = instance["capacities"]  # shape: (n_resources,)
    predecessors = instance["predecessors"]  # list of lists or similar
    
    n = len(durations)
    
    # Convert predecessors to a consistent format
    if isinstance(predecessors, np.ndarray):
        pred_lists = []
        for i in range(n):
            pred_lists.append(list(predecessors[i]))
    else:
        pred_lists = [list(p) for p in predecessors]
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            if i not in successors[p]:
                successors[p].append(i)
    
    # Topological order
    in_degree = [len(pred_lists[i]) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # tail[i] = longest duration from i to any sink (including i's own duration)
    tail = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        if not successors[i]:
            tail[i] = durations[i]
        else:
            tail[i] = durations[i] + max(tail[j] for j in successors[i])
    
    # head[i] = longest duration from any source to i (including i's own duration)
    head = np.zeros(n, dtype=float)
    for i in topo_order:
        if pred_lists[i]:
            head[i] = durations[i] + max(head[p] for p in pred_lists[i])
        else:
            head[i] = durations[i]
    
    # Project makespan without resource constraints (longest path)
    makespan_ignoring_resources = max(tail[i] for i in range(n)) if n > 0 else 0.0
    
    # Total float / slack
    slack = np.zeros(n, dtype=float)
    for i in range(n):
        slack[i] = makespan_ignoring_resources - head[i] - tail[i] + durations[i]
    
    # Resource demand intensity
    demand_intensity = np.zeros(n, dtype=float)
    n_resources = len(capacities)
    for i in range(n):
        for r in range(n_resources):
            cap = capacities[r]
            if cap > 0:
                demand_intensity[i] += demands[i][r] / cap
            else:
                demand_intensity[i] += demands[i][r] * 10.0
    
    # Lexicographic scoring
    # We want: Min Slack, Max Tail, Max Demand, Max Duration
    # To make this a single score where higher is better:
    # We can rank each metric and combine with large powers of 10.
    # Rank 1 is best.
    
    def rank_array(arr, descending=False):
        """
        Assign ranks 1..n. 1 is best.
        If descending=True, larger values get rank 1.
        If descending=False, smaller values get rank 1.
        """
        arr = np.asarray(arr, dtype=float)
        if descending:
            # Sort indices by value descending
            indices = np.argsort(-arr)
        else:
            # Sort indices by value ascending
            indices = np.argsort(arr)
        
        ranks = np.zeros(len(arr), dtype=float)
        # Handle ties: assign average rank or just the position? 
        # For strict lexicographic, we need to distinguish ties or handle them consistently.
        # Using np.argsort with kind='stable' preserves order for ties, which is arbitrary.
        # Better to use a dense rank or handle ties by secondary keys.
        # However, since we are combining multiple keys, we can just use the sorted position.
        # To ensure determinism, we break ties by index.
        
        # Create a stable sort by value, then by index
        if descending:
            order = np.lexsort((np.arange(len(arr)), -arr))
        else:
            order = np.lexsort((np.arange(len(arr)), arr))
        
        for rank, idx in enumerate(order):
            ranks[idx] = rank + 1.0
            
        return ranks

    # Rank 1 is best (lowest slack, highest tail, highest demand, highest duration)
    rank_slack = rank_array(slack, descending=False)       # Min slack is best
    rank_tail = rank_array(tail, descending=True)           # Max tail is best
    rank_demand = rank_array(demand_intensity, descending=True) # Max demand is best
    rank_dur = rank_array(durations, descending=True)       # Max duration is best
    
    # Combine with large weights to ensure lexicographic order
    # Weight for Slack is highest, then Tail, then Demand, then Duration
    # n=30, so max rank is 30. 
    # To be safe, use base 100.
    score = (
        (100.0 ** 3) * (n + 1 - rank_slack) +
        (100.0 ** 2) * (n + 1 - rank_tail) +
        (100.0 ** 1) * (n + 1 - rank_demand) +
        (100.0 ** 0) * (n + 1 - rank_dur)
    )
    
    return score


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (scheduled first when eligible).
    """
    start_time = time.time()
    
    try:
        scores = _compute_priority(instance)
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE_SECONDS:
            # Fallback
            return np.asarray(instance["durations"], dtype=float)
        return scores
    except Exception:
        # Fallback: longest processing time
        try:
            return np.asarray(instance["durations"], dtype=float)
        except Exception:
            n = len(instance.get("durations", []))
            return np.ones(n, dtype=float)
# EVOLVE-BLOCK-END
