"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    
    Uses Minimum Total Slack heuristic:
    - Activities with the least "free time" (slack) in the critical path 
      (ignoring resources) are prioritized.
    - This forces the scheduler to commit to tight sequences early.
    - Tie-breaking: Higher resource demand to resolve contention for heavy tasks.
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            if i is not None:
                successors[i].append(j)
    
    # 1. Forward Pass: Compute Earliest Start (ES)
    ES = np.zeros(n, dtype=float)
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in predecessors[j]:
            if i is not None:
                in_degree[j] += 1
    
    # Topological order
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            ES[succ] = max(ES[succ], ES[node] + durations[node])
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # Handle cycles gracefully (should not happen in valid RCPSP, but safe)
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)
        # Re-calculate ES for remaining? Assume 0 or propagate if possible. 
        # For simplicity, assume valid DAG.

    # Project duration without resource constraints (Critical Path Length)
    makespan_no_res = np.max(ES + durations) if n > 0 else 0.0
    
    # 2. Backward Pass: Compute Latest Start (LS)
    # LS[i] is the latest time activity i can start without delaying the project.
    # We work backwards from the end.
    LS = np.full(n, makespan_no_res, dtype=float) # Initialize with max possible LS
    
    # Process in reverse topological order
    for node in reversed(topo_order):
        if successors[node]:
            # Latest Start is min of (Latest Start of successors - current duration)
            # Wait, standard formula:
            # LF(i) = min(LS(j)) for j in Succ(i)
            # LS(i) = LF(i) - d(i)
            min_succ_LS = min(LS[s] for s in successors[node])
            LS[node] = min_succ_LS - durations[node]
        else:
            # Sink node: must finish by makespan_no_res
            LS[node] = makespan_no_res - durations[node]
            
    # Ensure LS is not negative (clamp to 0 if needed, though valid DAGs shouldn't go negative if makespan is correct)
    LS = np.maximum(LS, 0.0)
    
    # 3. Calculate Total Slack
    # Slack = LS - ES
    slack = LS - ES
    # Clamp small negatives due to floating point
    slack = np.maximum(slack, 0.0)
    
    # 4. Calculate Resource Demand
    demands = instance["demands"]
    total_demand = np.sum(np.asarray(demands, dtype=float), axis=1)
    
    # 5. Compute Priority Score
    # Primary: Inverse Slack (Low Slack = High Priority)
    # We want a score where lower slack gives higher score.
    # score_slack = max_slack - slack
    max_slack = np.max(slack) if np.max(slack) > 0 else 1.0
    norm_slack_score = (max_slack - slack) / max_slack if max_slack > 0 else 0.0
    
    # Secondary: Resource Demand (High Demand = High Priority)
    max_demand = np.max(total_demand) if np.max(total_demand) > 0 else 1.0
    norm_demand = total_demand / max_demand
    
    # Combine
    # Weight 0.8 for Slack, 0.2 for Demand
    score = 0.8 * norm_slack_score + 0.2 * norm_demand
    
    return score
# EVOLVE-BLOCK-END
