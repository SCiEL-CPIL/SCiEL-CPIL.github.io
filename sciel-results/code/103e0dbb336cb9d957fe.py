"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict

def priority(instance):
    """
    Computes priority scores based on Critical Path Method (CPM) slack.
    Activities with lower slack (more critical) receive higher priority scores.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    preds_raw = instance["predecessors"]
    n = len(durations)
    
    # Normalize predecessors to a list of lists of integers
    predecessors = []
    for i in range(n):
        p = preds_raw[i]
        # Convert to list of ints to ensure compatibility
        if isinstance(p, np.ndarray):
            predecessors.append([int(x) for x in p])
        else:
            # Handle case where it might be a scalar or other iterable
            try:
                predecessors.append([int(x) for x in p])
            except TypeError:
                if p is None or p == [] or p == ():
                    predecessors.append([])
                else:
                    predecessors.append([int(p)])

    # Build successor list
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)

    # 1. Forward Pass: Calculate Earliest Start (ES) and Earliest Finish (EF)
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    
    # We need to process nodes in topological order.
    # Since it's a DAG, we can use a simple topological sort or iterative relaxation.
    # Topological sort using Kahn's algorithm for safety.
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in predecessors[j]:
            in_degree[j] += 1
            
    queue = [i for i in range(n) if in_degree[i] == 0]
    topological_order = []
    while queue:
        # Pop a node (order doesn't strictly matter for correctness if we just append)
        u = queue.pop(0)
        topological_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If topological_order length < n, there's a cycle, but RCPSP instances are DAGs.
    # Just in case, add any remaining nodes (though this shouldn't happen).
    if len(topological_order) < n:
        for i in range(n):
            if i not in topological_order:
                topological_order.append(i)

    for u in topological_order:
        es[u] = 0
        for i in predecessors[u]:
            if ef[i] > es[u]:
                es[u] = ef[i]
        ef[u] = es[u] + durations[u]
        
    # Project end time
    project_end = np.max(ef) if n > 0 else 0.0

    # 2. Backward Pass: Calculate Latest Start (LS) and Latest Finish (LF)
    ls = np.zeros(n, dtype=float)
    lf = np.zeros(n, dtype=float)
    
    # Initialize LF for nodes with no successors to project_end
    for u in topological_order:
        if not successors[u]:
            lf[u] = project_end
        else:
            lf[u] = np.inf

    # Process in reverse topological order
    for u in reversed(topological_order):
        if successors[u]:
            min_ls = np.inf
            for v in successors[u]:
                if ls[v] < min_ls:
                    min_ls = ls[v]
            lf[u] = min_ls
        ls[u] = lf[u] - durations[u]

    # 3. Calculate Slack
    slack = ls - es
    
    # 4. Assign Priority
    # Higher priority for lower slack.
    # Using 1 / (slack + 1) ensures positive values and high priority for critical path (slack=0).
    # Adding 1 avoids division by zero and scales the priority reasonably.
    priority_scores = 1.0 / (slack + 1.0)
    
    return priority_scores
# EVOLVE-BLOCK-END
