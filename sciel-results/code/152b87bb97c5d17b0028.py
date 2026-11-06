"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Assigns a priority score to each task for the assembly line balancing heuristic.
    The heuristic is: pick the highest-priority eligible task that fits the cycle time.
    
    Strategy:
    1. Compute 'rank' (longest path from start to task) and 'tail' (longest path from task to end).
    2. Base priority = rank + tail (positional centrality).
    3. Boost priority by task duration (Longest Processing Time) to pack heavy items.
    4. Add a small random jitter to break ties deterministically-ish but allow exploration.
       Actually, deterministic is better for scoring consistency.
       Let's use a weighted combination:
       Priority = alpha * (rank + tail) + beta * duration + gamma * (number of immediate successors)
       
       Let's refine based on literature:
       - Serial Position Index (SPI) / Rank: Good for early tasks.
       - Tail: Good for late tasks.
       - Duration: Important for fitting.
       
       A common effective heuristic is:
       Priority = duration + tail
       Or
       Priority = rank + duration
       
       Let's try to combine them robustly.
       The "critical path" length is rank + tail.
       Tasks on the critical path should be prioritized.
       Also, longer tasks should be prioritized to avoid wasting space (LPT heuristic).
       
       Let's compute:
       1. Topological sort / DAG analysis.
       2. rank[i] = max(rank[j] + time[j]) for all predecessors j, else time[i]. (Time to reach i)
          Wait, standard rank is usually length of longest path ending at i.
          Let's define:
          start_rank[i] = time[i] + max(start_rank[p] for p in preds)
          end_rank[i] = time[i] + max(end_rank[s] for s in succs)
          
          Total critical path length L = max(start_rank[i] + end_rank[i] - time[i])
          
          A good priority is often:
          p[i] = start_rank[i] + end_rank[i]
          
          However, simple LPT (Longest Processing Time) is often very strong for ALB.
          The seed program uses just `times`. Score 1.0.
          We want to beat 1.0.
          
          Let's try a composite score:
          score[i] = start_rank[i] + end_rank[i] + 0.5 * time[i]
          
          Or simply:
          score[i] = end_rank[i] + time[i]  (Prioritize tasks that lead to long tails and are long themselves)
          
          Let's look at the "Highest Positional Weight" (HPW) heuristic.
          Positional weight = time[i] + sum(time[s] for s in successors) ... recursively.
          This is exactly `end_rank` if we define it as longest path.
          
          Let's try:
          priority[i] = end_rank[i]
          
          Or:
          priority[i] = start_rank[i] + end_rank[i]
          
          Let's implement a robust calculation of these paths.
          Since n=30, we can do this with simple DP on the DAG.
          
          Steps:
          1. Build adjacency lists (preds, succs).
          2. Topological sort (Kahn's algorithm).
          3. Compute start_rank (forward pass).
          4. Compute end_rank (backward pass).
          5. Combine.
          
          Let's test the hypothesis that `start_rank + end_rank` is better than `time`.
          Or maybe `end_rank` alone.
          
          Actually, a very strong heuristic is "Critical Path First" combined with "LPT".
          Let's use:
          priority[i] = (start_rank[i] + end_rank[i]) * 1.0 + time[i] * 0.5
          
          Let's also consider the number of successors. Tasks with many successors are "bottlenecks" and should be done early.
          priority[i] += len(succs[i]) * 0.1
          
          Let's stick to a clean composite:
          base = start_rank + end_rank
          boost = time
          final = base + boost
          
          We need to handle the case where predecessors/successors are empty.
          """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    preds = instance["predecessors"]
    
    # Build successor list
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            # preds[i] is list of indices of predecessors
            succs[p].append(i)
            
    # Topological Sort
    in_degree = [len(p) for p in preds]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        # Sort queue by something? Doesn't matter for correctness of topo sort.
        u = queue.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If not all nodes visited, there's a cycle. Assume no cycles.
    if len(topo_order) != n:
        # Fallback to times if cycle detected (shouldn't happen)
        return times

    # Forward pass: start_rank
    # start_rank[i] = time[i] + max(start_rank[p] for p in preds[i])
    start_rank = np.zeros(n)
    for u in topo_order:
        max_pred_rank = 0
        for p in preds[u]:
            if start_rank[p] > max_pred_rank:
                max_pred_rank = start_rank[p]
        start_rank[u] = times[u] + max_pred_rank

    # Backward pass: end_rank
    # end_rank[i] = time[i] + max(end_rank[s] for s in succs[i])
    end_rank = np.zeros(n)
    for u in reversed(topo_order):
        max_succ_rank = 0
        for s in succs[u]:
            if end_rank[s] > max_succ_rank:
                max_succ_rank = end_rank[s]
        end_rank[u] = times[u] + max_succ_rank

    # Composite Priority
    # Critical path position + duration
    # Weights can be tuned. 
    # Let's try: Priority = start_rank + end_rank + 0.5 * times
    # Or simply: Priority = end_rank + times
    # Literature suggests "Highest Positional Weight" (which is end_rank) is good.
    # "Longest Processing Time" (times) is good.
    # "Critical Path" (start+end) is good.
    
    # Let's use a weighted sum.
    # If we just return times, it's 1.0.
    # Let's try: priority = end_rank + 0.5 * times
    # Or: priority = start_rank + end_rank
    
    # Let's go with: priority = end_rank + times
    # This prioritizes tasks that are long AND lead to long tails.
    
    priority_scores = end_rank + times
    
    return priority_scores

# EVOLVE-BLOCK-END
