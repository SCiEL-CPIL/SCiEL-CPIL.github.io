"""
Strategy: Dynamic Priority Heuristic based on Critical Path Density and Fan-out.

Instead of searching for an optimal assignment (which is NP-hard and slow) or
reverse-engineering static scores from a found solution, we compute a robust
heuristic score for each task that reflects its structural importance in the
assembly line.

The score for a task t is a combination of:
1. Critical Path Density: times[t] / (times[t] + max_downstream_height[t])
   This prioritizes tasks that are significant portions of the critical path
   leading through them. It helps identify "bottlenecks" that constrain the
   cycle time more than others.
2. Fan-out Pressure: len(successors[t])
   Tasks with many immediate successors are branching points. Placing them
   early (high priority) ensures their successors are eligible sooner,
   reducing the "waiting time" for downstream tasks and potentially allowing
   better packing of subsequent stations.

We combine these into a single score. Since the evaluator uses a greedy
"highest priority eligible task" rule, higher scores mean the task is picked
earlier. We want to pick critical, high-fanout tasks early to establish the
backbone of the line.

This approach is O(N^2) or O(N log N) depending on implementation details,
which is extremely fast (microseconds), allowing us to run it safely within
the time limit without risk of timeout. It avoids the local minima traps of
static simple heuristics (like just "longest first") by considering the
global structure (downstream paths).
"""

# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

def priority(instance):
    """
    Computes priority scores for each task in the assembly line.
    
    Args:
        instance: dict with keys 'times', 'predecessors', 'cycle_time'
        
    Returns:
        numpy array of scores, one per task.
    """
    times = np.array(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    # 1. Build successor lists and in-degrees for topological sort
    succs = [[] for _ in range(n_tasks)]
    in_deg = [0] * n_tasks
    
    for i in range(n_tasks):
        for p in preds[i]:
            if p >= 0 and p < n_tasks:
                succs[p].append(i)
                in_deg[i] += 1
                
    # 2. Topological Sort (Kahn's algorithm)
    q = deque([i for i in range(n_tasks) if in_deg[i] == 0])
    topo_order = []
    indeg_copy = in_deg[:]
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succs[u]:
            indeg_copy[v] -= 1
            if indeg_copy[v] == 0:
                q.append(v)
                
    # If not all tasks are in topo_order, there's a cycle or invalid input.
    # Fallback to simple index order if that happens (shouldn't happen in valid ALB).
    if len(topo_order) < n_tasks:
        topo_order = list(range(n_tasks))
        # Rebuild succs/in_deg might be needed if we want to be robust, 
        # but for scoring, we just need the structure. 
        # If there's a cycle, the problem is ill-posed. We'll proceed with what we have.

    # 3. Compute "Height" (Longest Path to End) for each task
    # height[t] = times[t] + max(height[s] for s in successors[t])
    # This represents the remaining time on the critical path starting from t.
    height = [0.0] * n_tasks
    for u in reversed(topo_order):
        if not succs[u]:
            height[u] = times[u]
        else:
            max_h = 0.0
            for v in succs[u]:
                if height[v] > max_h:
                    max_h = height[v]
            height[u] = times[u] + max_h
            
    # 4. Compute Scores
    # We want to prioritize tasks that are "critical" and "high fanout".
    
    # Component 1: Critical Path Density
    # Ratio of task time to total downstream path length (including self).
    # If height[u] is large, and times[u] is a significant chunk, it's a bottleneck.
    # density[u] = times[u] / height[u] if height[u] > 0 else 0
    # Note: height[u] is always >= times[u] > 0 (assuming times > 0).
    
    # Component 2: Fan-out (Number of immediate successors)
    # fanout[u] = len(succs[u])
    
    # Combination:
    # We use a weighted sum. 
    # Normalization is helpful to balance the scales.
    # Max height is the critical path length.
    # Max fanout is the max out-degree.
    
    max_height = max(height) if height else 1.0
    if max_height == 0:
        max_height = 1.0
        
    max_fanout = max([len(s) for s in succs]) if succs else 1
    if max_fanout == 0:
        max_fanout = 1
        
    scores = np.zeros(n_tasks, dtype=float)
    
    for u in range(n_tasks):
        # Density: how much of the critical path through u is consumed by u itself?
        # High density means u is a "heavy" part of the critical path.
        density = times[u] / height[u]
        
        # Fanout: how many tasks depend directly on u?
        fanout = len(succs[u])
        
        # Normalize fanout to [0, 1] roughly
        norm_fanout = fanout / max_fanout
        
        # Combine. 
        # We want to emphasize density slightly more, as it directly relates to cycle time constraints.
        # But fanout helps with eligibility flow.
        # Let's try: score = density + 0.5 * norm_fanout
        # Or perhaps a multiplicative interaction?
        # Let's stick to additive for stability.
        
        # Another consideration: "Successor Weighted Height"?
        # If a task has many successors, its "influence" is higher.
        
        # Let's refine the "Critical Path Density" metric as per the directive:
        # "ratio of the task's duration to the total length of its longest downstream path"
        # This is exactly density = times[u] / height[u].
        
        # "combine this with the count of immediate successors"
        # score = density + alpha * fanout
        
        # To make the scores comparable and robust, let's normalize density to [0,1] as well?
        # Density is already in (0, 1] because times[u] <= height[u].
        
        # We can also consider the "position" in the topological order?
        # No, the greedy algorithm handles the order based on eligibility.
        
        # Let's use a slight boost for tasks with high fanout to ensure they are placed
        # early enough to unlock their successors.
        
        scores[u] = density + 0.3 * norm_fanout
        
    # Optional: Add a tiny bit of random noise to break ties in a consistent way?
    # No, deterministic is better for reproducibility and stability.
    
    return scores

# EVOLVE-BLOCK-END
