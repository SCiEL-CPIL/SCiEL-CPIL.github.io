import numpy as np
import time
from collections import defaultdict

def priority(instance):
    start_time = time.time()
    # Generous deadline, computation is tiny
    deadline = start_time + 5.0 
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)

    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        if predecessors[i]:
            for pred in predecessors[i]:
                successors[pred].append(i)
                
    # Calculate Critical Path (Longest path from node to sink)
    cp = [0.0] * n
    state = [0] * n # 0: unvisited, 1: in stack, 2: done
    
    def get_cp(i):
        if state[i] == 2:
            return cp[i]
        if state[i] == 1:
            return 0.0 # Cycle protection
            
        state[i] = 1
        max_succ_cp = 0.0
        for j in successors[i]:
            if time.time() > deadline:
                return times[i]
            s_cp = get_cp(j)
            if s_cp > max_succ_cp:
                max_succ_cp = s_cp
        cp[i] = times[i] + max_succ_cp
        state[i] = 2
        return cp[i]
        
    for i in range(n):
        get_cp(i)
        
    # Calculate upstream path (longest path from source to node)
    up = [0.0] * n
    state_up = [0] * n
    
    def get_up(i):
        if state_up[i] == 2:
            return up[i]
        if state_up[i] == 1:
            return 0.0
            
        state_up[i] = 1
        max_pred_up = 0.0
        for pred in predecessors[i]:
            if time.time() > deadline:
                return times[i]
            p_up = get_up(pred)
            if p_up > max_pred_up:
                max_pred_up = p_up
        up[i] = times[i] + max_pred_up
        state_up[i] = 2
        return up[i]
        
    for i in range(n):
        get_up(i)
        
    # Combine heuristics:
    # 1. Critical Path (downstream) - most important
    # 2. Upstream path - helps identify tasks that are bottlenecks from start
    # 3. Normalized slack - favor tasks that consume most station capacity
    #    relative to branching complexity they unlock
    # 4. Successor count - prefer tasks that unlock more work (reduces idle time)
    # 5. Task time - tie-breaker
    
    cycle_time = float(instance.get("cycle_time", 40.0))
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Avoid division by zero: tasks with no successors get a high slack (low priority)
    # Use max(1, num_successors) to handle terminal tasks
    denom = np.maximum(num_successors, 1.0)
    slack = (cycle_time - times) / denom
    
    # Normalize slack to [0, 1] range for stable weighting
    slack_min = np.min(slack)
    slack_max = np.max(slack)
    if slack_max > slack_min:
        slack_norm = (slack - slack_min) / (slack_max - slack_min)
    else:
        slack_norm = np.zeros_like(slack)
    
    # Invert: smaller slack (tighter capacity) -> higher priority
    bottleneck_score = 1.0 - slack_norm
    
    # Successor unlock heuristic: tasks with more successors unlock more work
    # Normalize to [0, 1]
    succ_max = np.max(num_successors) if n > 0 else 1.0
    succ_norm = num_successors / succ_max if succ_max > 0 else np.zeros_like(num_successors)
    
    # Normalize cp and up for consistent weighting
    cp_arr = np.asarray(cp, dtype=float)
    up_arr = np.asarray(up, dtype=float)
    
    cp_max = np.max(cp_arr) if n > 0 else 1.0
    up_max = np.max(up_arr) if n > 0 else 1.0
    cp_norm = cp_arr / cp_max if cp_max > 0 else cp_arr
    up_norm = up_arr / up_max if up_max > 0 else up_arr
    
    # Weighted combination with normalized components:
    # cp has highest weight (longest remaining path downstream)
    # up has moderate weight (longest path from source)
    # bottleneck_score favors tasks that are capacity bottlenecks
    # succ_norm favors tasks that unlock more successor work
    # times is a tiny tie-breaker
    priorities = (1.0 * cp_norm 
                  + 0.4 * up_norm 
                  + 0.25 * bottleneck_score 
                  + 0.15 * succ_norm 
                  + times * 1e-6)
    
    return priorities
