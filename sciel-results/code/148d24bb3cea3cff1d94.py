import numpy as np
import time

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
                
    # Calculate Total Downstream Work (Sum of times of task and all successors)
    tds = [0.0] * n
    state = [0] * n # 0: unvisited, 1: in stack, 2: done
    
    def get_tds(i):
        if state[i] == 2:
            return tds[i]
        if state[i] == 1:
            return 0.0 # Cycle protection
            
        state[i] = 1
        sum_succ_tds = 0.0
        for j in successors[i]:
            if time.time() > deadline:
                return times[i]
            s_tds = get_tds(j)
            sum_succ_tds += s_tds
        tds[i] = times[i] + sum_succ_tds
        state[i] = 2
        return tds[i]
        
    for i in range(n):
        get_tds(i)
        
    # Prioritize by Total Downstream Work
    # Add small tie-breaker: prefer longer tasks if TDS is equal
    priorities = np.asarray(tds, dtype=float) + times * 1e-9
    return priorities
