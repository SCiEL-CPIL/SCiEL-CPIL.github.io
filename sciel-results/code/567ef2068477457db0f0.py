import numpy as np
import time
from collections import deque

def priority(instance):
    start_time = time.time()
    deadline = 120
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([])
        
    # Build graph
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
            
    # Topological sort
    in_degree = [0] * n
    for j in range(n):
        for i in predecessors[j]:
            in_degree[j] += 1
            
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # Calculate metrics
    # 1. Positional Weights (RPW)
    pos_weights = times.copy()
    for u in reversed(topo_order):
        for v in successors[u]:
            pos_weights[u] += pos_weights[v]
            
    # 2. Successor Counts
    succ_counts = np.array([len(s) for s in successors])
    
    # 3. Predecessor Counts
    pred_counts = np.array([len(p) for p in predecessors])
    
    # 4. Critical Path Length (Depth)
    # L[i] = 1 + max(L[succ])
    depth = np.ones(n)
    for u in reversed(topo_order):
        if successors[u]:
            depth[u] = 1 + max(depth[v] for v in successors[u])
        else:
            depth[u] = 1.0
            
    # Candidate Heuristics
    # Higher value = Higher Priority
    candidates = {
        "rpw": pos_weights,
        "lpt": times,
        "succ": succ_counts,
        "pred": -pred_counts, # Prioritize tasks with fewer preds? Or more? 
                             # Let's try both. Tasks with 0 preds are available. 
                             # Maybe prioritize tasks that are "critical" in terms of preds?
                             # Let's just use raw counts and negated counts as candidates.
        "pred_neg": pred_counts,
        "depth": depth,
        "rpw_succ": pos_weights + succ_counts,
        "rpw_01_succ": pos_weights + 0.1 * succ_counts,
        "rpw_10_succ": pos_weights + 1.0 * succ_counts,
        "depth_succ": depth + succ_counts
    }
    
    # Simulation Function
    def simulate(prio_vec):
        # prio_vec is numpy array
        unassigned_preds = [len(p) for p in predecessors]
        available = [i for i in range(n) if unassigned_preds[i] == 0]
        assigned = [False] * n
        stations = 0
        
        while any(not a for a in assigned):
            stations += 1
            current_load = 0
            
            while True:
                best_task = -1
                best_prio = -np.inf
                
                for i in available:
                    if assigned[i]:
                        continue
                    if times[i] <= (40.0 - current_load): # cycle_time is 40? 
                       # Wait, instance has cycle_time.
                       # instance["cycle_time"]
                       pass 
                # I need to pass cycle_time to simulate.
                pass
                
    # Corrected simulate
    cycle_time = instance.get("cycle_time", 40.0)
    
    def simulate(prio_vec):
        unassigned_preds = [len(p) for p in predecessors]
        available = [i for i in range(n) if unassigned_preds[i] == 0]
        assigned = [False] * n
        stations = 0
        
        while any(not a for a in assigned):
            stations += 1
            current_load = 0
            
            while True:
                best_task = -1
                best_prio = -np.inf
                
                for i in available:
                    if assigned[i]:
                        continue
                    if times[i] <= (cycle_time - current_load):
                        if prio_vec[i] > best_prio:
                            best_prio = prio_vec[i]
                            best_task = i
                
                if best_task == -1:
                    break
                
                assigned[best_task] = True
                current_load += times[best_task]
                available.remove(best_task)
                
                for succ in successors[best_task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        available.append(succ)
                        
        return stations

    best_prio = None
    best_stations = float('inf')
    
    for name, prio in candidates.items():
        # Check time
        if time.time() - start_time > deadline - 10:
            break
            
        stations = simulate(prio)
        if stations < best_stations:
            best_stations = stations
            best_prio = prio
            
    if best_prio is None:
        # Fallback
        return pos_weights
        
    return best_prio
