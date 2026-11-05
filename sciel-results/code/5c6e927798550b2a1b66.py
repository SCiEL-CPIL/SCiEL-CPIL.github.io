"""
Assembly Line Balancing via Simulated Annealing with Critical Path Density Initialization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math
import heapq

def _simulate_greedy(order, times, succ, pred_counts_init, cycle_time):
    """
    Simulate the greedy packing process for a given priority order.
    Uses a heap of (rank, task) for O(log n) selection of best eligible task.
    """
    n = len(times)
    pred_counts = pred_counts_init.copy()
    
    # Rank array for quick lookup of priority
    rank = [0] * n
    for idx, task in enumerate(order):
        rank[task] = idx
        
    # Initial available tasks in heap keyed by rank
    heap = [(rank[i], i) for i in range(n) if pred_counts[i] == 0]
    heapq.heapify(heap)
    
    stations = 0
    current_load = 0.0
    assigned_count = 0
    
    while assigned_count < n:
        remaining_space = cycle_time - current_load
        
        # Find best task in available that fits
        best_task = -1
        best_rank = n + 1
        deferred = []
        
        while heap:
            r, t = heap[0]
            if times[t] <= remaining_space + 1e-9:
                heapq.heappop(heap)
                best_task = t
                break
            else:
                # Task doesn't fit; defer it
                deferred.append((r, t))
                heapq.heappop(heap)
        
        if best_task != -1:
            assigned_count += 1
            current_load += times[best_task]
            
            # Update successors
            for s in succ[best_task]:
                pred_counts[s] -= 1
                if pred_counts[s] == 0:
                    heapq.heappush(heap, (rank[s], s))
            
            # Put back deferred tasks
            for item in deferred:
                heapq.heappush(heap, item)
        else:
            if current_load > 1e-9:
                stations += 1
                current_load = 0.0
            else:
                break

    if current_load > 1e-9:
        stations += 1
        
    return stations

def priority(instance):
    """
    Uses Simulated Annealing initialized with a Critical Ratio heuristic.
    Critical Ratio = (Downstream Workload) / (Own Time).
    High ratio tasks are bottlenecks that should be prioritized to prevent idle time.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor graph
    pred = instance["predecessors"]
    succ = [[] for _ in range(n)]
    pred_counts_init = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
            pred_counts_init[i] += 1
            
    # --- Topological Sort ---
    in_degree = pred_counts_init.copy()
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    while queue:
        queue.sort()
        u = queue.pop(0)
        top_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(top_order) != n:
        # Cycle detected, fallback
        return times.copy()
        
    # --- Compute Metrics ---
    # 1. Longest Path (Critical Path)
    # 2. Total Downstream Workload (Sum of times of all transitive successors)
    
    dist = np.zeros(n, dtype=float)
    total_downstream = np.zeros(n, dtype=float)
    
    for u in reversed(top_order):
        max_succ_dist = 0.0
        sum_succ_downstream = 0.0
        for v in succ[u]:
            if dist[v] > max_succ_dist:
                max_succ_dist = dist[v]
            sum_succ_downstream += times[v] + total_downstream[v]
        
        dist[u] = times[u] + max_succ_dist
        total_downstream[u] = sum_succ_downstream
        
    # --- Critical Ratio ---
    # We want to prioritize tasks that have a large downstream workload relative to their own time.
    # This is the "bottleneck" intuition.
    # CR = total_downstream / (times + epsilon)
    # We also combine with dist (critical path) to ensure we don't ignore long chains.
    
    # Let's create a composite score for initialization.
    # Primary: Critical Ratio (downstream workload / own time)
    # Secondary: Dist (critical path length)
    # Tertiary: Own time (larger tasks first is often good for packing, but CR captures this inversely)
    
    # To avoid division by zero, add epsilon
    eps = 1e-6
    cr_scores = total_downstream / (times + eps)
    
    # Normalize scores for sorting? Or just use raw values.
    # Let's try a weighted combination.
    # Weight 1: CR
    # Weight 2: Dist
    
    # Actually, let's just sort by CR descending first, then Dist descending.
    # This focuses on "bottleneck" tasks.
    
    initial_order = sorted(range(n), key=lambda x: (-cr_scores[x], -dist[x], -times[x]))
    
    # Evaluate initial
    best_order = initial_order.copy()
    best_score = _simulate_greedy(best_order, times, succ, pred_counts_init, cycle_time)
    
    # Simulated Annealing
    current_order = best_order.copy()
    current_score = best_score
    T = 2.0
    cooling_rate = 0.995
    min_T = 0.05
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Choose move type
        if random.random() < 0.4:
            # Swap
            new_order = current_order.copy()
            i, j = random.sample(range(n), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            # Insertion
            new_order = current_order.copy()
            i, j = random.sample(range(n), 2)
            if i == j: continue
            item = new_order.pop(i)
            new_order.insert(j, item)
            
        new_score = _simulate_greedy(new_order, times, succ, pred_counts_init, cycle_time)
        
        if new_score < current_score:
            current_order = new_order
            current_score = new_score
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_order = current_order.copy()
        else:
            delta = new_score - current_score
            if delta > 0:
                if T < 1e-6:
                    prob = 0.0
                else:
                    prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_order = new_order
                    current_score = new_score
            no_improve_count += 1
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        # Adaptive restart
        if no_improve_count > max_no_improve:
            current_order = best_order.copy()
            current_score = best_score
            T = 1.0
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 1.5)
    
    # Local polish: try all single-position insertions from best_order
    # Use time-aware loop
    for _ in range(10):  # More passes if time allows
        if time.time() > deadline - 5.0:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            for j in range(n):
                if i == j:
                    continue
                new_order = best_order.copy()
                item = new_order.pop(i)
                new_order.insert(j, item)
                new_score = _simulate_greedy(new_order, times, succ, pred_counts_init, cycle_time)
                if new_score < best_score:
                    best_score = new_score
                    best_order = new_order
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Convert best_order to scores
    scores = np.zeros(n)
    for idx, task in enumerate(best_order):
        scores[task] = n - idx
        
    return scores
# EVOLVE-BLOCK-END
