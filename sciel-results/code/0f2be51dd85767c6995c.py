# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import deque

def _simulate_station_count(times, predecessors, cycle_time, priorities):
    """Simulate greedy assignment using a max-heap for speed."""
    n = len(times)
    # Build successor list
    successors = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i]
        pred_count[i] = len(preds)
        for p in preds:
            successors[p].append(i)
    
    # Track remaining predecessors for each task
    remaining_preds = pred_count[:]
    
    # Max-heap: store (-priority, task_id) for max-heap behavior
    heap = []
    for i in range(n):
        if remaining_preds[i] == 0:
            heapq.heappush(heap, (-priorities[i], i))
    
    station_count = 0
    remaining_time = 0
    assigned = [False] * n
    assigned_count = 0
    
    while assigned_count < n:
        if remaining_time == 0:
            station_count += 1
            remaining_time = cycle_time
        
        # Find best available task that fits
        best_task = -1
        best_neg_priority = np.inf
        temp_heap = []
        
        # Extract all from heap, find best that fits, put back others
        while heap:
            neg_prio, t = heapq.heappop(heap)
            if assigned[t]:
                continue
            if times[t] <= remaining_time:
                if neg_prio < best_neg_priority:
                    best_neg_priority = neg_prio
                    best_task = t
                    # Don't put this one back
                else:
                    temp_heap.append((neg_prio, t))
            else:
                temp_heap.append((neg_prio, t))
        
        # Put back tasks that didn't fit or weren't best
        for item in temp_heap:
            heapq.heappush(heap, item)
        
        if best_task == -1:
            # No task fits in current station, open new one
            station_count += 1
            remaining_time = cycle_time
            # Try again with full cycle time
            best_task = -1
            best_neg_priority = np.inf
            temp_heap = []
            while heap:
                neg_prio, t = heapq.heappop(heap)
                if assigned[t]:
                    continue
                if times[t] <= remaining_time:
                    if neg_prio < best_neg_priority:
                        best_neg_priority = neg_prio
                        best_task = t
                    else:
                        temp_heap.append((neg_prio, t))
                else:
                    temp_heap.append((neg_prio, t))
            for item in temp_heap:
                heapq.heappush(heap, item)
            
            if best_task == -1:
                # Force assign the smallest unassigned task
                min_t = min((t for t in range(n) if not assigned[t]), key=lambda x: times[x])
                best_task = min_t
                # Remove from heap if present (lazy deletion is fine)
        
        # Assign best_task
        assigned[best_task] = True
        assigned_count += 1
        remaining_time -= times[best_task]
        
        # Update successors
        for s in successors[best_task]:
            remaining_preds[s] -= 1
            if remaining_preds[s] == 0:
                heapq.heappush(heap, (-priorities[s], s))
    
    return station_count


def _compute_ranks(times, predecessors):
    """Compute rank for each task: longest path from task to end."""
    n = len(times)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Topological order using Kahn's algorithm
    in_deg = [len(predecessors[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for s in successors[u]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    
    # Longest path from node to end: process in reverse topo order
    longest = [0.0] * n
    for u in reversed(topo_order):
        if successors[u]:
            longest[u] = times[u] + max(longest[s] for s in successors[u])
        else:
            longest[u] = times[u]
    
    return longest, topo_order


def priority(instance):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Compute ranks and topological order
    ranks, topo_order = _compute_ranks(times, predecessors)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    out_degree = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Positional rank: position in topological order (earlier = higher priority)
    pos_rank = np.zeros(n)
    for pos, task in enumerate(topo_order):
        pos_rank[task] = (n - 1 - pos)  # Earlier tasks get higher values
    
    # Initial priority: combination of rank and time
    base_priorities = ranks + times
    
    # Evaluate initial
    best_score = _simulate_station_count(times, predecessors, cycle_time, base_priorities)
    best_priorities = base_priorities.copy()
    
    # Generate diverse analytical candidates
    candidates = []
    
    # Scheme 1: rank (longest path to end)
    candidates.append(ranks.copy())
    
    # Scheme 2: time only (LPT)
    candidates.append(times.copy())
    
    # Scheme 3: rank + alpha * time for various alpha
    for alpha in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        candidates.append(ranks + alpha * times)
    
    # Scheme 4: rank * beta + time
    for beta in [0.5, 1.0, 1.5, 2.0]:
        candidates.append(beta * ranks + times)
    
    # Scheme 5: number of successors (out-degree) bonus
    candidates.append(ranks + out_degree)
    candidates.append(ranks + 2.0 * out_degree)
    
    # Scheme 6: rank * (1 + out_degree)
    candidates.append(ranks * (1.0 + out_degree))
    candidates.append(ranks * (1.0 + 2.0 * out_degree))
    
    # Scheme 7: positional rank
    candidates.append(pos_rank.copy())
    candidates.append(pos_rank + times)
    candidates.append(pos_rank + ranks)
    candidates.append(0.5 * pos_rank + 0.5 * ranks)
    
    # Scheme 8: rank + out_degree * time
    candidates.append(ranks + out_degree * times)
    
    # Scheme 9: weighted combination of all features
    # Normalize features
    if np.max(ranks) > 0:
        ranks_norm = ranks / np.max(ranks)
    else:
        ranks_norm = ranks
    if np.max(times) > 0:
        times_norm = times / np.max(times)
    else:
        times_norm = times
    if np.max(out_degree) > 0:
        od_norm = out_degree / np.max(out_degree)
    else:
        od_norm = out_degree
    if np.max(pos_rank) > 0:
        pos_norm = pos_rank / np.max(pos_rank)
    else:
        pos_norm = pos_rank
    
    # Grid search over weights
    for w1 in [0.2, 0.4, 0.6, 0.8, 1.0]:
        for w2 in [0.0, 0.2, 0.4, 0.6]:
            for w3 in [0.0, 0.2, 0.4]:
                if w1 + w2 + w3 > 0:
                    cand = w1 * ranks_norm + w2 * times_norm + w3 * od_norm
                    candidates.append(cand)
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        score = _simulate_station_count(times, predecessors, cycle_time, cand)
        if score < best_score:
            best_score = score
            best_priorities = cand.copy()
    
    # Local search: perturb best_priorities with adaptive scales
    rng = np.random.default_rng(42)
    
    improved = True
    pass_count = 0
    while improved and (time.time() < deadline - 10):
        improved = False
        pass_count += 1
        
        # Adaptive perturbation scale: start large, decrease
        scale = max(0.1, 2.0 / (1 + 0.5 * pass_count))
        
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            
            # Perturb: pick a random task and adjust its priority
            i = rng.integers(0, n)
            perturbation = rng.normal(0, scale)
            new_priorities = best_priorities.copy()
            new_priorities[i] += perturbation
            
            score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
            if score < best_score:
                best_score = score
                best_priorities = new_priorities
                improved = True
    
    # Try pair swaps as well
    if time.time() < deadline - 10:
        improved = True
        while improved and (time.time() < deadline - 10):
            improved = False
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                i, j = rng.integers(0, n, size=2)
                if i == j:
                    continue
                new_priorities = best_priorities.copy()
                # Swap priorities
                new_priorities[i], new_priorities[j] = new_priorities[j], new_priorities[i]
                score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
                if score < best_score:
                    best_score = score
                    best_priorities = new_priorities
                    improved = True
    
    # Final polish: try small perturbations with very small scale
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            i = rng.integers(0, n)
            perturbation = rng.normal(0, 0.05)
            new_priorities = best_priorities.copy()
            new_priorities[i] += perturbation
            score = _simulate_station_count(times, predecessors, cycle_time, new_priorities)
            if score < best_score:
                best_score = score
                best_priorities = new_priorities
    
    return best_priorities
# EVOLVE-BLOCK-END
