<<<<<<< SEARCH
def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Greedy station filling: repeatedly pick the highest-priority eligible task that fits.
    Returns number of stations.
    """
    n = len(times)
    if n == 0:
        return 0
    
    successors = defaultdict(list)
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    assigned = [False] * n
    eligible = set(i for i in range(n) if unassigned_preds[i] == 0)
    
    if n > 0 and not eligible:
        return 10**9
    
    station_count = 1
    current_load = 0
    
    while eligible:
        candidates = sorted(eligible, key=lambda i: -priority_scores[i])
        placed = False
        
        for task in candidates:
            if times[task] <= cycle_time - current_load:
                assigned[task] = True
                current_load += times[task]
                eligible.discard(task)
                placed = True
                
                for succ in successors[task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        eligible.add(succ)
                break
        
        if not placed:
            station_count += 1
            current_load = 0
            if any(times[t] > cycle_time for t in eligible):
                return 10**9
    
    return station_count
=======
def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Greedy station filling: repeatedly pick the highest-priority eligible task that fits.
    Returns number of stations.
    """
    n = len(times)
    if n == 0:
        return 0
    
    successors = defaultdict(list)
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Use a max-heap (negate priorities) for efficient selection
    import heapq
    eligible_heap = []
    in_heap = [False] * n
    for i in range(n):
        if unassigned_preds[i] == 0:
            heapq.heappush(eligible_heap, (-priority_scores[i], i))
            in_heap[i] = True
    
    if n > 0 and not eligible_heap:
        return 10**9
    
    station_count = 1
    current_load = 0
    placed_count = 0
    
    while placed_count < n:
        # Find the highest-priority eligible task that fits
        placed = False
        skipped = []
        
        while eligible_heap:
            neg_pri, task = heapq.heappop(eligible_heap)
            in_heap[task] = False
            
            if times[task] <= cycle_time - current_load:
                # Place this task
                current_load += times[task]
                placed_count += 1
                placed = True
                
                for succ in successors[task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        heapq.heappush(eligible_heap, (-priority_scores[succ], succ))
                        in_heap[succ] = True
                break
            else:
                # Task doesn't fit, skip it for now
                skipped.append((neg_pri, task))
        
        if not placed:
            # No task fits in current station, start new station
            station_count += 1
            current_load = 0
            
            # Check if any remaining task exceeds cycle time
            for neg_pri, task in skipped:
                if times[task] > cycle_time:
                    return 10**9
            
            # Re-add skipped tasks to heap
            for neg_pri, task in skipped:
                heapq.heappush(eligible_heap, (neg_pri, task))
                in_heap[task] = True
    
    return station_count
>>>>>>> REPLACE

