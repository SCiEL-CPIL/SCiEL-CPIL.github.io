<<<<<<< SEARCH
def _add_back_search(points, primes_to_check, deadline):
    """Add-back search using incremental tracker with safety-scored ordering."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    all_candidates = set(range(N + 1)) - tracker.point_set
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Compute safety scores for all candidates
        # Use a heap for efficient lowest-score-first selection
        import heapq
        heap = []
        for v in all_candidates:
            s = tracker.safety_score(v)
            heapq.heappush(heap, (s, v))
        
        while heap and time.time() < deadline:
            s, v = heapq.heappop(heap)
            if v not in all_candidates:
                continue
            if tracker.can_add(v):
                tracker.add(v)
                all_candidates.discard(v)
                improved = True
                # After adding, some previously safe candidates may become unsafe
                # but we'll re-evaluate in the next outer loop iteration
                break  # Restart with fresh safety scores
    
    return list(tracker.point_set)
=======
def _add_back_search(points, primes_to_check, deadline):
    """Add-back search using incremental tracker with safety-scored ordering and local optimization."""
    tracker = _AdmissibleTracker(points, primes_to_check)
    all_candidates = set(range(N + 1)) - tracker.point_set
    
    # Phase 1: Greedy add-back with safety scoring
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Compute safety scores for all candidates
        # Use a heap for efficient lowest-score-first selection
        import heapq
        heap = []
        for v in all_candidates:
            s = tracker.safety_score(v)
            heapq.heappush(heap, (s, v))
        
        while heap and time.time() < deadline:
            s, v = heapq.heappop(heap)
            if v not in all_candidates:
                continue
            if tracker.can_add(v):
                tracker.add(v)
                all_candidates.discard(v)
                improved = True
                # After adding, some previously safe candidates may become unsafe
                # but we'll re-evaluate in the next outer loop iteration
                break  # Restart with fresh safety scores
    
    # Phase 2: Local optimization - try swapping one point for multiple
    # For each point in the current set, check if removing it allows adding 2+ new points
    if time.time() < deadline - 10:
        current_set = set(tracker.point_set)
        candidates = list(all_candidates)
        
        # Precompute safety scores for candidates
        candidate_safety = {v: tracker.safety_score(v) for v in candidates}
        
        # Try each point as a potential removal candidate
        # Sort by how "expensive" it is to remove (high safety score = more risky to remove)
        point_safety = {v: tracker.safety_score(v) for v in current_set}
        sorted_points = sorted(current_set, key=lambda v: -point_safety[v])
        
        for v_to_remove in sorted_points:
            if time.time() > deadline - 10:
                break
            
            # Temporarily remove v_to_remove
            tracker.remove(v_to_remove)
            current_set.discard(v_to_remove)
            
            # Find how many candidates can now be added
            added = []
            for v in sorted(candidates, key=lambda x: candidate_safety.get(x, float('inf'))):
                if v in current_set:
                    continue
                if tracker.can_add(v):
                    tracker.add(v)
                    current_set.add(v)
                    added.append(v)
                if len(added) > 1:
                    break
            
            # If we added more than we removed, keep the improvement
            if len(added) > 1:
                # Update candidates list
                candidates = [v for v in candidates if v not in current_set]
                for v in added:
                    candidate_safety[v] = tracker.safety_score(v)
            else:
                # Roll back
                for v in added:
                    tracker.remove(v)
                    current_set.discard(v)
                tracker.add(v_to_remove)
                current_set.add(v_to_remove)
    
    return list(tracker.point_set)
>>>>>>> REPLACE

