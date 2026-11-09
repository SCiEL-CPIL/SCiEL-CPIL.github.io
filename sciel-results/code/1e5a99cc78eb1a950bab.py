def get_max_discrepancy():
    while heap:
        neg_val, ap_id = heapq.heappop(heap)
        if -neg_val == abs(ap_sums[ap_id]):
            # Found valid max
            # But we popped it.
            # We need to return it, but also remember to put it back if rejected.
            # Or, we can just push it back immediately?
            # If we push it back, the heap size doesn't shrink.
            # But we need to check if there's a larger one?
            # No, heap is min-heap of -vals, so top is max.
            # So the first valid one is the max.
            # We can push it back.
            heapq.heappush(heap, (neg_val, ap_id))
            return -neg_val
    return 0
