import math

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort()
    N = len(divisors)
    
    # To speed up membership checks, we use a set
    divisor_set = set(divisors)
    
    # We search for the minimum number of divisors k
    # k can range from 1 to N
    for k in range(1, N + 1):
        # Recursive backtracking to find k distinct divisors summing to target
        # We pass the index to start to ensure distinctness (no reuse of earlier indices)
        # Actually, since we just need distinct values, and we iterate from start_idx,
        # we naturally pick distinct values from the sorted list.
        
        def search(start_idx, current_sum, selected):
            if len(selected) == k:
                if current_sum == target:
                    return selected
                return None
            
            remaining_needed = k - len(selected)
            
            # Pruning 1: Can we reach the target?
            # Max sum we can add is the sum of the largest 'remaining_needed' divisors
            # starting from start_idx.
            # Since divisors are sorted ascending, the largest are at the end.
            # We can pick any from start_idx to N-1.
            # To maximize sum, pick the largest available.
            # But we are constrained by start_idx? No, we can pick any index >= start_idx.
            # So the max sum is sum(divisors[N-remaining_needed : N])? 
            # Wait, we can't pick indices < start_idx.
            # So max sum is sum of the largest 'remaining_needed' elements in divisors[start_idx:].
            max_sum_add = sum(divisors[N - remaining_needed : N]) # This assumes we can pick the global largest.
            # But we must pick from start_idx onwards.
            # If start_idx > N - remaining_needed, we can't pick the global largest.
            # Correct max sum:
            if N - start_idx < remaining_needed:
                return None # Not enough divisors left
            
            # The largest 'remaining_needed' divisors available are the last 'remaining_needed'
            # in the slice divisors[start_idx:].
            # indices: N - remaining_needed to N - 1?
            # Only if N - remaining_needed >= start_idx.
            # If N - remaining_needed < start_idx, we take from start_idx to N-1? No, we need exactly remaining_needed.
            # So we take divisors[N-remaining_needed : N] if N-remaining_needed >= start_idx.
            # Otherwise, we can't form a sum this large? No, we just take the available ones.
            # Actually, if N - start_idx >= remaining_needed, the max sum is sum(divisors[N-remaining_needed:N])
            # provided that N-remaining_needed >= start_idx.
            # If N-remaining_needed < start_idx, it means we are forced to pick some smaller ones?
            # No, we can pick ANY from start_idx to N-1.
            # The largest 'remaining_needed' in that range are the last 'remaining_needed' of that range.
            # i.e., divisors[N-remaining_needed : N] is correct ONLY IF N-remaining_needed >= start_idx.
            # If N-remaining_needed < start_idx, then the largest available are divisors[start_idx : N] ? No, we need exactly remaining_needed.
            # So we take the last 'remaining_needed' of divisors[start_idx:].
            # Slice: divisors[N-remaining_needed : N] works if start_idx <= N-remaining_needed.
            # If start_idx > N-remaining_needed, then we must pick from start_idx to N-1, but there are N-start_idx < remaining_needed elements?
            # We already checked N - start_idx >= remaining_needed.
            # So the largest 'remaining_needed' are always the last 'remaining_needed' of the available slice.
            # Available slice is divisors[start_idx:N].
            # Last 'remaining_needed' of this slice are divisors[N-remaining_needed : N].
            # Is N-remaining_needed >= start_idx?
            # Yes, because N - start_idx >= remaining_needed => N - remaining_needed >= start_idx.
            
            max_sum_add = sum(divisors[N - remaining_needed : N])
            
            if current_sum + max_sum_add < target:
                return None

            # Pruning 2: Do we exceed the target?
            # Min sum we can add is the sum of the smallest 'remaining_needed' divisors
            # starting from start_idx.
            min_sum_add = sum(divisors[start_idx : start_idx + remaining_needed])
            
            if current_sum + min_sum_add > target:
                return None
            
            # Iterate through possible next divisors
            # We can pick any divisor from start_idx to N - remaining_needed
            # (we must leave enough divisors for the remaining slots)
            for i in range(start_idx, N - remaining_needed + 1):
                d = divisors[i]
                # Optional: if current_sum + d > target, we can break?
                # Since divisors are sorted, yes.
                if current_sum + d > target:
                    break
                
                # Recurse
                result = search(i + 1, current_sum + d, selected + [d])
                if result is not None:
                    return result
            
            return None

        solution = search(0, 0, [])
        if solution is not None:
            return solution
            
    return [] # Should not happen
