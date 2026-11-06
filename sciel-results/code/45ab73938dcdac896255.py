import math

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order
    divisors.sort(reverse=True)
    
    num_divisors = len(divisors)
    
    # Precompute prefix sums for O(1) range sum queries
    # prefix[i] = sum(divisors[0:i])
    prefix = [0] * (num_divisors + 1)
    for i in range(num_divisors):
        prefix[i + 1] = prefix[i] + divisors[i]
        
    def range_sum(start, end):
        # Sum of divisors[start:end]
        return prefix[end] - prefix[start]

    def find_subset(k):
        current_set = []
        
        def dfs(start_idx, current_sum, current_count):
            # If we have selected k items
            if current_count == k:
                if current_sum == target:
                    return current_set[:]
                else:
                    return None
            
            # If we don't have enough items left to reach count k
            if num_divisors - start_idx < k - current_count:
                return None
                
            # Pruning: Check if we can reach target with remaining picks
            items_needed = k - current_count
            
            # The maximum sum we can add is the sum of the largest 'items_needed' divisors
            # available from start_idx onwards. Since divisors are sorted descending,
            # these are divisors[start_idx : start_idx + items_needed].
            end_idx = start_idx + items_needed
            if end_idx <= num_divisors:
                max_add = range_sum(start_idx, end_idx)
                if current_sum + max_add < target:
                    return None
            else:
                # Not enough items, already handled above, but just in case
                return None

            # Iterate through possible next divisors
            for i in range(start_idx, num_divisors):
                d = divisors[i]
                
                # If adding this divisor exceeds target, skip
                if d > target - current_sum:
                    continue
                    
                # Additional pruning: If we pick d, can we still reach target with remaining picks?
                # We have picked 1 item (d), need items_needed - 1 more.
                # Next start index is i + 1.
                # Max add from remaining is sum(divisors[i+1 : i+1 + (items_needed - 1)])
                if items_needed > 1:
                    next_start = i + 1
                    next_end = next_start + (items_needed - 1)
                    if next_end <= num_divisors:
                        next_max_add = range_sum(next_start, next_end)
                        if current_sum + d + next_max_add < target:
                            # Since divisors are sorted desc, if we can't reach target with the largest
                            # remaining divisors, we can't reach it with smaller ones either.
                            # But wait, we are iterating i. If we skip d, we try next d.
                            # The condition `current_sum + d + next_max_add < target` means even with d
                            # and the largest possible subsequent divisors, we fall short.
                            # So we can break? No, because a smaller d might allow a larger subsequent sum?
                            # No, subsequent sum is independent of d except for the index shift.
                            # If we pick a smaller d, the index i increases, so the available subsequent
                            # divisors are smaller or equal. Thus the max_add decreases.
                            # So if current d fails this check, any larger i (smaller d) will also fail.
                            # So we can break.
                            break
                    else:
                        break

                current_set.append(d)
                result = dfs(i + 1, current_sum + d, current_count + 1)
                if result is not None:
                    return result
                current_set.pop()
                
        return dfs(0, 0, 0)

    # Try to find solution with minimum number of terms
    for k in range(1, 257):
        sol = find_subset(k)
        if sol is not None:
            return sol
            
    return []
