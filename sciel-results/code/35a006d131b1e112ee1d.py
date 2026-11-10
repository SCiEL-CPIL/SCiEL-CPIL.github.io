import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    n = 5040
    target = 4979
    
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.sort(reverse=True)
    num_divs = len(divs)
    
    # Prefix sums for fast range sum queries
    # prefix[i] is sum of divs[0..i-1]
    prefix = [0] * (num_divs + 1)
    for i in range(num_divs):
        prefix[i+1] = prefix[i] + divs[i]
        
    def range_sum(start, end):
        # sum of divs[start:end]
        return prefix[end] - prefix[start]

    for k in range(1, num_divs + 1):
        if time.time() > deadline:
            break
            
        if k == 1:
            if target in divs:
                return [target]
            continue
            
        if k == 2:
            seen = set()
            for d in divs:
                if d in seen:
                    continue
                rem = target - d
                if rem in divs and rem != d:
                    return [d, rem]
                seen.add(d)
            continue

        # DFS for k >= 3
        def dfs(start_idx, remaining_sum, slots_left, current_path):
            if time.time() > deadline:
                return None
                
            if slots_left == 0:
                if remaining_sum == 0:
                    return list(current_path)
                return None
                
            if remaining_sum < 0:
                return None
                
            # Pruning: Max possible sum
            # We pick 'slots_left' largest items from divs[start_idx:]
            # These are divs[start_idx : start_idx + slots_left]
            if start_idx + slots_left > num_divs:
                return None
            max_possible = range_sum(start_idx, start_idx + slots_left)
            if max_possible < remaining_sum:
                return None
                
            # Pruning: Min possible sum
            # We pick 'slots_left' smallest items from divs[start_idx:]
            # These are the last 'slots_left' items in the array, 
            # provided they are >= start_idx.
            # Since we are in a state where we can still pick slots_left items,
            # the smallest available items are at the end of the array.
            # Indices: num_divs - slots_left to num_divs.
            # We must ensure num_divs - slots_left >= start_idx.
            # This is guaranteed by the loop bound in the caller? 
            # In the caller: i goes up to num_divs - slots_left.
            # So start_idx (which is i+1) <= num_divs - slots_left + 1.
            # If start_idx == num_divs - slots_left + 1, then we have exactly slots_left items left?
            # No, if start_idx is that high, the slice divs[start_idx:] has length num_divs - start_idx.
            # We need slots_left items.
            # The smallest slots_left items in divs[start_idx:] are divs[num_divs-slots_left : num_divs]
            # IF num_divs - slots_left >= start_idx.
            # If num_divs - slots_left < start_idx, it means the "global" smallest items are not available.
            # In that case, the smallest available are just the last slots_left items of the available slice?
            # Actually, if the slice is divs[start_idx:], and it's sorted desc, the smallest are at the end.
            # The end is num_divs-1.
            # So we always take divs[num_divs-slots_left : num_divs] IF those indices are >= start_idx.
            # If not, we take divs[start_idx : start_idx + slots_left]? No, those are the LARGEST.
            # If the range of indices [num_divs-slots_left, num_divs) is not fully contained in [start_idx, num_divs),
            # it means start_idx > num_divs - slots_left.
            # This implies num_divs - start_idx < slots_left.
            # But we know we have enough items to pick slots_left, so num_divs - start_idx >= slots_left.
            # Therefore num_divs - slots_left >= start_idx is ALWAYS TRUE in valid DFS states.
            
            min_start_idx = num_divs - slots_left
            min_possible = range_sum(min_start_idx, num_divs)
            if min_possible > remaining_sum:
                return None

            # Iterate candidates
            # i is the index of the next item to pick.
            # We must leave enough items for the remaining slots.
            # So i can go up to num_divs - slots_left.
            for i in range(start_idx, num_divs - slots_left + 1):
                d = divs[i]
                if d > remaining_sum:
                    continue
                    
                current_path.append(d)
                res = dfs(i + 1, remaining_sum - d, slots_left - 1, current_path)
                if res is not None:
                    return res
                current_path.pop()
                
            return None

        result = dfs(0, target, k, [])
        if result is not None:
            return result

    # Fallback
    remainder = target
    chosen = []
    for d in divs:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
