import time

def construct():
    n = 12
    default_set = [1 << i for i in range(n)]
    best_max = max(default_set)
    best_set = default_set.copy()
    
    def find_valid_set(limit, time_limit=5):
        start_time = time.time()
        a = []
        sums = {0}
        current_sum = 0
        
        nodes_visited = 0
        max_nodes = 500000 # Adjusted for speed
        
        # Precompute max possible sum for remaining elements
        # If we have m elements left to pick, all <= limit, distinct, and > current_max
        # The max sum is sum of the m largest numbers <= limit
        # But they must be > current_max.
        
        def max_possible_remaining_sum(m, current_max_val):
            # We need m distinct integers > current_max_val and <= limit
            # To maximize sum, pick the largest available: limit, limit-1, ...
            total = 0
            val = limit
            for _ in range(m):
                if val <= current_max_val:
                    return -1 # Impossible
                total += val
                val -= 1
            return total
        
        def backtrack(idx, current_max_val):
            nonlocal nodes_visited, current_sum
            if time.time() - start_time > time_limit:
                return None
            if idx == n:
                return a.copy()
            
            nodes_visited += 1
            if nodes_visited > max_nodes:
                return None
            
            # Pruning: Check if we can reach the minimum required total sum (4095)
            # Min total sum for distinct subset sums of size n is 2^n - 1
            min_required_total_sum = (1 << n) - 1
            remaining_elements = n - idx
            max_rem_sum = max_possible_remaining_sum(remaining_elements, current_max_val)
            if max_rem_sum == -1:
                return None
            if current_sum + max_rem_sum < min_required_total_sum:
                return None
            
            start_val = current_max_val + 1 if idx > 0 else 1
            
            # Iterate candidates
            # To find a solution with small max, we might want to try smaller x first?
            # But smaller x makes it harder to satisfy distinct sums later.
            # However, we are bounded by 'limit'.
            # If we pick x too large, we might exceed limit for subsequent elements? No, limit is fixed.
            # Let's try x from start_val to limit.
            
            for x in range(start_val, limit + 1):
                # Check collision
                if x <= current_sum:
                    collision = False
                    for s in sums:
                        if s + x in sums:
                            collision = True
                            break
                    if collision:
                        continue
                
                # Valid
                a.append(x)
                old_sums = sums
                old_sum = current_sum
                
                # Update sums
                sums = set()
                for s in old_sums:
                    sums.add(s)
                    sums.add(s + x)
                
                current_sum = old_sum + x
                
                res = backtrack(idx + 1, x)
                if res is not None:
                    return res
                
                # Undo
                sums = old_sums
                current_sum = old_sum
                a.pop()
                
            return None
        
        return backtrack(0, 0)

    current_limit = best_max - 1
    # We will try to lower the limit.
    # To avoid spending too much time on impossible limits, we only try a few.
    # If 2047 fails, 2046 likely fails too? Not necessarily, but likely.
    
    while current_limit > 1000:
        sol = find_valid_set(current_limit, time_limit=5)
        if sol is not None:
            new_max = max(sol)
            if new_max < best_max:
                best_max = new_max
                best_set = sol
                current_limit = new_max - 1
            else:
                break
        else:
            # If we fail to find a solution, we assume it's not worth trying lower limits
            # immediately, or we just stop.
            # Given the time constraint, stopping is safer.
            break
            
    return best_set
