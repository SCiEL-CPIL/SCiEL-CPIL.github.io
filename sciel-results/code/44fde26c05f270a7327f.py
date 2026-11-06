# EVOLVE-BLOCK-START
from collections import deque

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort divisors in ascending order to enforce distinctness via index ordering
    divisors.sort()
    num_divs = len(divisors)
    
    # We want to find the minimum number of divisors summing to target.
    # State: (current_sum, last_index)
    # last_index is the index of the last divisor picked. 
    # The next divisor must have an index > last_index.
    # We use BFS because each step adds 1 to the count (unit weight).
    
    # dist[sum][last_idx] stores the minimum number of divisors used to reach this state.
    # We also need to reconstruct the path, so we store the parent state.
    # parent[sum][last_idx] = (prev_sum, prev_last_idx)
    
    # Initialize
    # Using -1 for last_idx to represent "no divisor picked yet"
    # We can shift indices by 1 so -1 becomes 0, or just handle it separately.
    # Let's use a dictionary for sparse access or a 2D list.
    # Max sum is target. Max index is num_divs - 1.
    # Let's use a 2D list of size (target+1) x (num_divs + 1).
    # last_idx can be -1 to num_divs-1. Map -1 to 0, i to i+1.
    
    INF = float('inf')
    # dist[s][i] where i is 0..num_divs
    # i=0 corresponds to last_idx = -1
    # i=k corresponds to last_idx = k-1
    
    dist = [[INF] * (num_divs + 1) for _ in range(target + 1)]
    parent = [[(-1, -1) for _ in range(num_divs + 1)] for _ in range(target + 1)]
    
    # Start state: sum=0, last_idx=-1 (mapped to 0)
    dist[0][0] = 0
    queue = deque()
    queue.append((0, 0)) # (sum, mapped_last_idx)
    
    # BFS
    while queue:
        s, last_idx_mapped = queue.popleft()
        count = dist[s][last_idx_mapped]
        
        # If we reached the target, we can stop
        if s == target:
            # Reconstruct path
            path = []
            cur_s = s
            cur_last = last_idx_mapped
            while cur_s != 0:
                prev_s, prev_last = parent[cur_s][cur_last]
                # The divisor picked was the one that led from prev_last to cur_last
                # cur_last is the mapped index of the divisor picked.
                # The actual index is cur_last - 1.
                div_idx = cur_last - 1
                path.append(divisors[div_idx])
                cur_s = prev_s
                cur_last = prev_last
            return path[::-1] # Reverse to get correct order (though order doesn't matter for sum)
        
        # Determine the minimum index we can pick next
        # If last_idx_mapped is 0 (last_idx -1), we can pick any index 0..num_divs-1
        # If last_idx_mapped is k (last_idx k-1), we can pick any index k..num_divs-1
        start_idx = last_idx_mapped
        
        for i in range(start_idx, num_divs):
            d = divisors[i]
            new_s = s + d
            if new_s > target:
                break # Since divisors are sorted, further divisors will be larger
            
            new_last_mapped = i + 1
            
            # If we haven't visited this state with a lower or equal count
            if count + 1 < dist[new_s][new_last_mapped]:
                dist[new_s][new_last_mapped] = count + 1
                parent[new_s][new_last_mapped] = (s, last_idx_mapped)
                queue.append((new_s, new_last_mapped))
                
    # If no solution found (should not happen for this problem instance)
    return []
# EVOLVE-BLOCK-END
