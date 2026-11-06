import time
import sys

def construct():
    n = 5040
    target = 4979
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # Get divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    
    # Filter out divisors larger than target
    divisors = [d for d in divisors if d <= target]
    N = len(divisors)
    
    # Precompute prefix sums for max sum pruning
    # prefix_sums[i] = sum(divisors[0:i])
    prefix_sums = [0] * (N + 1)
    for i in range(N):
        prefix_sums[i+1] = prefix_sums[i] + divisors[i]
        
    # Precompute sums of smallest j divisors for min sum pruning
    # divisors is sorted descending, so smallest are at the end.
    # small_sums[j] = sum of j smallest divisors = sum(divisors[N-j:N])
    small_sums = [0] * (N + 1)
    # We can compute this from prefix sums or just iterate
    # sum(divisors[N-j:N]) = prefix_sums[N] - prefix_sums[N-j]
    for j in range(N + 1):
        small_sums[j] = prefix_sums[N] - prefix_sums[N-j]
        
    # Search for minimum k
    for k in range(1, N + 1):
        # Check if it's even possible
        # Max sum with k divisors is sum of largest k: prefix_sums[k]
        if prefix_sums[k] < target:
            continue
            
        # Min sum with k divisors is sum of smallest k: small_sums[k]
        if small_sums[k] > target:
            # Since we are increasing k, and min sum increases with k?
            # Wait. Min sum of k distinct divisors.
            # As k increases, we must include more divisors.
            # The min sum of k+1 divisors is min_sum_k + next_smallest_divisor?
            # Yes, min sum is non-decreasing with k.
            # If min_sum[k] > target, then min_sum[k+1] > target too.
            # So we can break?
            # Not necessarily, because we are looking for *exactly* target.
            # If the minimum possible sum for k terms is already > target,
            # then no subset of size k can sum to target.
            # And for any k' > k, the min sum will be even larger.
            # So we can break the loop.
            break
            
        # DFS
        def dfs(idx, remaining, current_count):
            if remaining == 0:
                return True
            if current_count == k:
                return False
            if idx >= N:
                return False
            if remaining < 0:
                return False
                
            # Time check
            if time.time() > deadline:
                return False
                
            slots = k - current_count
            
            # Pruning 1: Not enough elements left
            if N - idx < slots:
                return False
                
            # Pruning 2: Max possible sum with 'slots' elements from idx
            # Largest 'slots' elements in divisors[idx:] are divisors[idx : idx+slots]
            max_possible_sum = prefix_sums[idx + slots] - prefix_sums[idx]
            if max_possible_sum < remaining:
                return False
                
            # Pruning 3: Min possible sum with 'slots' elements from idx
            # Smallest 'slots' elements in divisors[idx:] are the global smallest 'slots'
            # provided idx <= N - slots.
            # We checked N - idx >= slots, so idx <= N - slots.
            min_possible_sum = small_sums[slots]
            if min_possible_sum > remaining:
                return False

            # Include divisors[idx]
            if dfs(idx + 1, remaining - divisors[idx], current_count + 1):
                return True
                
            # Exclude divisors[idx]
            if dfs(idx + 1, remaining, current_count):
                return True
                
            return False

        # We need to reconstruct the path if found.
        # The boolean DFS is fast, but we need the actual divisors.
        # We can modify DFS to return the list or use a backtracking variable.
        
        def dfs_reconstruct(idx, remaining, current_count, path):
            if remaining == 0:
                return True
            if current_count == k:
                return False
            if idx >= N:
                return False
            if remaining < 0:
                return False
                
            # Time check
            if time.time() > deadline:
                return False
                
            slots = k - current_count
            
            if N - idx < slots:
                return False
                
            max_possible_sum = prefix_sums[idx + slots] - prefix_sums[idx]
            if max_possible_sum < remaining:
                return False
                
            min_possible_sum = small_sums[slots]
            if min_possible_sum > remaining:
                return False

            # Include divisors[idx]
            path.append(divisors[idx])
            if dfs_reconstruct(idx + 1, remaining - divisors[idx], current_count + 1, path):
                return True
            path.pop()
                
            # Exclude divisors[idx]
            if dfs_reconstruct(idx + 1, remaining, current_count, path):
                return True
                
            return False

        path = []
        if dfs_reconstruct(0, target, 0, path):
            return path
            
    # Fallback (should not be reached if solution exists)
    return []
