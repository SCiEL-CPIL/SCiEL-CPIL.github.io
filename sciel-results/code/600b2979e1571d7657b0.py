import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Strategy: Greedy construction with efficient mean-checking.
    # Key insight: To check if 'a' is the mean of any subset of 'others',
    # we check if 0 is achievable as a sum of a subset of (x_i - a) with size >= 2.
    # We use a DP that tracks the minimum number of elements needed to reach a sum.
    # If the min count to reach 0 is >= 2, then 'a' is the mean.
    
    def is_mean_of_subset(a, others):
        """
        Returns True if 'a' is the arithmetic mean of any subset of 'others' with size >= 2.
        Uses DP on differences (x - a).
        """
        n = len(others)
        if n < 2:
            return False
        
        # Calculate differences
        diffs = np.array([x - a for x in others], dtype=np.int32)
        
        # Range of possible sums
        # Max abs diff is 512. Max elements 48.
        # Sum range: [-512*48, 512*48] approx [-24576, 24576]
        # Let's be precise: min_sum = sum(min(0, d) for d in diffs)
        # max_sum = sum(max(0, d) for d in diffs)
        min_sum = 0
        max_sum = 0
        for d in diffs:
            if d < 0:
                min_sum += d
            else:
                max_sum += d
        
        if min_sum == 0 and max_sum == 0:
            # All diffs are 0? Impossible since others are distinct and != a.
            # But if others is empty, handled above.
            return False
        
        offset = -min_sum
        size = max_sum - min_sum + 1
        
        # dp[s] = min number of elements to reach sum s (shifted by offset)
        # Initialize with a large number (INF)
        INF = 1000
        dp = np.full(size, INF, dtype=np.int32)
        dp[offset] = 0  # Sum 0 with 0 elements
        
        # Process each difference
        for d in diffs:
            # new_dp = min(dp, dp shifted by d + 1)
            # We want to update dp in-place or use a temp array?
            # Since we are adding one element at a time, we can use a copy or careful in-place update.
            # In-place update is tricky with direction.
            # Let's use a temporary array for clarity and safety, or careful slicing.
            # Actually, standard 0/1 knapsack style:
            # If d > 0: dp[j] = min(dp[j], dp[j-d] + 1) for j from max_sum down to min_sum + d
            # If d < 0: dp[j] = min(dp[j], dp[j-d] + 1) for j from min_sum up to max_sum + d
            
            new_dp = dp.copy()
            if d > 0:
                # Shift left (higher indices)
                # new_dp[d:] = min(new_dp[d:], dp[:-d] + 1)
                # Note: indices are shifted by offset.
                # Sum s corresponds to index s + offset.
                # New sum s' = s + d corresponds to index s' + offset = s + offset + d.
                # So new_dp[index] can be updated from dp[index - d].
                new_dp[d:] = np.minimum(new_dp[d:], dp[:-d] + 1)
            elif d < 0:
                k = -d
                # Shift right (lower indices)
                # new_dp[:-k] = min(new_dp[:-k], dp[k:] + 1)
                new_dp[:-k] = np.minimum(new_dp[:-k], dp[k:] + 1)
            
            dp = new_dp
            
            # Early exit: if 0 is reachable with >= 2 elements
            # Index for sum 0 is offset.
            if dp[offset] >= 2:
                return True
        
        return False

    def check_addition(selected, x):
        """
        Check if adding x to selected keeps the set valid.
        1. x is not the mean of any subset of selected.
        2. No element a in selected is the mean of any subset of selected + {x}.
           This implies checking subsets that include x.
           Subset T of selected + {x} containing x, mean a.
           T = {x} U S', where S' is subset of selected.
           sum(T) = x + sum(S') = k * a  => sum(S') = k*a - x.
           We need to check if there exists a subset S' of selected of size k-1
           such that sum(S') = k*a - x.
           
           Alternatively, we can just run is_mean_of_subset for each a in selected
           with the list (selected + [x]) excluding a.
           This is O(N) calls to is_mean_of_subset, each O(N^2 * Range).
           With N=48, this might be slow if done for every candidate x in 1..512.
           512 * 48 * (48^2 * 25000) is too slow.
           
           We need a faster check for the incremental step.
           
           Let's refine the incremental check.
           For a fixed a in selected:
           We need to know if there is a subset of (selected + {x}) \ {a} with mean a.
           This is equivalent to: is 0 a sum of a subset of ( (selected + {x}) \ {a} - a ) with size >= 2?
           Let D_a = [y - a for y in selected if y != a]
           Let d_x = x - a
           We are adding d_x to the pool of differences.
           We maintain DP for D_a? That would be expensive to update for all a.
           
           Let's stick to the full check but optimize is_mean_of_subset.
           Or, simpler:
           Check 1: is_mean_of_subset(x, selected)
           Check 2: For each a in selected:
                      others = [y for y in selected if y != a] + [x]
                      is_mean_of_subset(a, others)
           
           Complexity:
           For each candidate x (512):
             Check 1: 1 DP call.
             Check 2: N DP calls.
             Total ~ 512 * 49 DP calls.
             Each DP call: N=48 elements, range ~25000.
             DP update: 48 * 25000 operations.
             Total ops: 512 * 49 * 48 * 25000 ~ 3 * 10^9.
             In Python/NumPy, this might be too slow (seconds to minutes).
             
           We need to prune.
           
           Optimization:
           If we process x in a specific order, maybe we can reuse info?
           Or use a faster check for small k?
           Most violations are for k=2 or k=3.
           If we check k=2 and k=3 first, we can reject most candidates quickly without full DP.
           
           Fast check for k=2:
           a is mean of {b, c} iff a = (b+c)/2 iff b+c = 2a.
           For x: is there b,c in selected such that b+c = 2x?
           For a: is there b in selected, b != a, such that a = (b+x)/2? i.e., b = 2a - x.
           
           Fast check for k=3:
           a is mean of {b, c, d} iff b+c+d = 3a.
           
           Let's implement a fast rejection for k=2 and k=3.
           If it passes, do the full DP check.
           """
        # Check 1: x is mean of subset of selected
        # Fast k=2: exists b,c in selected, b+c=2x
        # Use a set for selected for O(1) lookup
        sel_set = set(selected)
        for b in selected:
            if (2*x - b) in sel_set and (2*x - b) != b: # distinct
                return False
            # If 2x - b == b, then b=x, but x not in selected yet, so distinctness holds if x not in selected.
            # Wait, b is in selected. x is candidate.
            # If 2x - b == b, then b = x. But x is not in selected. So b != x.
            # So we just need (2x - b) in sel_set.
        
        # Fast k=3: exists b,c,d in selected, b+c+d=3x
        # This is harder to do O(N^2) without sets.
        # Let's just do the full DP for x if k=2 passes.
        if is_mean_of_subset(x, selected):
            return False
        
        # Check 2: For each a in selected, is a mean of subset of selected + {x}
        for a in selected:
            # Fast k=2: exists b in selected, b != a, such that b + x = 2a
            # b = 2a - x
            b_candidate = 2*a - x
            if b_candidate in sel_set and b_candidate != a:
                return False
            
            # If k=2 passes, check full
            # Construct others list
            # To avoid creating list every time, we can pass selected and x to a modified is_mean
            # is_mean_of_subset(a, selected + [x])
            # But we must exclude a from the "others".
            # is_mean_of_subset takes 'others' which should not contain 'a'.
            # So others = [y for y in selected if y != a] + [x]
            
            # Optimization: Only run full DP if N is small or if we suspect it's close.
            # Given the cost, let's try to run it.
            # If it's too slow, we might need to skip full DP for large N and rely on k=2,3?
            # But the task requires exactness.
            
            # Let's try running it.
            others = [y for y in selected if y != a]
            others.append(x)
            if is_mean_of_subset(a, others):
                return False
        
        return True

    def greedy_construct(order):
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
            if check_addition(selected, x):
                selected.append(x)
        return selected

    # Try different orderings
    best_set = []
    
    # 1. Ascending
    s1 = greedy_construct(range(1, 513))
    if len(s1) > len(best_set):
        best_set = s1
    
    # 2. Descending
    s2 = greedy_construct(range(512, 0, -1))
    if len(s2) > len(best_set):
        best_set = s2
    
    # 3. Random restarts if time permits
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            order = list(range(1, 513))
            random.shuffle(order)
            s = greedy_construct(order)
            if len(s) > len(best_set):
                best_set = s
    
    return best_set[:48]

if __name__ == "__main__":
    s = construct()
    print(len(s))
