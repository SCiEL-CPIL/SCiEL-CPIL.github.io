import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 90s limit
    
    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Compute LCM of 2..48
    L = reduce(lcm, range(2, 49))
    
    # Precompute integer weights
    # index i corresponds to denominator i+2
    # weight[i] = L // (i+2)
    weights = [L // d for d in denominators]
    
    # Meet-in-the-Middle Strategy
    # Split the list of items into two halves.
    # Since n=47, we split into 23 and 24.
    # 2^24 is 16,777,216. This is large but manageable if we prune aggressively.
    # However, 16 million iterations in Python might be slow.
    # Let's split into smaller chunks to keep enumeration fast, or use a smarter split.
    # Actually, 2^24 is too big for a simple loop in Python within 55s if we do it naively.
    # Let's split into 3 parts? Or 2 parts with 20/27?
    # 2^20 = 1,048,576. 2^27 = 134,217,728 (too big).
    # Let's try 20 and 27? No, 27 is too big.
    # How about 22 and 25? 2^22 = 4M, 2^25 = 33M. Still risky.
    # Let's stick to the current B&B but improve the heuristic? 
    # The prompt asks for a DIFFERENT strategy.
    
    # Let's try a different angle: 
    # The deficit is 1 - sum(1/d_i). 
    # This is equivalent to finding a subset S such that sum_{i in S} (L/d_i) is as close to L as possible from below.
    # 
    # Alternative Strategy: Local Search / Hill Climbing with Random Restarts.
    # 1. Start with a greedy solution (take items in descending order of weight until next would exceed L).
    # 2. Perform local swaps: try replacing one selected item with one unselected item to improve the sum (get closer to L).
    # 3. Or try adding an item and removing others.
    # 4. Random restarts to escape local optima.
    #
    # Given the structure of unit fractions, the "gap" structure might be exploitable.
    # However, B&B is exact. MITM is exact. Local search is heuristic.
    # The current B&B score is ~1.8e6. The seed score was 1.0. 
    # The best possible score is related to the minimal deficit.
    # If the minimal deficit is 1/K, the score is K.
    # Current score 1.8e6 implies deficit ~ 1/1.8e6.
    # 
    # Let's implement a **Simulated Annealing** approach.
    # It is a different design family (stochastic global optimization vs deterministic search tree).
    # It can potentially jump over local optima that trap greedy or simple B&B if B&B prunes incorrectly (though B&B with correct bound is exact, it might time out before finding the optimum).
    # Wait, B&B with correct bound IS exact. If it times out, it returns the best found so far.
    # If the optimal solution is deep in the tree, B&B might not find it.
    # Simulated Annealing explores the space randomly and can find deep solutions.
    
    # Let's implement Simulated Annealing.
    
    def sa_search():
        # Initial solution: Greedy
        # Sort indices by weight descending (already sorted in weights list)
        # But we want to maximize sum < L.
        # Greedy: take items in order of largest weight, stop if next makes sum >= L.
        current_sum = 0
        current_set = set()
        for i in range(n):
            if current_sum + weights[i] < L:
                current_sum += weights[i]
                current_set.add(i)
        
        best_sum = current_sum
        best_set = set(current_set)
        
        # Annealing parameters
        T = 1000.0
        cooling_rate = 0.999
        min_T = 0.1
        max_iterations = 200000
        
        import random
        
        # Precompute list of items for faster access
        # State is represented by a bitmask or a set. 
        # Using a set is slow for swaps. 
        # Let's use a boolean array or list.
        selected = [False] * n
        for i in current_set:
            selected[i] = True
            
        def evaluate(selected_list):
            s = 0
            for i in range(n):
                if selected_list[i]:
                    s += weights[i]
            return s
        
        current_selected = selected[:]
        current_eval = current_sum
        
        for it in range(max_iterations):
            if time.time() - start_time > timeout:
                break
                
            # Generate a neighbor
            # Randomly pick an item to flip (add or remove)
            # Or swap one in and one out.
            # Let's try a random flip first.
            i = random.randint(0, n - 1)
            new_selected = current_selected[:]
            new_selected[i] = not new_selected[i]
            
            new_eval = current_eval
            if new_selected[i]:
                new_eval += weights[i]
            else:
                new_eval -= weights[i]
                
            # Constraint: sum must be < L
            if new_eval >= L:
                # Invalid state. We can either reject immediately or penalize.
                # For strict constraint, reject.
                # But sometimes we need to remove an item to make room.
                # Let's stick to valid states only.
                # If we add an item and it exceeds L, we must remove at least one other item.
                # This makes simple flip difficult.
                # Let's use a "swap" move: remove one selected, add one unselected.
                # Or "add" move: add one unselected (if valid).
                # Or "remove" move: remove one selected.
                
                # Let's try a different neighbor generation:
                # 1. Pick a random selected item to remove.
                # 2. Pick a random unselected item to add.
                # This maintains the "potential" to stay close to L.
                
                # Actually, let's just restart if we get stuck or use a simpler heuristic.
                # Simulated Annealing on constrained problems is tricky.
                
                # Let's try a different strategy: **Genetic Algorithm**? No, too complex.
                
                # Let's go back to **Meet in the Middle** but with a smaller split to ensure it fits in time.
                # Split into 3 parts? 
                # n=47. Split into 16, 16, 15.
                # 2^16 = 65536. 
                # Enumerate all subsets of part 1 (65k), part 2 (65k), part 3 (32k).
                # This is small!
                # Then combine?
                # Combining 3 lists of 65k, 65k, 32k is hard.
                
                # Let's stick to 2 parts: 23 and 24.
                # 2^23 = 8,388,608.
                # 2^24 = 16,777,216.
                # Total 25 million states.
                # In Python, iterating 25M times takes ~5-10 seconds if optimized.
                # Storing 25M integers takes ~200MB. Might be okay.
                # But we need to prune.
                
                # Let's implement MITM with 23/24 split.
                pass

        return best_set, best_sum

    # Let's implement the MITM properly.
    
    def solve_mitm():
        # Split indices into two halves
        mid = n // 2 # 23
        left_indices = list(range(mid))
        right_indices = list(range(mid, n))
        
        left_weights = weights[:mid]
        right_weights = weights[mid:]
        
        def generate_subsets(weights_list):
            # Returns list of (sum, mask)
            # We only care about sums < L
            m = len(weights_list)
            subsets = []
            # Use iterative bit manipulation or recursion?
            # Iterative is faster.
            # 2^m iterations
            for mask in range(1 << m):
                s = 0
                # Calculate sum for this mask
                # Optimization: use DP or just sum.
                # Summing 23 items * 8M times = 200M operations. Too slow in Python.
                
                # Better: Generate recursively with pruning?
                # Or use Gray code?
                
                # Let's use a recursive generator that yields (sum, mask)
                # and prunes if sum >= L.
                pass
            
            return subsets
        
        # Actually, 2^23 is 8.4M. 
        # If we do a simple loop and sum, it's slow.
        # We can precompute sums for half of the bits?
        
        # Let's try a different split: 20 and 27? No.
        # How about 22 and 25?
        # 2^22 = 4M. 2^25 = 33M.
        
        # Let's try to optimize the subset sum generation.
        # We can use a list of sums and update it.
        # sums[0] = 0
        # for each weight w:
        #   new_sums = [s + w for s in sums if s + w < L]
        #   sums += new_sums
        # This is O(2^m * m) in worst case but often pruned.
        # For unit fractions, many subsets will exceed L?
        # Sum of 1/2 + ... + 1/48 is approx ln(48) + gamma ~ 4.5.
        # So most subsets will be < L (which corresponds to sum of weights < L).
        # Wait, L is huge. The condition is sum(weights) < L.
        # weights[i] = L // (i+2).
        # Sum of ALL weights is > L.
        # But random subsets?
        # Expected sum of a random subset is (1/2) * Sum(weights).
        # Sum(weights) = L * Sum(1/d). Sum(1/d) ~ 4.5.
        # So expected sum is 2.25 * L.
        # So about half of all subsets will have sum > L?
        # No, the distribution is not normal.
        # But pruning `if s + w < L` will cut the tree significantly.
        
        def gen_subsets_pruned(w_list):
            # Returns list of (sum, mask) for valid subsets
            # m = len(w_list)
            # We represent mask as integer
            results = [(0, 0)]
            for i, w in enumerate(w_list):
                new_results = []
                bit = 1 << i
                for s, m in results:
                    ns = s + w
                    if ns < L:
                        new_results.append((ns, m | bit))
                results.extend(new_results)
            return results

        # This recursive extension might be slow if the list grows too large.
        # Let's check the size.
        # If we don't prune, it's 2^m.
        # With pruning, it depends.
        
        # Let's try this MITM approach.
        # If it's too slow, we fall back to B&B.
        
        try:
            # Split into two halves
            # To balance the time, maybe 23 and 24 is okay if pruning is effective.
            # Let's try 20 and 27? No, 27 is too big.
            # Let's try 22 and 25.
            # 2^22 = 4.2M. 2^25 = 33M.
            
            # Let's stick to 23/24.
            mid = 23
            left_w = weights[:mid]
            right_w = weights[mid:]
            
            # Generate left subsets
            left_subsets = gen_subsets_pruned(left_w)
            # Sort by sum
            left_subsets.sort(key=lambda x: x[0])
            
            # Generate right subsets
            right_subsets = gen_subsets_pruned(right_w)
            # Sort by sum
            right_subsets.sort(key=lambda x: x[0])
            
            # Now find best pair (l, r) such that l.sum + r.sum < L
            # Maximize l.sum + r.sum
            # Two-pointer technique
            # i points to largest left sum
            # j points to largest right sum
            
            i = len(left_subsets) - 1
            j = 0
            best_sum = 0
            best_mask = 0
            
            # We want max sum < L.
            # left is sorted ascending. right is sorted ascending.
            # Start with i = last (largest), j = 0 (smallest).
            # If left[i] + right[j] < L, update best, try to increase right (j+1).
            # If left[i] + right[j] >= L, decrease left (i-1).
            
            # Wait, this finds the best for a FIXED i?
            # No, standard two-pointer for "max sum < K" with two sorted arrays:
            # i = len(A)-1, j = 0
            # while i >= 0 and j < len(B):
            #   s = A[i] + B[j]
            #   if s < K:
            #     update best
            #     j += 1
            #   else:
            #     i -= 1
            
            # This works because if A[i] + B[j] >= K, then A[i] + B[j'] for j'>j is also >= K.
            # So we must decrease i.
            # If A[i] + B[j] < K, we found a valid pair. Can we do better?
            # Yes, by increasing j (increasing B part).
            
            i = len(left_subsets) - 1
            j = 0
            best_sum = 0
            best_l_mask = 0
            best_r_mask = 0
            
            while i >= 0 and j < len(right_subsets):
                s_l = left_subsets[i][0]
                s_r = right_subsets[j][0]
                total = s_l + s_r
                if total < L:
                    if total > best_sum:
                        best_sum = total
                        best_l_mask = left_subsets[i][1]
                        best_r_mask = right_subsets[j][1]
                    j += 1
                else:
                    i -= 1
                    
            if best_sum > 0:
                # Convert masks to indices
                indices = []
                # Left mask corresponds to indices 0..mid-1
                for k in range(mid):
                    if best_l_mask & (1 << k):
                        indices.append(k)
                # Right mask corresponds to indices mid..n-1
                for k in range(len(right_w)):
                    if best_r_mask & (1 << k):
                        indices.append(mid + k)
                        
                return indices
            
        except Exception as e:
            pass
            
        return None

    result = solve_mitm()
    if result:
        return result
        
    # Fallback to B&B if MITM fails or times out
    # (Re-using the BBB logic from the prompt)
    
    def lcm_func(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm_func, range(2, 49))
    weights = [L // d for d in denominators]
    n = len(weights)
    
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
        
    best_sum_weight = 0
    best_indices = []
    
    def update_best(sum_w, indices):
        nonlocal best_sum_weight, best_indices
        if sum_w > best_sum_weight:
            best_sum_weight = sum_w
            best_indices = list(indices)
            
    def dfs(idx, current_sum, current_indices):
        nonlocal best_sum_weight, best_indices
        
        if time.time() - start_time > timeout:
            return
            
        if current_sum + suffix_sum[idx] <= best_sum_weight:
            return
            
        if idx == n:
            if current_sum > best_sum_weight and current_sum < L:
                update_best(current_sum, current_indices)
            return
            
        if current_sum + weights[idx] < L:
            dfs(idx + 1, current_sum + weights[idx], current_indices + [idx])
            
        if time.time() - start_time > timeout:
            return
            
        dfs(idx + 1, current_sum, current_indices)
        
    dfs(0, 0, [])
    
    if not best_indices:
        return [0]
        
    return best_indices
