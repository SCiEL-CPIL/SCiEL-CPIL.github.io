import time
import itertools
import random
from collections import deque

def construct(n):
    start_time = time.time()
    timeout = 110.0  # Safety margin before 150s kill
    
    # For n=1, return "1"
    if n == 1:
        return "1"
    
    # For small n, we can use known optimal values or fast construction
    # n=2: "121" (len 3)
    # n=3: "123121321" (len 9) - actually optimal is 9
    # n=4: optimal is 33
    # n=5: optimal is 153
    # n=6: recursive gives 873, optimal known? LB 872. Recursive is 873.
    # n=7: recursive gives 5913. Known best is 5905. LB 5884.
    
    def get_recursive_superpermutation(k):
        if k == 1:
            return [1]
        
        prev = get_recursive_superpermutation(k - 1)
        
        # Extract unique permutations of length k-1 from prev in order
        perms = []
        seen = set()
        for i in range(len(prev) - (k - 1) + 1):
            window = tuple(prev[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        new_word = []
        for p in perms:
            block = list(p) + [k] + list(p)
            # Overlap
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
            
        return new_word

    # Get the recursive solution
    recursive_word = get_recursive_superpermutation(n)
    best_word = recursive_word
    best_len = len(best_word)
    
    # For n <= 5, recursive is optimal, return it
    if n <= 5:
        return "".join(map(str, best_word))
    
    # For n=6,7, try to improve via local search
    if n <= 7 and time.time() - start_time < timeout - 5:
        # Extract permutation order from recursive word
        perm_order = []
        seen = set()
        for i in range(len(best_word) - n + 1):
            window = tuple(best_word[i:i+n])
            if len(set(window)) == n and window not in seen:
                seen.add(window)
                perm_order.append(window)
        
        all_perms = set(itertools.permutations(range(1, n+1)))
        if set(perm_order) != all_perms:
            return "".join(map(str, best_word))
        
        num_perms = len(perm_order)
        
        # Precompute overlap matrix? 
        # 5040x5040 is 25M entries. 
        # Storing as a list of bytearrays or similar might be fast enough.
        # But let's try on-the-fly with caching first.
        
        overlap_cache = {}
        
        def calc_overlap(a, b):
            key = (a, b)
            if key in overlap_cache:
                return overlap_cache[key]
            # max k < n such that a[n-k:] == b[:k]
            max_ov = 0
            # Check from largest k down
            # a is tuple, b is tuple
            for k in range(n-1, 0, -1):
                if a[n-k:] == b[:k]:
                    max_ov = k
                    break
            overlap_cache[key] = max_ov
            return max_ov
        
        def total_length(order):
            total = n
            for i in range(len(order)-1):
                total += n - calc_overlap(order[i], order[i+1])
            return total
        
        current_order = list(perm_order)
        current_len = total_length(current_order)
        
        # Hill Climbing
        # We will try to swap adjacent elements or do 2-opt moves.
        # Given the time limit, simple adjacent swaps with restarts might be best.
        
        improved = True
        no_improve_count = 0
        max_no_improve = 100 # Restart if stuck
        
        while improved and time.time() - start_time < timeout - 10:
            improved = False
            no_improve_count = 0
            
            # Randomize starting point if stuck? 
            # Or just continue.
            
            # Try all adjacent swaps
            for i in range(num_perms - 1):
                if time.time() - start_time > timeout - 15:
                    break
                
                # Calculate cost delta for swapping i and i+1
                # Old edges: (i-1, i), (i, i+1), (i+1, i+2)
                # New edges: (i-1, i+1), (i+1, i), (i, i+2)
                
                old_cost = 0
                new_cost = 0
                
                if i > 0:
                    old_cost += n - calc_overlap(current_order[i-1], current_order[i])
                    new_cost += n - calc_overlap(current_order[i-1], current_order[i+1])
                
                old_cost += n - calc_overlap(current_order[i], current_order[i+1])
                new_cost += n - calc_overlap(current_order[i+1], current_order[i])
                
                if i+2 < num_perms:
                    old_cost += n - calc_overlap(current_order[i+1], current_order[i+2])
                    new_cost += n - calc_overlap(current_order[i], current_order[i+2])
                
                if new_cost < old_cost:
                    # Perform swap
                    current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                    current_len += new_cost - old_cost
                    improved = True
                    # After an improvement, we might want to restart the scan
                    # to catch new opportunities, or just continue.
                    # Restarting is safer for finding global min in local search.
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Try a random restart or a larger move (2-opt)
                    # For simplicity, let's try a random 2-opt move
                    if time.time() - start_time > timeout - 15:
                        break
                    
                    # Pick two random indices i, j (i < j)
                    # Reverse the segment [i, j]
                    # This is a larger move.
                    # Cost calculation is more complex.
                    # Let's just do a few random adjacent swaps to escape local minima?
                    # Or just accept the local minimum.
                    
                    # Let's try one random 2-opt
                    i = random.randint(0, num_perms - 2)
                    j = random.randint(i + 1, num_perms - 1)
                    
                    # Calculate cost change for reversing i..j
                    # This is expensive to calculate exactly without iterating.
                    # Let's just do a simple random shuffle of a small segment?
                    # Actually, let's just stop if we are stuck for a while.
                    # The recursive solution is a good baseline.
                    pass
        
        # Reconstruct word from best order
        if current_len < best_len:
            new_word = list(current_order[0])
            for i in range(1, num_perms):
                a = current_order[i-1]
                b = current_order[i]
                ov = calc_overlap(a, b)
                new_word.extend(b[ov:])
            
            best_word = new_word
            best_len = len(best_word)
    
    # Check time
    if time.time() - start_time > timeout:
        return "".join(map(str, best_word))
    
    return "".join(map(str, best_word))
