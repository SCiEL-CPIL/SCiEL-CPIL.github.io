"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Find 12 distinct positive integers with all 4096 subset sums distinct,
    # minimizing the largest element.
    
    def check_valid(arr):
        """Check if all subset sums are distinct."""
        sums = {0}
        for x in arr:
            new_sums = {s + x for s in sums}
            if new_sums & sums:
                return False
            sums = sums | new_sums
        return len(sums) == 4096
    
    def find_greedy(deadline):
        """Greedy: pick smallest valid next element at each step."""
        arr = [1]
        sums = {0}
        
        for step in range(11):
            if time.time() > deadline:
                return None
            
            found = False
            current_sum = sum(arr)
            upper = min(2048, current_sum + 1)
            
            for x in range(arr[-1] + 1, upper + 1):
                if time.time() > deadline:
                    return None
                
                # Check collision efficiently
                collides = False
                for s in sums:
                    if (s + x) in sums:
                        collides = True
                        break
                
                if not collides:
                    new_sums = {s + x for s in sums}
                    sums = sums | new_sums
                    arr.append(x)
                    found = True
                    break
            
            if not found:
                x = current_sum + 1
                if x > 2048:
                    return None
                new_sums = {s + x for s in sums}
                sums = sums | new_sums
                arr.append(x)
        
        return arr
    
    def local_search(arr, deadline):
        """Try to reduce the largest element via local search."""
        arr = list(arr)
        best_max = max(arr)
        
        while time.time() < deadline:
            improved = False
            
            # Try reducing the largest element
            idx = arr.index(max(arr))
            old_val = arr[idx]
            
            # Try smaller values for this position
            for new_val in range(arr[idx-1] + 1, old_val):
                if time.time() > deadline:
                    break
                
                candidate = arr[:idx] + [new_val] + arr[idx+1:]
                if len(set(candidate)) != 12:
                    continue
                
                if check_valid(candidate):
                    arr = candidate
                    best_max = max(arr)
                    improved = True
                    break
            
            if not improved:
                # Try swapping elements or other perturbations
                # Random perturbation: pick a random position and try nearby values
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    
                    i = random.randint(0, 11)
                    old_val = arr[i]
                    min_val = arr[i-1] + 1 if i > 0 else 1
                    max_val = arr[i+1] - 1 if i < 11 else 2048
                    
                    if max_val <= min_val:
                        continue
                    
                    new_val = random.randint(min_val, max_val)
                    if new_val == old_val:
                        continue
                    
                    candidate = arr[:]
                    candidate[i] = new_val
                    
                    if len(set(candidate)) != 12:
                        continue
                    
                    if check_valid(candidate):
                        arr = candidate
                        if max(arr) < best_max:
                            best_max = max(arr)
                        improved = True
                        break
        
        return arr
    
    # Start with greedy
    result = find_greedy(deadline - 10)
    
    if result is not None:
        if len(result) == 12 and len(set(result)) == 12 and all(x > 0 for x in result) and max(result) <= 1048576:
            if check_valid(result):
                # Try to improve with local search
                if time.time() < deadline - 5:
                    improved = local_search(result, deadline - 5)
                    if improved and check_valid(improved) and max(improved) < max(result):
                        result = improved
                return result
    
    # Fallback: powers of 2
    return [1 << i for i in range(12)]
# EVOLVE-BLOCK-END
