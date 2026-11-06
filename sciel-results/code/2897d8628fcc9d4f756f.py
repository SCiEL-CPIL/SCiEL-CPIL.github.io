import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline with 10s margin before 90s kill
    
    # Verify a candidate: check all 2^12 subset sums are distinct
    def verify(arr):
        sums = set()
        n = len(arr)
        for mask in range(1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += arr[i]
            if s in sums:
                return False
            sums.add(s)
        return True
    
    # Check if arr is superincreasing (guarantees distinct subset sums)
    def is_superincreasing(arr):
        running_sum = 0
        for a in arr:
            if a <= running_sum:
                return False
            running_sum += a
        return True
    
    # Generate a random superincreasing sequence with max element = max_elem
    def generate_superincreasing(max_elem, n=12):
        if max_elem < n:
            return None
        arr = []
        remaining = max_elem
        # We need n elements, each > sum of previous
        # Work backwards: last element = max_elem, previous must be < max_elem
        # But also sum of all before last < max_elem
        # So sum of first n-1 < max_elem
        # And each element in first n-1 must be superincreasing among themselves
        
        # Strategy: pick the last element as max_elem, then greedily pick
        # the remaining n-1 as a superincreasing sequence with sum < max_elem
        if max_elem < 2**(n-1):
            # Can't fit n-1 superincreasing elements with sum < max_elem
            # if max_elem < 2^(n-1) - 1... actually sum of first n-1 superincreasing
            # minimum is 1+2+4+...+2^(n-2) = 2^(n-1)-1
            # So we need max_elem >= 2^(n-1) for a standard superincreasing
            # But we can try non-standard: just check if feasible
            pass
        
        # Try to construct: elements a1 < a2 < ... < an = max_elem
        # a1 >= 1, a2 > a1, a3 > a1+a2, etc.
        # For superincreasing: a_k > sum(a_1..a_{k-1})
        
        # Greedy from the front:
        arr = []
        running = 0
        for i in range(n - 1):
            # Minimum next element is running + 1
            min_val = running + 1
            # We need to leave room for the last element = max_elem
            # And the sum of all n-1 elements must be < max_elem
            # Current sum after adding this element: running + val
            # We need running + val + (sum of remaining n-1-i elements) < max_elem
            # Minimum sum of remaining: each at least prev_sum + 1
            # This is complex; let's just try random values
            arr.append(1)
            running += 1
        # Now we have n-1 ones, which is not superincreasing after the first
        # Let me redo this properly
        
        # Better approach: build from front with randomness
        arr = []
        running = 0
        for i in range(n):
            min_val = running + 1
            if i == n - 1:
                # Last element must be exactly max_elem
                if max_elem < min_val:
                    return None
                arr.append(max_elem)
            else:
                # Pick a value >= min_val, but leave room for remaining elements
                # Minimum sum of remaining (n-1-i) elements if superincreasing starting from running+val:
                # elem1 >= running+val+1, elem2 >= running+val+1+(running+val+1)+1, etc.
                # This grows fast. Let's just pick min_val and see if we can complete
                val = min_val
                arr.append(val)
                running += val
        return arr
    
    # Local search: try to reduce the maximum element
    def local_search(arr, max_iter=10000):
        arr = sorted(arr[:])
        current_max = max(arr)
        best_arr = arr[:]
        best_max = current_max
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try to reduce the largest element
            # The largest element must be > sum of all others (for superincreasing)
            # So minimum possible largest = sum(others) + 1
            others_sum = sum(arr) - arr[-1]
            min_possible = others_sum + 1
            
            if arr[-1] > min_possible:
                # Try reducing the last element
                new_val = min_possible + random.randint(0, max(0, arr[-1] - min_possible - 1))
                if new_val > others_sum:
                    test_arr = arr[:-1] + [new_val]
                    if is_superincreasing(test_arr):
                        arr = test_arr
                        current_max = max(arr)
                        improved = True
            
            if not improved:
                # Try modifying non-last elements to allow smaller max
                # Pick a random index (not last) and try to increase it slightly
                # so we can decrease the last element
                idx = random.randint(0, len(arr) - 2)
                old_val = arr[idx]
                # Try increasing this element
                for delta in random.sample(range(1, 20), min(5, range(1, 20))):
                    new_val = old_val + delta
                    if new_val > sum(arr[:idx]):  # maintain superincreasing at this position
                        test_arr = arr[:idx] + [new_val] + arr[idx+1:]
                        # Now check if we can reduce the last element
                        others_sum2 = sum(test_arr) - test_arr[-1]
                        if test_arr[-1] > others_sum2 + 1:
                            new_last = others_sum2 + 1 + random.randint(0, max(0, test_arr[-1] - others_sum2 - 2))
                            if new_last > others_sum2:
                                test_arr[-1] = new_last
                                if is_superincreasing(test_arr):
                                    arr = test_arr
                                    current_max = max(arr)
                                    improved = True
                                    break
            
            if current_max < best_max:
                best_max = current_max
                best_arr = arr[:]
            
            # Occasionally try a fresh restart
            if iteration % 500 == 0 and iteration > 0:
                if time.time() > deadline:
                    break
                restart_max = best_max - 1
                if restart_max >= 2000:
                    trial = generate_superincreasing(restart_max)
                    if trial and is_superincreasing(trial) and max(trial) < best_max:
                        arr = trial
                        best_arr = trial[:]
                        best_max = max(trial)
        
        return best_arr, best_max
    
    # Start with powers of 2
    best_arr = [1 << i for i in range(12)]
    best_max = max(best_arr)  # 2048
    
    # Try various approaches to find better solutions
    # Approach 1: Local search from powers of 2
    trial_arr, trial_max = local_search(best_arr, max_iter=5000)
    if trial_max < best_max:
        best_arr = trial_arr
        best_max = trial_max
    
    # Approach 2: Try systematic construction with smaller max
    # For superincreasing: a_k > sum(a_1..a_{k-1})
    # Minimum max for n=12 superincreasing starting with 1: 1,2,4,8,16,32,64,128,256,512,1024,2048
    # That's exactly powers of 2. So superincreasing with smallest start gives 2048.
    
    # But non-superincreasing sets can also have distinct subset sums!
    # Let's try a different approach: greedy with verification
    
    # Approach 3: Greedy construction trying to keep elements small
    def greedy_construct():
        arr = []
        used_sums = {0}
        for i in range(12):
            # Try smallest possible value
            for val in range(1, 2049):
                if val in arr:
                    continue
                # Check if adding val creates any duplicate subset sum
                new_sums = set()
                valid = True
                for s in used_sums:
                    ns = s + val
                    if ns in used_sums or ns in new_sums:
                        valid = False
                        break
                    new_sums.add(ns)
                if valid:
                    arr.append(val)
                    used_sums = used_sums | new_sums
                    break
        return arr
    
    greedy_arr = greedy_construct()
    if verify(greedy_arr):
        greedy_max = max(greedy_arr)
        if greedy_max < best_max:
            best_arr = greedy_arr
            best_max = greedy_max
    
    # Approach 4: More aggressive local search with random restarts
    if time.time() < deadline - 5:
        for restart in range(20):
            if time.time() > deadline:
                break
            # Random initial solution
            arr = []
            used_sums = {0}
            valid = True
            for i in range(12):
                # Pick random values, biased toward small
                found = False
                for _ in range(100):
                    val = random.randint(1, best_max)
                    if val in arr:
                        continue
                    new_sums = set()
                    ok = True
                    for s in used_sums:
                        ns = s + val
                        if ns in used_sums or ns in new_sums:
                            ok = False
                            break
                        new_sums.add(ns)
                    if ok:
                        arr.append(val)
                        used_sums = used_sums | new_sums
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if not valid:
                continue
            if verify(arr):
                arr_max = max(arr)
                if arr_max < best_max:
                    best_arr = arr
                    best_max = arr_max
                # Local search from this starting point
                if time.time() < deadline - 3:
                    refined, refined_max = local_search(arr, max_iter=2000)
                    if refined_max < best_max:
                        best_arr = refined
                        best_max = refined_max
    
    # Final verification
    if not verify(best_arr):
        # Fallback to powers of 2
        best_arr = [1 << i for i in range(12)]
    
    return best_arr
