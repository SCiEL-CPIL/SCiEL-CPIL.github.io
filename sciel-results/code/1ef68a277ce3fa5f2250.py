import random
import time
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Precompute indices for 4-AP checks
    # For each x, and each role (0: x is 1st, 1: x is 2nd, 2: x is 3rd, 3: x is 4th),
    # store the list of d values and the corresponding indices in the array.
    # Actually, it's easier to just check the conditions directly with numpy.
    
    # Let's optimize can_add using numpy
    # in_S is a numpy boolean array of size N+1
    
    def can_add(x, in_S):
        # x is 1-indexed. in_S[0] is unused.
        
        # Case 1: x, x+d, x+2d, x+3d
        # d >= 1, x+3d <= N
        max_d1 = (N - x) // 3
        if max_d1 >= 1:
            ds = np.arange(1, max_d1 + 1)
            idx1 = x + ds
            idx2 = x + 2 * ds
            idx3 = x + 3 * ds
            # Check if any d makes all three True
            if np.any(in_S[idx1] & in_S[idx2] & in_S[idx3]):
                return False
        
        # Case 2: x-d, x, x+d, x+2d
        # d >= 1, x-d >= 1, x+2d <= N
        max_d2 = min(x - 1, (N - x) // 2)
        if max_d2 >= 1:
            ds = np.arange(1, max_d2 + 1)
            idx1 = x - ds
            idx2 = x + ds
            idx3 = x + 2 * ds
            if np.any(in_S[idx1] & in_S[idx2] & in_S[idx3]):
                return False
        
        # Case 3: x-2d, x-d, x, x+d
        # d >= 1, x-2d >= 1, x+d <= N
        max_d3 = min((x - 1) // 2, N - x)
        if max_d3 >= 1:
            ds = np.arange(1, max_d3 + 1)
            idx1 = x - 2 * ds
            idx2 = x - ds
            idx3 = x + ds
            if np.any(in_S[idx1] & in_S[idx2] & in_S[idx3]):
                return False
        
        # Case 4: x-3d, x-2d, x-d, x
        # d >= 1, x-3d >= 1
        max_d4 = (x - 1) // 3
        if max_d4 >= 1:
            ds = np.arange(1, max_d4 + 1)
            idx1 = x - 3 * ds
            idx2 = x - 2 * ds
            idx3 = x - ds
            if np.any(in_S[idx1] & in_S[idx2] & in_S[idx3]):
                return False
        
        return True

    best_size = 0
    best_set = None
    
    # Randomized Greedy Construction
    max_iterations = 2000 # Can likely increase this due to speedup
    for i in range(max_iterations):
        if time.time() > deadline:
            break
            
        order = np.random.permutation(N) + 1 # 1 to N
        
        in_S = np.zeros(N + 1, dtype=bool)
        count = 0
        
        for x in order:
            x = int(x)
            if can_add(x, in_S):
                in_S[x] = True
                count += 1
        
        if count > best_size:
            best_size = count
            best_set = in_S.copy()
            
    # Local Search / Polishing
    # With numpy, we can afford a simple swap search if time permits
    # But first, let's see how much time we save.
    # The previous loop was very slow. This one should be much faster.
    # We can try to add more iterations.
    
    # Let's add a second phase of greedy with different orders if time permits
    # Or just increase max_iterations significantly.
    
    # Let's try to do a local search: try to swap elements
    # For each x not in S, try to remove some y from S to make x addable.
    # This is O(N^2) checks. With numpy, each check is fast.
    # N=512, N^2 = 262144.
    # If each check takes ~10us, that's ~2.6s. Feasible.
    
    if best_set is not None:
        # Try to improve by swapping
        # We'll iterate a few times
        for _ in range(2): # 2 passes
            if time.time() > deadline - 5:
                break
            improved = False
            # Create a list of elements not in S
            not_in_S = [i for i in range(1, N+1) if not best_set[i]]
            in_S_list = [i for i in range(1, N+1) if best_set[i]]
            
            # Randomize order for diversity
            random.shuffle(not_in_S)
            random.shuffle(in_S_list)
            
            for x in not_in_S:
                if time.time() > deadline - 5:
                    break
                # Check if x can be added if we remove some y
                # We need to find if there is a y in S such that S - {y} + {x} is valid.
                # This means x is not in a 4-AP with S - {y}.
                # Equivalently, all 4-APs involving x in S must contain y.
                # Let's just try removing each y and checking.
                # To speed up, we can check if x is "blocked" by specific y's.
                
                # Simple approach:
                # For each y in in_S_list, temporarily remove y, check if x can be added.
                # If yes, do the swap.
                # This is O(|S|) checks per x.
                # Total O(|S| * |not_in_S|) checks.
                # ~256 * 256 = 65536 checks.
                # Each check is can_add(x, in_S_without_y).
                # We can optimize can_add to accept a "removed" element?
                # Or just modify the array.
                
                # Let's try a few swaps
                for y in in_S_list:
                    # Check if removing y allows adding x
                    # We need to check if x is in a 4-AP with S \ {y}.
                    # This is slightly complex. 
                    # Alternative: just do the swap and verify validity? 
                    # Verification is O(N * D). 
                    # Better: check conflicts.
                    
                    # Let's just skip complex swaps for now and rely on greedy + more iterations.
                    # The speedup from numpy should allow many more greedy iterations.
                    pass
                
                # If we found a valid swap, we would update best_set here.
                # For now, let's just increase greedy iterations.
            
            # Since we didn't implement swap logic fully in the loop above, 
            # let's just do more greedy iterations if time permits.
            # The loop above already ran max_iterations.
            # Let's run another batch.
            for i in range(2000):
                if time.time() > deadline:
                    break
                order = np.random.permutation(N) + 1
                in_S = np.zeros(N + 1, dtype=bool)
                count = 0
                for x in order:
                    x = int(x)
                    if can_add(x, in_S):
                        in_S[x] = True
                        count += 1
                if count > best_size:
                    best_size = count
                    best_set = in_S.copy()
                    improved = True
            if not improved:
                break

    if best_set is not None:
        result = [i for i in range(1, N+1) if best_set[i]]
    else:
        # Fallback
        result = []
        in_S = np.zeros(N + 1, dtype=bool)
        for x in range(1, N+1):
            if can_add(x, in_S):
                in_S[x] = True
                result.append(x)
        
    return result
