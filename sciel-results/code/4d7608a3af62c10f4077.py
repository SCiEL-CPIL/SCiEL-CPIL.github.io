import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for 120s limit
    
    N = 1000
    max_sum = 2 * N
    
    def try_add(A, in_A, sum_counts, x):
        if x in in_A:
            return False
        if sum_counts[2 * x] >= 2:
            return False
        for y in A:
            if sum_counts[x + y] >= 2:
                return False
        in_A.add(x)
        A.append(x)
        sum_counts[2 * x] += 1
        for y in A[:-1]:
            sum_counts[x + y] += 1
        return True

    def remove_element(A, in_A, sum_counts, x):
        if x not in in_A:
            return False
        in_A.remove(x)
        A.remove(x)
        sum_counts[2 * x] -= 1
        for y in A:
            sum_counts[x + y] -= 1
        return True

    def count_blocking(A, in_A, sum_counts, x):
        """Count how many sums would be violated if x were added."""
        if x in in_A:
            return 0
        block = 0
        if sum_counts[2 * x] >= 2:
            block += 1
        for y in A:
            if sum_counts[x + y] >= 2:
                block += 1
        return block

    def greedy_build(seed):
        """Build a B2[2] set using greedy with scoring."""
        rng = random.Random(seed)
        A = []
        in_A = set()
        sum_counts = [0] * (2 * N + 1)
        
        # Pre-compute initial scores: for each candidate, count how many
        # sums are already "tight" (count >= 1) that it would touch
        for x in range(1, N + 1):
            if try_add(A, in_A, sum_counts, x):
                continue
        
        return A, in_A, sum_counts

    def local_search(A, in_A, sum_counts, deadline, rng):
        """Local search with swaps and 1-for-2 moves."""
        best_set = A[:]
        best_size = len(A)
        
        # Pre-compute candidate list
        candidates = [x for x in range(1, N + 1) if x not in in_A]
        
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            
            # Try to add any element with minimal blocking
            if candidates:
                # Find candidate with minimum blocking
                min_block = float('inf')
                best_x = None
                # Sample a subset for speed
                sample_size = min(len(candidates), 200)
                sample = rng.sample(candidates, sample_size)
                for x in sample:
                    b = count_blocking(A, in_A, sum_counts, x)
                    if b < min_block:
                        min_block = b
                        best_x = x
                    if b == 0:
                        break
                
                if best_x is not None and min_block == 0:
                    if try_add(A, in_A, sum_counts, best_x):
                        candidates.remove(best_x)
                        if len(A) > best_size:
                            best_size = len(A)
                            best_set = A[:]
                        improved = True
                        no_improve_count = 0
                        continue
            
            # Try swaps (1-for-1)
            if A and candidates:
                swapped = False
                for _ in range(100):
                    y = rng.choice(A)
                    x = rng.choice(candidates)
                    
                    # Check if swap is valid
                    if x in in_A:
                        continue
                    
                    # Simulate: remove y, add x
                    # Check 2x
                    s = 2 * x
                    contrib_y = 0
                    if 2 * y == s:
                        contrib_y = 1
                    else:
                        z = s - y
                        if z in in_A and z != y:
                            contrib_y = 1
                    if sum_counts[s] - contrib_y + 1 > 2:
                        continue
                    
                    # Check x+z for z in A, z != y
                    valid = True
                    for z in A:
                        if z == y:
                            continue
                        s = x + z
                        contrib_y = 0
                        if 2 * y == s:
                            contrib_y = 1
                        else:
                            w = s - y
                            if w in in_A and w != y:
                                contrib_y = 1
                        if sum_counts[s] - contrib_y + 1 > 2:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        remove_element(A, in_A, sum_counts, y)
                        in_A.add(x)
                        A.append(x)
                        sum_counts[2 * x] += 1
                        for z in A[:-1]:
                            sum_counts[x + z] += 1
                        candidates.remove(x)
                        candidates.append(y)
                        swapped = True
                        if len(A) > best_size:
                            best_size = len(A)
                            best_set = A[:]
                        improved = True
                        break
                
                if swapped:
                    no_improve_count = 0
                    continue
            
            # Try 1-for-2 moves
            if A and len(candidates) >= 2:
                did_12 = False
                for _ in range(50):
                    y = rng.choice(A)
                    idx1 = rng.randint(0, len(candidates) - 1)
                    idx2 = rng.randint(0, len(candidates) - 1)
                    while idx2 == idx1:
                        idx2 = rng.randint(0, len(candidates) - 1)
                    
                    x1 = candidates[idx1]
                    x2 = candidates[idx2]
                    
                    if x1 in in_A or x2 in in_A or x1 == x2:
                        continue
                    
                    # Check validity of 1-for-2
                    valid = True
                    
                    # Helper: contribution of y to sum s
                    def y_contrib(s):
                        if 2 * y == s:
                            return 1
                        z = s - y
                        if z in in_A and z != y:
                            return 1
                        return 0
                    
                    # Helper: contribution of x to sum s in context of A\{y}
                    def x_contrib(s, x):
                        if 2 * x == s:
                            return 1
                        z = s - x
                        if z in in_A and z != y and z != x:
                            return 1
                        return 0
                    
                    # Check 2*x1
                    s = 2 * x1
                    c = sum_counts[s] - y_contrib(s) + x_contrib(s, x1) + x_contrib(s, x2)
                    if c > 2:
                        valid = False
                    
                    if valid:
                        # Check 2*x2
                        s = 2 * x2
                        c = sum_counts[s] - y_contrib(s) + x_contrib(s, x1) + x_contrib(s, x2)
                        if c > 2:
                            valid = False
                    
                    if valid:
                        # Check x1+x2
                        s = x1 + x2
                        c = sum_counts[s] - y_contrib(s) + x_contrib(s, x1) + x_contrib(s, x2)
                        if c > 2:
                            valid = False
                    
                    if valid:
                        # Check x1+z and x2+z for z in A\{y}
                        for z in A:
                            if z == y:
                                continue
                            
                            s = x1 + z
                            cy = y_contrib(s)
                            cx1 = 1  # pair (x1, z)
                            cx2 = 0
                            if 2 * x2 == s:
                                cx2 = 1
                            else:
                                w = s - x2
                                if w in in_A and w != y and w != x2:
                                    cx2 = 1
                            if sum_counts[s] - cy + cx1 + cx2 > 2:
                                valid = False
                                break
                        
                        if valid:
                            s = x2 + z
                            cy = y_contrib(s)
                            cx2 = 1  # pair (x2, z)
                            cx1 = 0
                            if 2 * x1 == s:
                                cx1 = 1
                            else:
                                w = s - x1
                                if w in in_A and w != y and w != x1:
                                    cx1 = 1
                            if sum_counts[s] - cy + cx1 + cx2 > 2:
                                valid = False
                                break
                    
                    if valid:
                        # Perform 1-for-2
                        remove_element(A, in_A, sum_counts, y)
                        in_A.add(x1)
                        A.append(x1)
                        sum_counts[2 * x1] += 1
                        for z in A[:-1]:
                            sum_counts[x1 + z] += 1
                        in_A.add(x2)
                        A.append(x2)
                        sum_counts[2 * x2] += 1
                        for z in A[:-1]:
                            sum_counts[x2 + z] += 1
                        candidates.remove(x1)
                        candidates.remove(x2)
                        candidates.append(y)
                        did_12 = True
                        if len(A) > best_size:
                            best_size = len(A)
                            best_set = A[:]
                        improved = True
                        break
                
                if did_12:
                    no_improve_count = 0
                    continue
            
            if not improved:
                no_improve_count += 1
                # Remove a random element to escape
                if A and rng.random() < 0.5:
                    y = rng.choice(A)
                    remove_element(A, in_A, sum_counts, y)
                    candidates.append(y)
        
        return best_set

    def polish(A, in_A, sum_counts, deadline):
        """Final polish: try all possible single additions."""
        best_set = A[:]
        best_size = len(A)
        
        candidates = [x for x in range(1, N + 1) if x not in in_A]
        
        while time.time() < deadline:
            improved = False
            for x in candidates[:]:
                if try_add(A, in_A, sum_counts, x):
                    candidates.remove(x)
                    if len(A) > best_size:
                        best_size = len(A)
                        best_set = A[:]
                    improved = True
            if not improved:
                break
        
        return best_set

    # Main: multiple restarts
    global_best = []
    global_best_size = 0
    
    # First, a simple greedy to get a baseline
    A0, in_A0, sc0 = greedy_build(42)
    global_best = A0[:]
    global_best_size = len(A0)
    
    # Local search on the greedy result
    rng = random.Random(12345)
    best1 = local_search(A0[:], in_A0.copy(), sc0[:], deadline - 10.0, rng)
    if len(best1) > global_best_size:
        global_best = best1
        global_best_size = len(best1)
    
    # Polish
    if time.time() < deadline - 5.0:
        # Rebuild sum_counts for the best set
        in_A_best = set(global_best)
        sc_best = [0] * (2 * N + 1)
        for x in global_best:
            sc_best[2 * x] += 1
            for y in global_best:
                if y > x:
                    sc_best[x + y] += 1
        polished = polish(global_best[:], in_A_best.copy(), sc_best[:], deadline - 2.0)
        if len(polished) > global_best_size:
            global_best = polished
            global_best_size = len(polished)
    
    return global_best
