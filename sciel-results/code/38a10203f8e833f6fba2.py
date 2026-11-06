import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    N = 256
    
    # Precompute all 3-APs and 4-APs
    # 3-AP: (a, a+d, a+2d)
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
            
    # 4-AP: (a, a+d, a+2d, a+3d)
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # Map element to indices of 3-APs it belongs to
    elem_to_3aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(idx)
            
    # Map element to indices of 4-APs it belongs to
    elem_to_4aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(idx)

    def count_3aps(A):
        # A is a boolean array of size N+1
        count = 0
        for ap in three_aps:
            if A[ap[0]] and A[ap[1]] and A[ap[2]]:
                count += 1
        return count

    def has_4ap(A):
        # Returns a 4-AP if found, else None
        for ap in four_aps:
            if A[ap[0]] and A[ap[1]] and A[ap[2]] and A[ap[3]]:
                return ap
        return None

    def improve_add(A):
        # Try to add elements that don't create 4-APs and increase 3-APs
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_x = -1
            
            for x in range(1, N + 1):
                if A[x]:
                    continue
                
                # Check if adding x creates a 4-AP
                creates_4ap = False
                for idx in elem_to_4aps[x]:
                    ap = four_aps[idx]
                    # Check if other 3 elements are in A
                    if all(A[y] for y in ap if y != x):
                        creates_4ap = True
                        break
                
                if creates_4ap:
                    continue
                
                # Calculate gain in 3-APs
                gain = 0
                for idx in elem_to_3aps[x]:
                    ap = three_aps[idx]
                    # Check if other 2 elements are in A
                    if all(A[y] for y in ap if y != x):
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_gain > 0 and best_x != -1:
                A[best_x] = True
                improved = True
                
    best_A = None
    best_score = -1
    
    # Multiple restarts with different randomization in removal strategy
    while time.time() < deadline - 5.0:
        A = np.ones(N + 1, dtype=bool)
        A[0] = False
        
        # Greedy removal of 4-APs
        # We maintain a set of active 4-AP indices? 
        # Simpler: just loop through four_aps and check.
        # To speed up, we can keep a list of "violating" 4-APs.
        violating_4aps = set(range(len(four_aps)))
        
        # Initial check
        active_4aps = []
        for idx in violating_4aps:
             ap = four_aps[idx]
             if A[ap[0]] and A[ap[1]] and A[ap[2]] and A[ap[3]]:
                 active_4aps.append(idx)
                 
        while active_4aps:
            # Pick a random violating 4-AP
            idx = random.choice(active_4aps)
            ap = four_aps[idx]
            
            # For each element in ap, count current 3-APs involving it
            min_count = float('inf')
            candidates = []
            for x in ap:
                # Count 3-APs containing x that are currently fully in A
                count = 0
                for ap_idx in elem_to_3aps[x]:
                    ap3 = three_aps[ap_idx]
                    if A[ap3[0]] and A[ap3[1]] and A[ap3[2]]:
                        count += 1
                if count < min_count:
                    min_count = count
                    candidates = [x]
                elif count == min_count:
                    candidates.append(x)
            
            # Remove a random candidate
            x_to_remove = random.choice(candidates)
            A[x_to_remove] = False
            
            # Update active_4aps: remove any 4-AP that contains x_to_remove
            # Also, some 4-APs might have been "fixed" if x_to_remove was the only missing piece? 
            # No, we are removing, so we only fix 4-APs that contained x_to_remove.
            # We need to update the list of active 4-APs.
            # It's faster to just rebuild the list of active 4-APs occasionally or maintain it carefully.
            # Given N=256, len(four_aps) is small (~5000). Rebuilding is fast.
            active_4aps = []
            for j in range(len(four_aps)):
                ap_j = four_aps[j]
                if A[ap_j[0]] and A[ap_j[1]] and A[ap_j[2]] and A[ap_j[3]]:
                    active_4aps.append(j)
                    
        # Now A is 4-AP free
        improve_add(A)
        
        score = count_3aps(A)
        if score > best_score:
            best_score = score
            best_A = A.copy()
            
        # Perturb: Remove some random elements from best_A to escape local optimum
        if best_A is not None:
            A = best_A.copy()
            num_remove = random.randint(3, 8)
            indices = [i for i in range(1, N + 1) if A[i]]
            if len(indices) > num_remove:
                to_remove = random.sample(indices, num_remove)
                for x in to_remove:
                    A[x] = False
            
            # Re-run greedy removal (though A might already be 4-AP free, removals don't create 4-APs)
            # So A is still 4-AP free.
            # Just run improve_add
            improve_add(A)
            
            score = count_3aps(A)
            if score > best_score:
                best_score = score
                best_A = A.copy()

    if best_A is None:
        return [1]
        
    return [i for i in range(1, N + 1) if best_A[i]]
