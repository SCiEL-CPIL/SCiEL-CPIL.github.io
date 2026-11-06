import numpy as np
import time

def count_3aps_numpy(A):
    """
    Count 3-APs in boolean array A (length n).
    A[i] is True if (i+1) is in the set.
    """
    n = len(A)
    total = 0
    # d is the difference
    # Max d such that 2*d < n
    max_d = n // 2
    for d in range(1, max_d + 1):
        # a, a+d, a+2d
        # a ranges from 0 to n - 1 - 2*d
        max_a = n - 1 - 2 * d
        if max_a < 0:
            continue
        # Slices:
        # a: 0 to max_a
        # a+d: d to max_a+d
        # a+2d: 2*d to max_a+2*d
        mask = A[:max_a+1] & A[d:d+max_a+1] & A[2*d:2*d+max_a+1]
        total += np.sum(mask)
    return total

def creates_4ap_numpy(A, x):
    """
    Check if adding x to A creates a 4-AP.
    x is index 0-based.
    """
    n = len(A)
    # x is 1st term: x, x+d, x+2d, x+3d
    for d in range(1, n):
        if x + 3 * d >= n:
            break
        if A[x+d] and A[x+2*d] and A[x+3*d]:
            return True
    # x is 2nd term: x-d, x, x+d, x+2d
    for d in range(1, n):
        if x - d < 0:
            break
        if x + 2 * d >= n:
            break
        if A[x-d] and A[x+d] and A[x+2*d]:
            return True
    # x is 3rd term: x-2d, x-d, x, x+d
    for d in range(1, n):
        if x - 2 * d < 0:
            break
        if x + d >= n:
            break
        if A[x-2*d] and A[x-d] and A[x+d]:
            return True
    # x is 4th term: x-3d, x-2d, x-d, x
    for d in range(1, n):
        if x - 3 * d < 0:
            break
        if A[x-3*d] and A[x-2*d] and A[x-d]:
            return True
    return False

def count_3aps_for_element(A, x):
    """
    Count 3-APs in A that include element x.
    """
    n = len(A)
    count = 0
    # x is 1st: x, x+d, x+2d
    for d in range(1, n):
        if x + 2 * d >= n: break
        if A[x+d] and A[x+2*d]:
            count += 1
    # x is 2nd: x-d, x, x+d
    for d in range(1, n):
        if x - d < 0: break
        if x + d >= n: break
        if A[x-d] and A[x+d]:
            count += 1
    # x is 3rd: x-2d, x-d, x
    for d in range(1, n):
        if x - 2 * d < 0: break
        if A[x-2*d] and A[x-d]:
            count += 1
    return count

def construct():
    n = 256
    # Initialize with Cantor set (numbers with no 1 in base 3)
    # This set is 4-AP free.
    A = np.zeros(n, dtype=bool)
    for i in range(1, n + 1):
        # Convert i to base 3
        num = i
        has_one = False
        while num > 0:
            if num % 3 == 1:
                has_one = True
                break
            num //= 3
        if not has_one:
            A[i - 1] = True
            
    # If Cantor set is too small or empty, add some randoms?
    # Cantor set in [1, 256] should be decent.
    # Let's check size.
    # 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35...
    # It's roughly 2/3 of the numbers? No, 2/3 of digits are 0 or 2.
    # Density is (2/3)^k. For 256 (approx 3^5=243), density is (2/3)^5 ~ 0.13.
    # So ~33 elements.
    # This is a valid start.
    
    current_score = count_3aps_numpy(A)
    best_score = current_score
    best_A = A.copy()
    
    start_time = time.time()
    time_limit = 55  # Leave 5s buffer
    
    while time.time() - start_time < time_limit:
        added = False
        best_gain = 0
        best_elem = -1
        
        # Try to add elements
        # To speed up, we can check candidates in random order or all.
        # n=256, so checking all is fine.
        for x in range(n):
            if A[x]:
                continue
            if creates_4ap_numpy(A, x):
                continue
            
            # Calculate gain
            # We can approximate gain by counting 3-APs involving x in the new set.
            # But we need exact new score vs old score.
            # new_score = old_score + (3-APs involving x in new set)
            # The 3-APs involving x in new set are those where x is part of the AP.
            # Since x was not in A, any 3-AP involving x must have x in it.
            # So gain = count_3aps_for_element(A, x) ?
            # Wait. If we add x, new 3-APs are those that include x.
            # Do these 3-APs exist in A? No, because x wasn't in A.
            # So gain is exactly the number of 3-APs in (A U {x}) that contain x.
            # This is equivalent to count_3aps_for_element(A, x) if we consider A as the context.
            # Let's verify:
            # 3-AP (a, b, c). If x is one of them, and the other two are in A.
            # Yes.
            gain = count_3aps_for_element(A, x)
            
            if gain > best_gain:
                best_gain = gain
                best_elem = x
                
        if best_gain > 0:
            A[best_elem] = True
            current_score += best_gain
            if current_score > best_score:
                best_score = current_score
                best_A = A.copy()
            added = True
            
        if not added:
            # Local optimum. Try to remove an element to escape.
            # Remove element with lowest contribution to 3-APs.
            min_contrib = float('inf')
            min_elem = -1
            for x in range(n):
                if A[x]:
                    contrib = count_3aps_for_element(A, x)
                    if contrib < min_contrib:
                        min_contrib = contrib
                        min_elem = x
            
            if min_elem != -1:
                A[min_elem] = False
                # Recalculate score
                current_score = count_3aps_numpy(A)
                
    # Convert best_A to list of 1-based indices
    result = [i + 1 for i in range(n) if best_A[i]]
    return result
