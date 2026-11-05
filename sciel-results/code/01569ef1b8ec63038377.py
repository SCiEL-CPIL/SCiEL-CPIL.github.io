import time
import sys
from fractions import Fraction
import bisect

def construct():
    start = time.time()
    deadline = start + 85  # 85 second budget
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
            
    def lcm(a, b):
        return a // gcd(a, b) * b
        
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are integers
    weights = [L // d for d in denominators]
    
    # Split into two sets A and B
    mid = n // 2
    A_weights = weights[:mid]
    B_weights = weights[mid:]
    
    # Generate subset sums for A
    sums_A = [(0, 0)]
    for i, w in enumerate(A_weights):
        current_len = len(sums_A)
        for k in range(current_len):
            s, m = sums_A[k]
            sums_A.append((s + w, m | (1 << i)))
            
    # Sort A by sum
    sums_A.sort(key=lambda x: x[0])
    
    # Generate subset sums for B
    sums_B = [(0, 0)]
    for i, w in enumerate(B_weights):
        current_len = len(sums_B)
        for k in range(current_len):
            s, m = sums_B[k]
            sums_B.append((s + w, m | (1 << i)))
            
    # Sort B by sum
    sums_B.sort(key=lambda x: x[0])
    
    # Two-pointer technique to find max A_sum + B_sum < L
    # As A_sum decreases (i goes down), max allowed B increases (j goes up)
    
    best_sum = -1
    best_a_idx = 0
    best_b_idx = 0
    
    j = 0
    len_B = len(sums_B)
    
    # Iterate A from largest to smallest
    for i in range(len(sums_A) - 1, -1, -1):
        a_sum, a_mask = sums_A[i]
        
        # Advance j as far as possible while a_sum + sums_B[j][0] < L
        while j + 1 < len_B and a_sum + sums_B[j + 1][0] < L:
            j += 1
        
        if j >= 0:
            b_sum, b_mask = sums_B[j]
            total = a_sum + b_sum
            if total > best_sum:
                best_sum = total
                best_a_idx = i
                best_b_idx = j
                
        # Time check
        if time.time() > deadline - 15:
            break

    if best_sum < 0:
        # Fallback to a simple valid set
        # 1/2 + 1/3 + 1/7 = 41/42 < 1
        return [0, 1, 5]

    # Reconstruct indices
    a_mask = sums_A[best_a_idx][1]
    b_mask = sums_B[best_b_idx][1]
    
    selected = []
    for i in range(mid):
        if a_mask & (1 << i):
            selected.append(i)
    for i in range(n - mid):
        if b_mask & (1 << i):
            selected.append(mid + i)
            
    # Verify
    total_frac = sum(Fraction(1, denominators[k]) for k in selected)
    if not (total_frac < 1):
        return [0, 1, 5]
    
    # Local search polish: try to improve by swapping elements
    # Try adding elements that keep sum < 1, or removing and adding different elements
    best_deficit = 1 - total_frac
    improved = True
    max_iterations = 1000
    
    while improved and time.time() < deadline - 5:
        improved = False
        current_sum = total_frac
        current_indices = set(selected)
        
        # Try adding each unselected element
        for idx in range(n):
            if idx not in current_indices:
                new_sum = current_sum + Fraction(1, denominators[idx])
                if new_sum < 1:
                    new_deficit = 1 - new_sum
                    if new_deficit < best_deficit:
                        best_deficit = new_deficit
                        selected.append(idx)
                        current_indices.add(idx)
                        current_sum = new_sum
                        improved = True
        
        # Try swapping: remove one element, add another
        if time.time() > deadline - 8:
            break
        
        for remove_idx in list(current_indices):
            sum_without_remove = current_sum - Fraction(1, denominators[remove_idx])
            for add_idx in range(n):
                if add_idx not in current_indices:
                    new_sum = sum_without_remove + Fraction(1, denominators[add_idx])
                    if new_sum < 1:
                        new_deficit = 1 - new_sum
                        if new_deficit < best_deficit:
                            best_deficit = new_deficit
                            selected.remove(remove_idx)
                            current_indices.discard(remove_idx)
                            selected.append(add_idx)
                            current_indices.add(add_idx)
                            current_sum = new_sum
                            improved = True
                            break
            if time.time() > deadline - 8:
                break
        
        # Try removing one element if it improves (unlikely but check)
        if time.time() > deadline - 8:
            break
        
        for remove_idx in list(current_indices):
            new_sum = current_sum - Fraction(1, denominators[remove_idx])
            new_deficit = 1 - new_sum
            if new_deficit < best_deficit:
                best_deficit = new_deficit
                selected.remove(remove_idx)
                current_indices.discard(remove_idx)
                current_sum = new_sum
                improved = True
    
    # Final verification
    total_frac = sum(Fraction(1, denominators[k]) for k in selected)
    if total_frac < 1 and len(selected) > 0:
        return selected
    else:
        return [0, 1, 5]
