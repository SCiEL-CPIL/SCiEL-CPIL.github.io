import time
from math import lcm
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    weights = [L // d for d in denominators]
    
    # Meet-in-the-middle: 23 + 24
    mid = 23
    n1 = mid
    n2 = n - mid
    
    # Generate subset sums for first half
    sums1 = [0]
    masks1 = [0]
    for i in range(n1):
        w = weights[i]
        bit = 1 << i
        new_sums = []
        new_masks = []
        for idx in range(len(sums1)):
            ns = sums1[idx] + w
            if ns < L:
                new_sums.append(ns)
                new_masks.append(masks1[idx] | bit)
        sums1.extend(new_sums)
        masks1.extend(new_masks)
        if time.time() > deadline - 25:
            break
    
    # Generate subset sums for second half
    sums2 = [0]
    masks2 = [0]
    for i in range(n2):
        w = weights[mid + i]
        bit = 1 << i
        new_sums = []
        new_masks = []
        for idx in range(len(sums2)):
            ns = sums2[idx] + w
            if ns < L:
                new_sums.append(ns)
                new_masks.append(masks2[idx] | bit)
        sums2.extend(new_sums)
        masks2.extend(new_masks)
        if time.time() > deadline - 25:
            break
    
    # Sort second half by sum
    order2 = sorted(range(len(sums2)), key=lambda i: sums2[i])
    sorted_sums2 = [sums2[i] for i in order2]
    sorted_masks2 = [masks2[i] for i in order2]
    
    best_sum = -1
    best_m1 = 0
    best_m2 = 0
    
    for idx1 in range(len(sums1)):
        if time.time() > deadline - 5:
            break
        s1 = sums1[idx1]
        m1 = masks1[idx1]
        target = L - s1 - 1
        pos = bisect.bisect_right(sorted_sums2, target)
        if pos > 0:
            s2 = sorted_sums2[pos - 1]
            m2 = sorted_masks2[pos - 1]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_m1 = m1
                best_m2 = m2
    
    # Reconstruct
    result_indices = []
    for i in range(n1):
        if best_m1 & (1 << i):
            result_indices.append(i)
    for i in range(n2):
        if best_m2 & (1 << i):
            result_indices.append(mid + i)
    
    if not result_indices:
        return [0]
    
    # Local search
    recip = [Fraction(1, d) for d in denominators]
    current_subset = set(result_indices)
    current_sum = sum(recip[i] for i in current_subset)
    best_deficit = Fraction(1, 1) - current_sum
    
    if best_deficit <= 0:
        return result_indices
    
    # 1-swap local search
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        in_sub = list(current_subset)
        not_in_sub = [i for i in range(n) if i not in current_subset]
        
        best_gain = Fraction(0)
        best_swap = None
        
        for i in in_sub:
            for j in not_in_sub:
                new_sum = current_sum - recip[i] + recip[j]
                if new_sum < 1:
                    gain = recip[j] - recip[i]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, j)
        
        if best_swap:
            i, j = best_swap
            current_subset.remove(i)
            current_subset.add(j)
            current_sum += best_gain
            improved = True
            new_deficit = Fraction(1, 1) - current_sum
            if new_deficit < best_deficit:
                best_deficit = new_deficit
                result_indices = list(current_subset)
    
    # Add single
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            not_in_sub = [i for i in range(n) if i not in current_subset]
            best_gain = Fraction(0)
            best_add = None
            for j in not_in_sub:
                new_sum = current_sum + recip[j]
                if new_sum < 1:
                    gain = recip[j]
                    if gain > best_gain:
                        best_gain = gain
                        best_add = j
            if best_add is not None:
                current_subset.add(best_add)
                current_sum += best_gain
                improved = True
                new_deficit = Fraction(1, 1) - current_sum
                if new_deficit < best_deficit:
                    best_deficit = new_deficit
                    result_indices = list(current_subset)
    
    # Remove single (to enable better additions)
    if time.time() < deadline - 3:
        # Only remove if we can immediately add back with net gain
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            in_sub = list(current_subset)
            not_in_sub = [i for i in range(n) if i not in current_subset]
            
            best_gain = Fraction(0)
            best_move = None
            
            for i in in_sub:
                for j in not_in_sub:
                    new_sum = current_sum - recip[i] + recip[j]
                    if new_sum < 1:
                        gain = recip[j] - recip[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (i, j)
            
            if best_move:
                i, j = best_move
                current_subset.remove(i)
                current_subset.add(j)
                current_sum += best_gain
                improved = True
                new_deficit = Fraction(1, 1) - current_sum
                if new_deficit < best_deficit:
                    best_deficit = new_deficit
                    result_indices = list(current_subset)
    
    # 2-swap: remove 1, add 2
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            in_sub = list(current_subset)
            not_in_sub = [i for i in range(n) if i not in current_subset]
            
            best_gain = Fraction(0)
            best_move = None
            
            for i in in_sub:
                for j1 in range(len(not_in_sub)):
                    for j2 in range(j1 + 1, len(not_in_sub)):
                        a, b = not_in_sub[j1], not_in_sub[j2]
                        new_sum = current_sum - recip[i] + recip[a] + recip[b]
                        if new_sum < 1:
                            gain = recip[a] + recip[b] - recip[i]
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (i, a, b)
            
            if best_move:
                i, a, b = best_move
                current_subset.remove(i)
                current_subset.add(a)
                current_subset.add(b)
                current_sum += best_gain
                improved = True
                new_deficit = Fraction(1, 1) - current_sum
                if new_deficit < best_deficit:
                    best_deficit = new_deficit
                    result_indices = list(current_subset)
    
    # 2-swap: remove 2, add 1
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            in_sub = list(current_subset)
            not_in_sub = [i for i in range(n) if i not in current_subset]
            
            best_gain = Fraction(0)
            best_move = None
            
            for i1 in range(len(in_sub)):
                for i2 in range(i1 + 1, len(in_sub)):
                    a, b = in_sub[i1], in_sub[i2]
                    for j in not_in_sub:
                        new_sum = current_sum - recip[a] - recip[b] + recip[j]
                        if new_sum < 1:
                            gain = recip[j] - recip[a] - recip[b]
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (a, b, j)
            
            if best_move:
                a, b, j = best_move
                current_subset.remove(a)
                current_subset.remove(b)
                current_subset.add(j)
                current_sum += best_gain
                improved = True
                new_deficit = Fraction(1, 1) - current_sum
                if new_deficit < best_deficit:
                    best_deficit = new_deficit
                    result_indices = list(current_subset)
    
    return result_indices
