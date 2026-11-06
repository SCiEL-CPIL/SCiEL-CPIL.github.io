import time
from math import lcm
from fractions import Fraction
import bisect
import random

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
    
    # Local search with multiple restarts
    recip = [Fraction(1, d) for d in denominators]
    best_deficit = Fraction(1, 1) - sum(recip[i] for i in result_indices)
    best_subset = set(result_indices)
    
    def local_search(start_subset, time_limit):
        current_subset = set(start_subset)
        current_sum = sum(recip[i] for i in current_subset)
        best_def_local = Fraction(1, 1) - current_sum
        best_sub_local = set(current_subset)
        
        if best_def_local <= 0:
            return best_sub_local, best_def_local
        
        # 1-swap local search (best improvement)
        improved = True
        while improved and time.time() < time_limit:
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
                if new_deficit < best_def_local:
                    best_def_local = new_deficit
                    best_sub_local = set(current_subset)
        
        # Add single
        if time.time() < time_limit - 1:
            improved = True
            while improved and time.time() < time_limit - 1:
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
                    if new_deficit < best_def_local:
                        best_def_local = new_deficit
                        best_sub_local = set(current_subset)
        
        # 2-swap: remove 1, add 2
        if time.time() < time_limit - 1:
            improved = True
            while improved and time.time() < time_limit - 1:
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
                    if new_deficit < best_def_local:
                        best_def_local = new_deficit
                        best_sub_local = set(current_subset)
        
        # 2-swap: remove 2, add 1
        if time.time() < time_limit - 1:
            improved = True
            while improved and time.time() < time_limit - 1:
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
                    if new_deficit < best_def_local:
                        best_def_local = new_deficit
                        best_sub_local = set(current_subset)
        
        return best_sub_local, best_def_local
    
    # Run local search from MITM result
    if time.time() < deadline - 10:
        best_subset, best_deficit = local_search(result_indices, deadline - 5)
    
    # Multiple restarts with random perturbations
    rng = random.Random(42)
    num_restarts = 50
    restart_time_budget = deadline - 10 - time.time()
    if restart_time_budget > 5:
        time_per_restart = restart_time_budget / num_restarts
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Create a perturbed starting point
            start = list(best_subset)
            # Randomly swap 2-4 elements
            num_swaps = rng.randint(2, 4)
            for _ in range(num_swaps):
                if len(start) > 0:
                    idx_remove = rng.randint(0, len(start) - 1)
                    candidates = [i for i in range(n) if i not in start]
                    if candidates:
                        idx_add = rng.choice(candidates)
                        start[idx_remove] = idx_add
            
            start_set = set(start)
            # Ensure sum < 1
            s = sum(recip[i] for i in start_set)
            if s >= 1:
                # Remove random elements until sum < 1
                while s >= 1 and start_set:
                    elem = rng.choice(list(start_set))
                    start_set.remove(elem)
                    s -= recip[elem]
            
            if not start_set:
                continue
            
            sub, def_local = local_search(start_set, min(deadline - 3, time.time() + time_per_restart))
            
            if def_local < best_deficit:
                best_deficit = def_local
                best_subset = sub
    
    result_indices = list(best_subset)
    if not result_indices:
        result_indices = [0]
    
    return result_indices
