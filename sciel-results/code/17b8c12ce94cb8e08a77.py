import time
import numpy as np
import random

def _get_divisors(n):
    divisors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted([d for d in divisors if d > 1])

def _score_classes_fast(classes, base=51975):
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return base - int(covered.sum())

def _greedy_construct(base=51975, order='ascending'):
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    if order == 'descending':
        divisors = divisors[::-1]
    elif order == 'random':
        random.shuffle(divisors)
    
    for m in divisors:
        best_r = 0
        best_count = -1
        # Optimization: Instead of iterating all r and summing, 
        # we can just iterate. m is at most ~17000, so this is fine.
        for r in range(m):
            count = int((covered[r::m] == 0).sum())
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes

def _local_search(classes, base=51975, time_limit=10, max_iterations=50):
    start_time = time.time()
    n = len(classes)
    
    # Initialize counts
    counts = np.zeros(base, dtype=np.int16)
    for m, r in classes:
        counts[r::m] += 1
        
    current_score = int((counts == 0).sum())
    best_score = current_score
    best_classes = [c[:] for c in classes]
    
    # Sort classes by modulus? Not strictly necessary, but might help convergence
    # We will iterate through indices 0..n-1
    
    improved = True
    iteration = 0
    while improved and time.time() - start_time < time_limit and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            m_i, r_i = best_classes[i]
            
            # Find the best residue for this modulus
            best_r = r_i
            best_delta = 0 # We want to minimize score, so we look for negative delta (improvement)
            
            # We need to evaluate the delta for changing r_i to new_r
            # Delta = (loss of unique coverage from old_r) - (gain of new coverage from new_r)
            # Loss = count of x in old_positions where counts[x] == 1
            # Gain = count of x in new_positions where counts[x] == 0
            
            old_positions = np.arange(r_i, base, m_i)
            old_unique_loss = int((counts[old_positions] == 1).sum())
            
            for new_r in range(m_i):
                if new_r == r_i:
                    continue
                new_positions = np.arange(new_r, base, m_i)
                new_gain = int((counts[new_positions] == 0).sum())
                
                # If we switch, the score change is:
                # + old_unique_loss (these become uncovered)
                # - new_gain (these become covered)
                # Note: There is an overlap effect if old and new positions overlap?
                # If x is in both old and new positions:
                # Old: counts[x] >= 1. New: counts[x] >= 1.
                # The "loss" calculation assumed counts[x] drops by 1.
                # The "gain" calculation assumed counts[x] was 0.
                # If x is in intersection:
                # - It is NOT in the set where counts[x]==0 (so not in new_gain).
                # - It IS in the set where counts[x]==1? If counts[x]==1, it was only covered by this class.
                #   But if we switch to new_r, and x is in new_r::m, then x is still covered.
                #   So counts[x] does NOT drop to 0.
                #   My formula `old_unique_loss` counts x where counts[x]==1.
                #   If x is in intersection, and counts[x]==1, it is counted in loss.
                #   But since x is also in new_positions, it remains covered.
                #   So we overestimated the loss.
                #   Similarly, if x is in intersection, it is not counted in gain (since counts[x] != 0).
                #   
                # Correct Delta Calculation:
                # Let S_old = positions of old_r, S_new = positions of new_r.
                # Let U = set of x where counts[x] == 0.
                # Let V = set of x where counts[x] == 1.
                # 
                # If we remove old_r and add new_r:
                # 1. For x in S_old \ S_new:
                #    counts[x] decreases by 1.
                #    If x in V, it becomes 0 (Uncovered). Score increases by 1.
                # 2. For x in S_new \ S_old:
                #    counts[x] increases by 1.
                #    If x in U, it becomes 1 (Covered). Score decreases by 1.
                # 3. For x in S_old \cap S_new:
                #    counts[x] unchanged.
                #
                # So:
                # Loss = | (S_old \ S_new) \cap V |
                # Gain = | (S_new \ S_old) \cap U |
                # Delta = Loss - Gain
                
                # Efficient calculation:
                # Loss = |S_old \cap V| - |S_old \cap S_new \cap V|
                # Gain = |S_new \cap U| - |S_old \cap S_new \cap U|
                
                # Since U and V are disjoint, and S_old \cap S_new is a subset of S_old and S_new.
                # We can compute:
                # loss_raw = int((counts[old_positions] == 1).sum())
                # gain_raw = int((counts[new_positions] == 0).sum())
                
                # Intersection correction:
                # We need to subtract the overlap from both?
                # If x is in intersection:
                # - It contributes to loss_raw if counts[x]==1.
                # - It does NOT contribute to gain_raw (counts[x]!=0).
                # - But in the actual change, x remains covered. So the "loss" for this x should be 0.
                # - So we must subtract 1 from Loss for each x in intersection with counts[x]==1.
                
                # How to compute intersection counts efficiently?
                # S_old = r_i::m_i, S_new = new_r::m_i.
                # Intersection is non-empty only if r_i == new_r mod gcd(m_i, m_i)? No, same modulus.
                # If m_i is the same, S_old and S_new are disjoint unless r_i == new_r.
                # Wait. S_old = {r_i + k*m_i}. S_new = {new_r + k*m_i}.
                # If r_i != new_r, are they disjoint?
                # x = r_i + k1*m_i = new_r + k2*m_i
                # r_i - new_r = (k2-k1)*m_i
                # r_i - new_r is a multiple of m_i.
                # Since 0 <= r_i, new_r < m_i, the only multiple of m_i in (-m_i, m_i) is 0.
                # So r_i - new_r = 0 => r_i == new_r.
                # Thus, for distinct residues of the SAME modulus, the sets are DISJOINT.
                
                # Therefore, S_old \cap S_new is EMPTY for new_r != r_i.
                # So Loss = loss_raw and Gain = gain_raw.
                
                delta = old_unique_loss - new_gain
                
                if delta < best_delta:
                    best_delta = delta
                    best_r = new_r
            
            if best_r != r_i:
                # Apply change
                old_positions = np.arange(r_i, base, m_i)
                new_positions = np.arange(best_r, base, m_i)
                
                counts[old_positions] -= 1
                counts[new_positions] += 1
                
                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_classes[i][1] = best_r
                else:
                    # Revert? No, we are doing steepest descent, we take the best move.
                    # If best_delta was positive (worse), we wouldn't have updated best_r != r_i 
                    # unless we forced it. But we initialized best_delta=0 and best_r=r_i.
                    # So we only change if delta < 0 (improvement).
                    pass
                
                best_classes[i][1] = best_r
                improved = True
                r_i = best_r # Update for next iterations if needed, though we move to next i

    return best_classes

def _multi_start_local_search(base=51975, time_limit=60, num_starts=5):
    start_time = time.time()
    best_classes = None
    best_score = float('inf')
    
    orders = ['ascending', 'descending', 'random', 'random', 'random']
    for idx, order in enumerate(orders):
        if time.time() - start_time > time_limit:
            break
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        # Allocate time for local search
        ls_time = min(remaining_time * 0.8, 15)
        
        classes = _greedy_construct(base, order=order)
        classes = _local_search(classes, base, time_limit=ls_time)
        score = _score_classes_fast(classes, base)
        
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
    
    return best_classes

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    classes = _multi_start_local_search(base, time_limit=min(90, int(deadline - time.time()) - 10))
    
    if time.time() < deadline - 5 and classes is not None:
        remaining = int(deadline - time.time()) - 2
        if remaining > 3:
            classes = _local_search(classes, base, time_limit=min(remaining, 15))
    
    return classes
