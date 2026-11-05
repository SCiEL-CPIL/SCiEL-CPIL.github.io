"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    N = 512
    
    # Precompute weights: w(a) = 1 / (512 - a)
    # Higher a means higher weight
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
    
    # Greedy Initialization
    # Prioritize larger 'a' because 1/(512-a) is larger.
    # We want a set of pairwise coprime integers.
    chosen = []
    chosen_set = set()
    for a in range(N - 1, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            chosen_set.add(a)
    
    def calc_score(s):
        return sum(weights[a] for a in s)
    
    def is_coprime_to_set(num, s, exclude=None):
        for b in s:
            if b == exclude:
                continue
            if gcd(num, b) != 1:
                return False
        return True

    current_score = calc_score(chosen)
    best_score = current_score
    best_chosen = list(chosen)
    
    # 2. Local Search (1-swap optimization)
    # Try to replace an element in 'chosen' with an element not in 'chosen'
    # if it increases the total score and maintains pairwise coprimality.
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        for i, a_remove in enumerate(chosen):
            if time.time() > deadline:
                break
                
            # Try to add any candidate that is coprime to all other chosen elements
            # and has a higher weight than a_remove
            weight_remove = weights[a_remove]
            
            # We want to find the best a_add (highest weight, i.e., largest a)
            # that is coprime to all others.
            # Iterate from 511 down to 1
            best_add = -1
            for a_add in range(N - 1, 0, -1):
                if a_add in chosen_set:
                    continue
                # Check if a_add is coprime to all chosen except a_remove
                if is_coprime_to_set(a_add, chosen, exclude=a_remove):
                    # Since we iterate from largest to smallest, the first valid one is the best
                    best_add = a_add
                    break
            
            if best_add != -1:
                weight_add = weights[best_add]
                # Only swap if it improves score
                if weight_add > weight_remove + 1e-12:
                    # Perform swap
                    chosen[i] = best_add
                    chosen_set.remove(a_remove)
                    chosen_set.add(best_add)
                    current_score += weight_add - weight_remove
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_chosen = list(chosen)
                    # Restart the while loop to re-evaluate with new state
                    # Breaking the for loop and letting the while loop restart is cleaner
                    break
            
    # 3. 2-Swap Optimization
    # Try to replace two elements with two others to escape local optima
    # This is O(N^4) in worst case, but N is small (512) and set size is small (~100-200)
    # We limit the search to avoid timeout.
    
    if time.time() < deadline:
        improved2 = True
        while improved2:
            improved2 = False
            if time.time() > deadline:
                break
            
            n = len(chosen)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    
                    a_rem1 = chosen[i]
                    a_rem2 = chosen[j]
                    others = [chosen[k] for k in range(n) if k != i and k != j]
                    
                    # Find best pair (a_add1, a_add2)
                    # To be efficient, we iterate a_add1 from largest, then a_add2 from largest
                    # We need a_add1 != a_add2, both not in chosen_set, and coprime to others and each other.
                    
                    best_pair = None
                    best_gain = 0.0
                    w_rem1 = weights[a_rem1]
                    w_rem2 = weights[a_rem2]
                    
                    # Iterate candidates for add1
                    for a_add1 in range(N - 1, 0, -1):
                        if a_add1 in chosen_set or a_add1 == a_rem1 or a_add1 == a_rem2:
                            continue
                        if not is_coprime_to_set(a_add1, others):
                            continue
                        
                        w_add1 = weights[a_add1]
                        # If even the best possible second element doesn't help, skip?
                        # Max possible gain is w_add1 + max_w - w_rem1 - w_rem2
                        # But we need to find the actual best second element.
                        
                        # Iterate candidates for add2
                        for a_add2 in range(N - 1, 0, -1):
                            if a_add2 == a_add1:
                                continue
                            if a_add2 in chosen_set or a_add2 == a_rem1 or a_add2 == a_rem2:
                                continue
                            if not is_coprime_to_set(a_add2, others + [a_add1]):
                                continue
                            
                            w_add2 = weights[a_add2]
                            gain = (w_add1 + w_add2) - (w_rem1 + w_rem2)
                            
                            if gain > best_gain:
                                best_gain = gain
                                best_pair = (a_add1, a_add2)
                                
                            # Optimization: if we found a pair with both elements larger than removed ones, 
                            # and we are iterating downwards, we might break early?
                            # Not strictly, because a smaller add2 might allow a much larger add1?
                            # No, we iterate add1 outer, add2 inner.
                            # If we found a valid pair, is it the best?
                            # We want to maximize w_add1 + w_add2.
                            # Since w is increasing with a, larger a is better.
                            # So the first valid pair found (largest a_add1, then largest a_add2) is likely the best?
                            # Not necessarily. A slightly smaller a_add1 might allow a much larger a_add2?
                            # No, a_add2 is independent of a_add1's value except for coprimality.
                            # However, if a_add1 is large, it might conflict with many large a_add2 candidates.
                            # So we should keep searching to be safe, or at least a bit.
                            # Given time constraints, let's just take the first valid pair found?
                            # No, that's greedy. Let's keep the best found.
                            # But this inner loop is expensive.
                            # Let's break if gain is very high?
                            # For now, let's just find the first valid pair?
                            # Actually, iterating 511 down to 1 for both is 511*511 ~ 260k checks per (i,j) pair.
                            # n is ~100. n^2/2 ~ 5000 pairs. 5000 * 260k = 1.3 billion. Too slow.
                            # We need to break early.
                            # If we find a pair where a_add1 > a_rem1 and a_add2 > a_rem2, it's likely good.
                            # Let's just take the first valid pair found (largest a_add1, largest a_add2).
                            # This is a heuristic.
                            break
                        
                        if best_pair is not None:
                            # If we found a pair, do we break?
                            # If we break, we assume the first valid a_add1 is the best.
                            # This is a valid heuristic given time.
                            break
                    
                    if best_pair is not None and best_gain > 1e-12:
                        a_add1, a_add2 = best_pair
                        chosen[i] = a_add1
                        chosen[j] = a_add2
                        chosen_set.remove(a_rem1)
                        chosen_set.remove(a_rem2)
                        chosen_set.add(a_add1)
                        chosen_set.add(a_add2)
                        current_score += best_gain
                        improved2 = True
                        if current_score > best_score:
                            best_score = current_score
                            best_chosen = list(chosen)
                        break # Restart while loop
    
    return best_chosen
# EVOLVE-BLOCK-END
