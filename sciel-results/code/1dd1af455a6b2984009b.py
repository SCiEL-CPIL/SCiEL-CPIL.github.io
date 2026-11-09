import time
import random
import math
from collections import namedtuple

# State tuple for easier passing
State = namedtuple('State', 'elements, score')

# Maximum time allowed for the function (evaluator kills at 90s, we want safety)
TIME_LIMIT = 45.0

def is_3ap_free(elements):
    """
    Check if the set of elements contains no non-constant 3-term arithmetic progression.
    """
    if len(elements) < 3:
        return True
    s = set(elements)
    elements_list = sorted(elements)
    n = len(elements_list)
    for i in range(n):
        a = elements_list[i]
        for j in range(i+1, n):
            b = elements_list[j]
            # Case 1: c is such that a, b, c is AP. c = 2b - a
            c = 2 * b - a
            if c > b and c in s:
                return False
            # Case 2: c is such that c, a, b is AP. c = 2a - b
            c = 2 * a - b
            if c < a and c in s:
                return False
    return True

def base_p_no_zero_digits(x, p):
    """
    Return True if x has no zero digits when written in base p.
    """
    while x > 0:
        if x % p == 0:
            return False
        x //= p
    return True

def construct():
    start_time = time.time()
    
    # Reciprocals for numbers 1 to 256
    recip = [0.0] * 257
    for i in range(1, 257):
        recip[i] = 1.0 / i
        
    all_numbers = list(range(1, 257))
    
    # Strategy: Use base-11 digit restriction to get a guaranteed 3-AP-free set
    # Numbers with no zero digits in base 11 are 3-AP free (Salem-Spencer theorem variant)
    # This gives a high-quality starting point
    
    # First, try the base-11 approach
    current_elements = []
    current_set = set()
    current_score = 0.0
    
    for x in all_numbers:
        if base_p_no_zero_digits(x, 11):
            current_elements.append(x)
            current_set.add(x)
            current_score += recip[x]
    
    # Verify it's actually 3-AP free (should be, but let's check)
    if not is_3ap_free(current_set):
        # Fallback to greedy if something went wrong
        current_elements = []
        current_set = set()
        current_score = 0.0
        for x in all_numbers:
            current_set.add(x)
            if is_3ap_free(current_set):
                current_elements.append(x)
                current_score += recip[x]
            else:
                current_set.remove(x)
    
    # Track best solution found
    best_elements = list(current_elements)
    best_score = current_score
    
    # Lightweight local search: try swapping small-index elements with large-index elements
    # This can improve reciprocal sum without risking AP violations
    # We'll do a few rounds of targeted swaps
    
    # Sort current elements by value
    current_elements_sorted = sorted(current_elements)
    
    # Try to replace small elements with larger ones that maintain 3-AP-freeness
    # Focus on the smallest elements first since they have the highest reciprocal weight
    # Wait - we want to MAXIMIZE reciprocal sum, so we want SMALL elements (1/x is larger for small x)
    # So we should try to KEEP small elements and ADD more small elements if possible
    
    # Actually, let's do a more standard local search:
    # 1. Try adding elements not in the set
    # 2. Try swapping elements to improve the score
    
    available_to_add = [x for x in all_numbers if x not in current_set]
    
    # Simulated Annealing for refinement
    current_temp = 0.02
    cooling_rate = 0.9998
    min_temp = 1e-7
    
    while time.time() - start_time < TIME_LIMIT:
        if current_temp < min_temp:
            break
            
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try adding a random number
            if available_to_add:
                idx = random.randint(0, len(available_to_add) - 1)
                x_add = available_to_add[idx]
                
                trial_set = current_set | {x_add}
                
                if is_3ap_free(trial_set):
                    delta_score = recip[x_add]
                    trial_score = current_score + delta_score
                    
                    # Always accept improvement
                    # Accept worse with prob exp(delta/T)
                    if delta_score > 0 or (delta_score < 0 and random.random() < math.exp(delta_score / current_temp)):
                        current_elements.append(x_add)
                        current_set.add(x_add)
                        current_score = trial_score
                        available_to_add.pop(idx)
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_elements = list(current_elements)
                        
        elif move_type < 0.7:
            # Try removing a random number
            if current_elements:
                idx = random.randint(0, len(current_elements) - 1)
                x_rem = current_elements[idx]
                
                delta_score = -recip[x_rem]
                trial_score = current_score + delta_score
                
                if delta_score > 0 or (delta_score < 0 and random.random() < math.exp(delta_score / current_temp)):
                    current_elements.pop(idx)
                    current_set.remove(x_rem)
                    current_score = trial_score
                    available_to_add.append(x_rem)
                    
        else:
            # Try a swap: remove one element, add another
            if current_elements and available_to_add:
                idx_rem = random.randint(0, len(current_elements) - 1)
                idx_add = random.randint(0, len(available_to_add) - 1)
                
                x_rem = current_elements[idx_rem]
                x_add = available_to_add[idx_add]
                
                # Only consider swaps that might improve score
                # We want to remove large x (low reciprocal) and add small x (high reciprocal)
                if x_add < x_rem:
                    trial_set = (current_set - {x_rem}) | {x_add}
                    
                    if is_3ap_free(trial_set):
                        delta_score = recip[x_add] - recip[x_rem]
                        trial_score = current_score + delta_score
                        
                        if delta_score > 0 or (delta_score < 0 and random.random() < math.exp(delta_score / current_temp)):
                            current_elements[idx_rem] = x_add
                            current_set.remove(x_rem)
                            current_set.add(x_add)
                            current_score = trial_score
                            available_to_add[idx_add] = x_rem
                            
                            if current_score > best_score:
                                best_score = current_score
                                best_elements = list(current_elements)
                
                # Also try the reverse: remove small, add large (might enable other additions later)
                elif x_add > x_rem and random.random() < 0.1:
                    trial_set = (current_set - {x_rem}) | {x_add}
                    
                    if is_3ap_free(trial_set):
                        delta_score = recip[x_add] - recip[x_rem]
                        trial_score = current_score + delta_score
                        
                        if delta_score > 0 or (delta_score < 0 and random.random() < math.exp(delta_score / current_temp)):
                            current_elements[idx_rem] = x_add
                            current_set.remove(x_rem)
                            current_set.add(x_add)
                            current_score = trial_score
                            available_to_add[idx_add] = x_rem
                            
                            if current_score > best_score:
                                best_score = current_score
                                best_elements = list(current_elements)
        
        current_temp *= cooling_rate
        
        # Safety check: ensure we're not taking too long
        if time.time() - start_time > TIME_LIMIT - 2.0:
            break
    
    # Final verification
    if not is_3ap_free(best_elements):
        # Fallback to a known valid set
        return [x for x in all_numbers if base_p_no_zero_digits(x, 11)]
    
    return best_elements

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
