"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    # 1. Greedy Initialization
    # Prioritize larger 'a' because 1/(512-a) is larger.
    chosen = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    
    # Helper functions
    def calc_score(s):
        return sum(1.0 / (512.0 - a) for a in s)
    
    def is_coprime_to_set(num, s, exclude=None):
        for b in s:
            if b == exclude:
                continue
            if gcd(num, b) != 1:
                return False
        return True

    current_score = calc_score(chosen)
    
    # 2. Local Search (1-swap optimization)
    # Try to replace an element in 'chosen' with an element not in 'chosen'
    # if it increases the total score and maintains pairwise coprimality.
    
    # Precompute candidates not in chosen
    all_nums = set(range(1, 512))
    chosen_set = set(chosen)
    candidates = [a for a in range(511, 0, -1) if a not in chosen_set]
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > timeout:
            break
            
        for i, a_remove in enumerate(chosen):
            if time.time() - start_time > timeout:
                break
                
            # Try to add any candidate that is coprime to all other chosen elements
            # and has a higher weight than a_remove
            weight_remove = 1.0 / (512.0 - a_remove)
            
            for a_add in candidates:
                if a_add == a_remove:
                    continue
                weight_add = 1.0 / (512.0 - a_add)
                
                # Only consider if adding this increases the score potential
                # We are replacing a_remove with a_add.
                # Net change: weight_add - weight_remove
                if weight_add <= weight_remove:
                    continue
                
                # Check if a_add is coprime to all chosen except a_remove
                if is_coprime_to_set(a_add, chosen, exclude=a_remove):
                    # Perform swap
                    chosen[i] = a_add
                    chosen_set.remove(a_remove)
                    chosen_set.add(a_add)
                    # Update candidates list? 
                    # It's easier to just regenerate or mark, but since we are in a loop,
                    # let's just update the set and the list by filtering or using a set for candidates too.
                    # For efficiency with small N, we can just rebuild the candidate list if needed, 
                    # or simply note that 'candidates' is a list of non-chosen.
                    # Let's update the candidates list to reflect the change.
                    # Actually, 'candidates' is static in the first pass? No, it changes.
                    # Let's just use a set for chosen and a list for candidates, 
                    # but updating the list is O(N). N=512, so it's fine.
                    # Or simpler: just check against chosen_set.
                    
                    current_score += weight_add - weight_remove
                    improved = True
                    # Break out of inner loop to restart scan with new state?
                    # Or continue? Restarting is safer to find global local optimum.
                    # But breaking inner loop and continuing outer might miss things.
                    # Let's break inner, and then re-evaluate.
                    # To be efficient, we can just break inner and let the while loop restart.
                    # However, we need to update 'candidates' list.
                    # Let's just break and let the while loop handle it.
                    # But we need 'candidates' to be current.
                    # Let's restructure: use a set for chosen.
                    break
            
            # If we made a swap, we should probably restart the outer loop 
            # or at least update the candidate list.
            # Given the small size, restarting the while loop is cleanest.
            if improved:
                # Rebuild candidates list
                candidates = [a for a in range(511, 0, -1) if a not in chosen_set]
                break # Break outer loop to restart while
    
    return chosen
# EVOLVE-BLOCK-END
