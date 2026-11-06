"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from functools import lru_cache

def construct():
    deadline = time.time() + 85.0  # 5s safety margin before 90s kill
    target = factorial(12)  # 479001600
    
    # Get all divisors of target that are > 12
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Generate all divisors
    divisors = set()
    for a in range(11):  # 2^a, a=0..10
        for b in range(6):  # 3^b, b=0..5
            for c in range(3):  # 5^c, c=0..2
                for d in range(2):  # 7^d, d=0..1
                    for e in range(2):  # 11^e, e=0..1
                        val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d) * (11 ** e)
                        if val > 12:
                            divisors.add(val)
    
    # Sort divisors - we'll use them in various orders
    divisors_sorted = sorted(divisors)
    
    best = [target]  # fallback: single factor
    best_max = target
    
    # For search: we build a list of factors, each > 12, distinct, product = target
    # We track the remaining product and the set of used factors
    # Branch and bound: if max(current_factors) >= best_max, prune (since we want to minimize max)
    # Actually, we can still have a solution where max is equal but we want strictly better or equal
    
    def search(remaining, used, current_factors, max_so_far):
        nonlocal best, best_max
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            # Found a complete factorization
            if current_factors:
                cur_max = max_so_far
                if cur_max < best_max:
                    best_max = cur_max
                    best = list(current_factors)
            return
        
        if not current_factors and remaining > 1:
            # No factors chosen yet, need to pick at least one
            pass
        
        # If we've already used 12 factors, we can't add more
        if len(current_factors) >= 12:
            return
        
        # The remaining must be factorable into distinct factors > 12
        # Each new factor must be a divisor of remaining, > 12, not in used
        # Also, if we pick factor f, then remaining/f must be factorable into distinct factors > 12
        # (or remaining/f == 1 if this is the last factor)
        
        # Pruning: if max_so_far >= best_max and we already have factors, 
        # we can only improve if we find a solution with max < best_max
        # But even if max_so_far < best_max, adding more factors could increase max
        
        # Key pruning: the minimum possible value for any new factor is 13
        # If remaining < 13, we can't pick another factor (unless remaining == 1)
        # If remaining > 1 and remaining < 13, dead end
        
        # Try each divisor of remaining that is > 12 and not in used
        # To find good solutions early, try smaller factors first
        # But also, we want to keep max_so_far small
        
        # Get divisors of remaining that are > 12
        # remaining is a divisor of target, so its divisors are a subset of target's divisors
        
        # For efficiency, iterate through sorted divisors and check if they divide remaining
        # But that's expensive. Instead, let's generate divisors of remaining on the fly.
        
        # Actually, since remaining divides target, we can find its divisors by checking
        # which of our precomputed divisors divide remaining.
        
        # Better: generate divisors of remaining directly
        # remaining has the form 2^a * 3^b * 5^c * 7^d * 11^e where a<=10, b<=5, etc.
        # We can factor remaining and generate its divisors.
        
        # For now, let's try a simpler approach: iterate through sorted divisors
        # and check if d divides remaining and d > 12 and d not in used
        
        # To make this efficient, we can precompute for each possible remaining value
        # the list of valid next factors. But remaining can take many values.
        
        # Let's just iterate through divisors_sorted and filter
        for d in divisors_sorted:
            if time.time() > deadline:
                return
            
            if d > remaining:
                break  # since sorted, no more divisors will divide remaining
            
            if d in used:
                continue
            
            if remaining % d != 0:
                continue
            
            new_remaining = remaining // d
            new_max = max(max_so_far, d)
            
            # Pruning: if new_max >= best_max and we already have at least one factor,
            # we can't improve (since all future factors will be >= 13, but max is already >= best_max)
            # Actually, we CAN have a solution with the same max, but we only care about improving
            # So if new_max > best_max, we can prune
            # If new_max == best_max, we might still find a solution but it won't be better
            # Let's allow == to find alternative solutions, but prune >
            if new_max > best_max:
                continue
            
            # Additional pruning: if new_remaining > 1, it must be factorable into
            # distinct factors > 12, none of which are in used ∪ {d}
            # The minimum product of k distinct factors > 12 is at least 13*14*15*...
            # But this is complex. Let's add a simple check:
            # If new_remaining > 1, it must be at least 13 (since next factor > 12)
            # Actually, if new_remaining > 1, the next factor must be >= 13, so new_remaining >= 13
            # But new_remaining could be a single factor, so just need new_remaining > 12 or new_remaining == 1
            
            if new_remaining > 1:
                if new_remaining < 13:
                    continue  # can't form a valid factor
                # Also, if we've used many factors, we need enough room
                # The remaining must be expressible as product of distinct factors > 12
                # Minimum number of additional factors needed: 0 (if new_remaining itself is a valid factor)
                # Maximum: 12 - len(current_factors) - 1
                # For now, just check that new_remaining is a divisor of target (it is, by construction)
            
            # Recurse
            used.add(d)
            current_factors.append(d)
            search(new_remaining, used, current_factors, new_max)
            current_factors.pop()
            used.remove(d)
            
            # Check time after each branch
            if time.time() > deadline:
                return
        
        # Also consider: what if we stop here and remaining is itself a valid factor?
        # That's handled when we pick d = remaining in the loop above (if remaining > 12 and not in used)
        # But we also need to handle the case where remaining is a valid factor and we pick it
        # This is covered in the loop
    
    search(target, set(), [], 0)
    
    # Ensure we return a valid result
    if best and max(best) < best_max + 1:  # sanity check
        return best
    return [target]

# EVOLVE-BLOCK-END
