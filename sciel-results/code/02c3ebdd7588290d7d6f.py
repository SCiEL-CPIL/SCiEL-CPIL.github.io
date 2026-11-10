"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    p = 1009
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    self_inv = [False] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        if x * x % p == 1:
            self_inv[x] = True
    
    # Self-inverse: x * x % p == 1 => x = 1 or x = p-1 = 1008
    # So 1 and 1008 can never be in a valid interval
    
    best_start = 2
    best_len = 1  # [2,2] is valid since 2*2=4 != 1
    
    # For each start s in [1, 1008], find max valid end
    for s in range(1, 1009):
        if time.time() > deadline:
            break
        
        # Pruning: if the max possible length from s is not better than best_len, skip
        if 1008 - s + 1 <= best_len:
            continue
            
        # If s itself is self-inverse, skip
        if self_inv[s]:
            continue
        
        # Extend end as far as possible
        e = s
        valid = True
        # We need to check: for all x in [s, e], inv[x] is not in [s, e]
        # and x*x % p != 1 (but we already excluded self-inverses for s)
        
        # Efficient approach: start with e=s, try to extend
        # The constraint is: for all x in [s,e], inv[x] must be outside [s,e]
        # i.e., inv[x] < s or inv[x] > e
        
        # Start with just [s, s]
        # Check if s is self-inverse (already checked)
        # Now try to extend e
        e = s
        while e < 1008:
            # Try adding e+1
            new_e = e + 1
            # Check if new_e is self-inverse
            if self_inv[new_e]:
                break
            # Check if inv[new_e] is in [s, new_e]
            if s <= inv[new_e] <= new_e:
                break
            # Also need to check: for all existing x in [s, e], inv[x] is not in [s, new_e]
            # Since inv[x] was not in [s, e] before, it could only be new_e
            # So check: is there any x in [s,e] such that inv[x] == new_e?
            # That means x == inv[new_e]. So check if inv[new_e] is in [s, e]
            # But we already checked inv[new_e] >= s and inv[new_e] <= new_e
            # If inv[new_e] == new_e, it's self-inverse (handled above)
            # If inv[new_e] < new_e, it's in [s, e] only if inv[new_e] >= s
            # We already check inv[new_e] >= s and inv[new_e] <= new_e
            # So if inv[new_e] is in [s, new_e-1] = [s, e], that's a conflict
            # And if inv[new_e] == new_e, self-inverse (handled)
            # So the check inv[new_e] >= s and inv[new_e] <= new_e covers it
            # BUT we also need: for all x in [s, e], inv[x] != new_e
            # inv[x] == new_e iff x == inv[new_e]
            # So we need inv[new_e] not in [s, e]
            # Which is: inv[new_e] < s or inv[new_e] > e
            # Combined with inv[new_e] != new_e (not self-inverse)
            # So: inv[new_e] < s or (inv[new_e] > e and inv[new_e] <= new_e)
            # Since new_e = e+1, inv[new_e] <= new_e and inv[new_e] > e means inv[new_e] == new_e
            # Which is self-inverse. So effectively: inv[new_e] < s
            # Wait, that's not right. Let me re-think.
            
            # For the interval [s, new_e] to be valid:
            # For all x in [s, new_e]: x*x % p != 1 AND inv[x] not in [s, new_e]
            # 
            # For x in [s, e] (already valid): inv[x] not in [s, e]. 
            # Now inv[x] could be new_e. So we need inv[x] != new_e for all x in [s,e].
            # inv[x] == new_e iff x == inv[new_e].
            # So we need inv[new_e] not in [s, e].
            #
            # For x = new_e: new_e*new_e % p != 1 (checked) AND inv[new_e] not in [s, new_e].
            # inv[new_e] not in [s, new_e] means inv[new_e] < s or inv[new_e] > new_e.
            # Since inv values are in [1, 1008], inv[new_e] > new_e is possible.
            
            # So the combined check for adding new_e:
            # 1. new_e * new_e % p != 1
            # 2. inv[new_e] not in [s, new_e]  (i.e., inv[new_e] < s or inv[new_e] > new_e)
            # 3. inv[new_e] not in [s, e]      (to ensure old elements don't conflict with new_e)
            #    But condition 2 already implies inv[new_e] < s or inv[new_e] > new_e >= e+1 > e
            #    So condition 3 is implied by condition 2!
            
            # Wait: condition 2 says inv[new_e] < s OR inv[new_e] > new_e.
            # If inv[new_e] < s, then inv[new_e] is not in [s, e]. Good.
            # If inv[new_e] > new_e = e+1, then inv[new_e] > e, so not in [s, e]. Good.
            # So condition 2 implies condition 3. 
            #
            # But wait, we also need: for all x in [s, e], inv[x] is not new_e.
            # inv[x] == new_e iff x == inv[new_e].
            # So we need inv[new_e] not in [s, e].
            # From condition 2: inv[new_e] < s or inv[new_e] > new_e > e.
            # So inv[new_e] is not in [s, e]. 
            # 
            # So we just need conditions 1 and 2!
            
            if inv[new_e] < s or inv[new_e] > new_e:
                e = new_e
            else:
                break
        
        # Now [s, e] is the maximal valid interval starting at s
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
