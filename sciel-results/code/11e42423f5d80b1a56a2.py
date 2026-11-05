"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # A set S is 3-AP-free if for all distinct a,b in S, 2*b - a is not in S
    # (and a != b, so we check a < b to avoid duplicates)
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a 3-AP."""
        # x can be the smallest, middle, or largest in a 3-AP
        # If x is smallest: need a,b in S with x + b = 2a -> b = 2a - x, so 2a-x in S for some a in S, a > x
        # If x is middle: need a,c in S with a + c = 2x -> c = 2x - a, so 2x-a in S for some a in S
        # If x is largest: need a,b in S with a + x = 2b -> a = 2b - x, so 2b-x in S for some b in S, b < x
        
        # More systematically: for each a in S, check if the third term of AP involving a and x is in S
        # APs involving x: (x, y, z) where 2y = x+z
        #   If x is first: y = (x+z)/2, need z in S and (x+z) even and y in S
        #   If x is second: y = (a+x)/2 for some a in S, need (a+x) even and y in S  
        #   If x is third: y = (a+x)/2 for some a in S, need (a+x) even and y in S
        
        # Simpler: for each a in S with a != x:
        #   Check if a, x, and some c form AP: 2x = a+c -> c = 2x-a, check c in S
        #   Check if a, b, x form AP: 2b = a+x -> b = (a+x)/2, check (a+x) even and b in S
        #   Check if x, b, a form AP: 2b = x+a -> same as above
        
        for a in S:
            if a == x:
                continue
            # x is middle: a, x, c -> c = 2x - a
            c = 2 * x - a
            if c in S and c != a:
                return False
            # x is first or last: a, b, x or x, b, a -> b = (a+x)/2
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in S and b != a and b != x:
                    return False
        return True
    
    # Greedy construction: add numbers in order of increasing value (decreasing reciprocal weight)
    # This prioritizes small numbers which have higher weight
    S = set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)
    
    current_score = sum(Fraction(1, x) for x in S)
    
    # Local search: hill climbing with swaps
    # Try to remove a large element and add a smaller one that was previously blocked
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Try adding any missing small number (1 to 128) if possible
        for x in range(1, 129):
            if time.time() > deadline - 10:
                break
            if x in S:
                continue
            if can_add(x, S):
                S.add(x)
                current_score += Fraction(1, x)
                improved = True
        
        # Try swapping: remove a large element, see if we can add a smaller one
        for x in range(256, 128, -1):
            if time.time() > deadline - 10:
                break
            if x not in S:
                continue
            
            # Try removing x
            S.remove(x)
            old_score = current_score
            current_score -= Fraction(1, x)
            
            # Try adding smaller numbers
            found_improvement = False
            for y in range(1, x):
                if y in S:
                    continue
                if can_add(y, S):
                    S.add(y)
                    current_score += Fraction(1, y)
                    if current_score > old_score:
                        found_improvement = True
                        improved = True
                        break
                    else:
                        S.remove(y)
                        current_score -= Fraction(1, y)
            
            if not found_improvement:
                # Revert: put x back
                S.add(x)
                current_score += Fraction(1, x)
    
    # Final verification: ensure no 3-AP exists
    S_list = sorted(S)
    valid = True
    S_set = set(S_list)
    for i, a in enumerate(S_list):
        for b in S_list[i+1:]:
            c = 2 * b - a
            if c in S_set:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: rebuild greedily (shouldn't happen if logic is correct)
        S = set()
        for x in range(1, 257):
            if can_add(x, S):
                S.add(x)
    
    return list(S)
# EVOLVE-BLOCK-END
