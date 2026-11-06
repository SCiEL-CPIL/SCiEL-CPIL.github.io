import math

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    
    # Known optimal solution with 9 distinct pairwise coprime sums (all prime)
    A = [4, 16, 40]
    B = [1, 3, 7]
    
    # Verification
    sums = [a + b for a in A for b in B]
    distinct_sums = list(set(sums))
    
    # Check validity
    valid = len(distinct_sums) == 9
    if valid:
        for i in range(len(distinct_sums)):
            for j in range(i + 1, len(distinct_sums)):
                if math.gcd(distinct_sums[i], distinct_sums[j]) > 1:
                    valid = False
                    break
            if not valid:
                break
                
    if valid:
        return [A, B]
    else:
        # Fallback: Another valid solution (suboptimal or optimal if found)
        # If the primary fails (which it shouldn't), we need a valid return.
        # Since we can't easily find another 9-prime grid, we return a valid 5-sum set?
        # No, the task wants to maximize. If we return 5, we lose.
        # But the primary is hardcoded and correct. This branch is dead code.
        # We return the primary anyway or a simple valid set.
        return [A, B]
