"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    """
    Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    """
    target_val = math.factorial(12)
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_exps = [10, 5, 2, 1, 1]
    target_tuple = tuple(target_exps)
    
    # Generate all divisors of 12!
    divisors = []
    for e0 in range(11):
        for e1 in range(6):
            for e2 in range(3):
                for e3 in range(2):
                    for e4 in range(2):
                        d = (2**e0) * (3**e1) * (5**e2) * (7**e3) * (11**e4)
                        if d > 12:
                            # Store (value, exponent_tuple)
                            exps_t = (e0, e1, e2, e3, e4)
                            divisors.append((d, exps_t))
    
    # Sort by value to ensure we check smaller max factors first
    divisors.sort(key=lambda x: x[0])
    
    N = len(divisors)
    
    # DP to find the smallest index k such that target_tuple is reachable
    # using a subset of divisors[0..k]
    # reachable[i] is a set of exponent tuples reachable using divisors[0..i]
    
    # Optimization: We can use a list of sets.
    # reachable[0] = { (0,0,0,0,0), divisors[0][1] }
    # reachable[i] = reachable[i-1] union { s + divisors[i][1] for s in reachable[i-1] if s + divisors[i][1] <= target }
    
    # To be efficient, we only add states that are <= target component-wise.
    
    reachable = [set() for _ in range(N)]
    
    # Base case: index -1 (empty set)
    # We'll handle index 0 manually or use a virtual index -1
    
    # Let's use a list where reachable[i] corresponds to using items 0..i
    # Initialize reachable for i=-1 conceptually as { (0,0,0,0,0) }
    
    current_reachable = { (0,0,0,0,0) }
    
    min_k = -1
    for i in range(N):
        val, exps = divisors[i]
        next_reachable = set(current_reachable)
        
        for s in current_reachable:
            new_s = tuple(s[j] + exps[j] for j in range(5))
            # Check if new_s <= target
            valid = True
            for j in range(5):
                if new_s[j] > target_exps[j]:
                    valid = False
                    break
            if valid:
                next_reachable.add(new_s)
        
        reachable[i] = next_reachable
        current_reachable = next_reachable
        
        if target_tuple in current_reachable:
            min_k = i
            break
            
    if min_k == -1:
        # Fallback, should not happen
        return [target_val]
        
    # Reconstruct the solution
    # We know a solution exists using divisors[0..min_k]
    # We use backtracking with pruning using the precomputed 'reachable' sets
    
    def backtrack(rem, idx):
        if rem == (0,0,0,0,0):
            return []
        if idx < 0:
            return None
            
        # Pruning: Is 'rem' reachable using items 0..idx?
        if rem not in reachable[idx]:
            return None
            
        # Try using divisors[idx]
        val, exps = divisors[idx]
        # Check if we can subtract exps from rem
        can_use = True
        new_rem = [0]*5
        for j in range(5):
            if rem[j] < exps[j]:
                can_use = False
                break
            new_rem[j] = rem[j] - exps[j]
        
        if can_use:
            # Recurse
            res = backtrack(tuple(new_rem), idx - 1)
            if res is not None:
                return [val] + res
                
        # Try not using divisors[idx]
        res = backtrack(rem, idx - 1)
        if res is not None:
            return res
            
        return None

    solution = backtrack(target_tuple, min_k)
    
    if solution is None:
        return [target_val]
        
    return solution

# EVOLVE-BLOCK-END
