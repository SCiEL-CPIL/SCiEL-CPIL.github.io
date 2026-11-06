"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    target = Fraction(5, 31)
    best_solution = None
    best_max_denom = 10**7 + 1
    
    # Initial greedy solution to set an upper bound
    r = target
    greedy_vals = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        greedy_vals.append(d)
        r -= Fraction(1, d)
    if greedy_vals:
        best_max_denom = max(greedy_vals)
        best_solution = greedy_vals[:]
        
    # DFS to find a better solution (smaller max denominator)
    # We want to minimize the largest denominator.
    # We explore denominators in increasing order.
    
    def dfs(rem, min_d, k, current_path):
        nonlocal best_solution, best_max_denom, deadline
        
        if time.time() > deadline:
            return None
            
        if k == 0:
            if rem == 0:
                return []
            return None
            
        if rem <= 0:
            return None
            
        # Lower bound for d
        # 1/d <= rem => d >= 1/rem
        d_min = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        # Upper bound for d
        # rem - 1/d must be representable by k-1 terms starting from d+1.
        # Max sum of k-1 terms starting from d+1 is approx sum_{i=1}^{k-1} 1/(d+i)
        # If rem - 1/d > max_sum, then d is too small (we need to subtract more, so smaller d? 
        # Wait. If rem is fixed.
        # Let f(d) = rem - 1/d. As d increases, 1/d decreases, so f(d) increases.
        # Let g(d) = sum_{i=1}^{k-1} 1/(d+i). As d increases, g(d) decreases.
        # We need f(d) <= g(d).
        # This inequality holds for small d and fails for large d.
        # So this gives an UPPER bound on d.
        
        # Calculate max_sum for k-1 terms
        # If k-1 is large, this sum is small.
        max_sum = Fraction(0, 1)
        for i in range(1, k):
            max_sum += Fraction(1, d_min + i) # Using d_min as a starting point for the bound check is tricky.
            # Actually, we should check the condition for each d.
            # But we can estimate d_max first.
            
        # A simpler upper bound:
        # The smallest possible denominator for the next term is d+1.
        # The largest possible value for the sum of the remaining k-1 terms is 
        # achieved by taking the smallest possible denominators: d+1, d+2, ..., d+k-1.
        # So we require rem - 1/d <= sum_{j=1}^{k-1} 1/(d+j).
        # Let's find d_max by iterating from d_min upwards until this condition fails.
        # Since d_min is the smallest valid d, and the condition holds for small d,
        # we can iterate d from d_min and stop when the condition fails.
        
        d = d_min
        while True:
            if time.time() > deadline:
                return None
                
            # Prune if d is already larger than the best max found
            if d >= best_max_denom:
                break
                
            # Check if rem - 1/d is too large for the remaining k-1 terms
            if k > 1:
                # Calculate max possible sum of k-1 terms starting from d+1
                max_possible_rem = Fraction(0, 1)
                for j in range(1, k):
                    max_possible_rem += Fraction(1, d + j)
                
                current_rem = rem - Fraction(1, d)
                if current_rem > max_possible_rem:
                    # d is too small? No, if current_rem > max_possible_rem, 
                    # it means we haven't subtracted enough. We need a smaller d (larger 1/d).
                    # But d is already at the lower bound d_min. 
                    # If d_min doesn't work, no larger d will work either (because 1/d gets smaller, current_rem gets larger).
                    # So if it fails at d_min, the branch is dead.
                    break
                    
            # Check if rem - 1/d is too small?
            # The minimum sum of k-1 terms starting from d+1 is not bounded below by 0, but by 0.
            # However, if current_rem is very small, we might need a very large denominator.
            # But we are bounded by best_max_denom.
            # If current_rem > 0, we can always find a denominator large enough?
            # Yes, but it might exceed best_max_denom.
            
            # Recurse
            current_path.append(d)
            res = dfs(rem - Fraction(1, d), d + 1, k - 1, current_path)
            current_path.pop()
            
            if res is not None:
                # Found a valid completion
                full_path = current_path + res
                max_d = max(full_path)
                if max_d < best_max_denom:
                    best_max_denom = max_d
                    best_solution = full_path[:]
                # We can return to try other d's?
                # Since we iterate d in increasing order, and we want to minimize the MAX,
                # a smaller d might lead to a larger max later, or a smaller max later.
                # We should continue searching to see if a larger d leads to a smaller max overall?
                # Unlikely, but possible. However, for performance, we might want to stop 
                # if we find a solution with max_d == d (i.e., d is the max).
                # But d is the smallest in the path so far. The max is likely in the rest.
                # Let's continue to be safe, but prune if best_max_denom == d_min (theoretical min).
                
            # Move to next d
            d += 1
            
        return None

    # We need to determine the maximum number of terms k.
    # The greedy solution has 3 terms.
    # Could a 4-term solution have a smaller max?
    # 1/8 + 1/28 + 1/1736. Max 1736.
    # Try k=4.
    # If we use 4 terms, we can spread the value, potentially keeping denominators smaller.
    # Let's search for k=2, 3, 4, 5.
    
    for k in range(2, 6):
        dfs(target, 1, k, [])
        if time.time() > deadline:
            break

    return best_solution

# EVOLVE-BLOCK-END
