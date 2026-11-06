import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit
    
    n = 256
    # Use Fraction for exact comparison to avoid floating point issues
    # though float is likely fine for this magnitude, Fraction is safer for "exact" requirement in details?
    # The prompt says "score is its floating serialization with exact numerator/denominator in details".
    # This implies the evaluator calculates the exact sum.
    # For our internal search, float is faster. We can use float for comparison and Fraction for final output if needed?
    # The return is just the list. The evaluator computes the score.
    # So we just need to maximize the sum. Float is sufficient for comparison during search.
    
    def has_3ap_with(x, S):
        """
        Check if adding x to S creates a 3-AP.
        S is a set of integers.
        """
        # Case 1: x is the middle term. a, x, b. a+b = 2x.
        # a and b must be in S.
        # Iterate a in S, check if 2x - a in S.
        # We must ensure a != b? No, if a=b, then 2a=2x => a=x. But x not in S yet.
        # Also a < x < b or b < x < a.
        # Just check if there exists a in S such that 2x-a in S and 2x-a != a?
        # If 2x-a == a, then a=x. x not in S. So a != 2x-a is guaranteed if x not in S.
        for a in S:
            b = 2 * x - a
            if b in S:
                return True
        
        # Case 2: x is an end term.
        # x, a, b => 2a - x = b.
        # a, b, x => 2b - x = a.
        # These are symmetric. We just need to check if there is an a in S such that 2a-x in S.
        # Wait, if x is the first term, x < a < b.
        # If x is the last term, a < b < x.
        # In both cases, there exist a, b in S such that x, a, b form an AP.
        # This is equivalent to checking if there is an a in S such that (2a - x) in S.
        # Let's verify.
        # If x is first: x, a, b. b = 2a - x. So if we find a in S such that 2a-x in S, we have a 3-AP.
        # If x is last: a, b, x. a = 2b - x. So if we find b in S such that 2b-x in S, we have a 3-AP.
        # So checking `any((2*a - x) in S for a in S)` covers both end cases.
        
        for a in S:
            b = 2 * a - x
            if b in S:
                return True
        return False

    def greedy_construct():
        S = set()
        for x in range(1, n + 1):
            if not has_3ap_with(x, S):
                S.add(x)
        return S

    def calculate_score(S):
        return sum(1.0 / x for x in S)

    # Initial Solution
    best_set = greedy_construct()
    best_score = calculate_score(best_set)
    
    # Local Search
    # We will try to improve by removing one element and re-greedy filling.
    # To avoid getting stuck in a local optimum, we can randomize the removal.
    
    # Pre-compute list of all numbers
    all_numbers = list(range(1, n + 1))
    
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        
        # Choose a removal candidate
        # Strategy:
        # 1. Random element
        # 2. Largest element (lowest weight)
        # 3. Element that blocks the most "small" candidates? (Too expensive to compute every time)
        
        if not best_set:
            break
            
        # Mix of random and greedy (largest)
        if random.random() < 0.5:
            rem_candidate = random.choice(list(best_set))
        else:
            rem_candidate = max(best_set)
            
        # Create a temporary set
        temp_set = set(best_set)
        temp_set.remove(rem_candidate)
        
        # Greedy fill
        # We want to add numbers from [1, 256] \ temp_set
        # Sorted by ascending value (highest reciprocal first)
        candidates = [x for x in all_numbers if x not in temp_set]
        
        # Optimization: We can just iterate 1..256 and check if not in temp_set
        # This is faster than building a list and sorting if we just iterate in order.
        # But we need to add them if valid.
        # If we iterate 1..256, we are effectively doing the same greedy fill.
        
        for x in range(1, n + 1):
            if x not in temp_set:
                if not has_3ap_with(x, temp_set):
                    temp_set.add(x)
                    
        new_score = calculate_score(temp_set)
        
        if new_score > best_score:
            best_score = new_score
            best_set = temp_set
        else:
            # If we didn't improve, we might want to keep the current best and try another removal.
            # The loop naturally does this by picking a new candidate next iteration.
            pass
            
        # Optional: If we've been stuck for a while, try a more aggressive restart?
        # For now, the local search is strong enough.
        
    # Ensure we return a list
    return sorted(best_set)

# EVOLVE-BLOCK-END
