import time
import random
from fractions import Fraction

def construct():
    N = 256
    start_time = time.time()
    time_limit = 80.0  # Safety margin before 90s kill
    
    # Precompute reciprocals as Fractions for exact arithmetic
    recip = [Fraction(0)] + [Fraction(1, i) for i in range(1, N + 1)]
    
    # Precompute conflict: for each x, which other elements conflict with x
    # Two elements conflict if they are part of a 3-AP with a third element in [1,N]
    # Actually, we need: for a set S, x can be added iff for all a in S,
    #   (2x - a) not in S [x is middle]
    #   (2a - x) not in S [a is middle, x is end]
    
    # Precompute for each x, the set of pairs (a, c) such that a, x, c form AP
    # and for each x, the set of pairs where x is an endpoint
    
    # For branch and bound, we process elements in order of decreasing weight (1/x)
    # i.e., from 1 to N (smallest first, highest weight)
    
    # Precompute: for each x, what elements would be excluded if we include x?
    # If we include x, we cannot include any a such that 2x-a is also in the set
    # This is complex. Let's use a different approach.
    
    # Branch and bound with bitmask representation
    # We process elements in order 1, 2, ..., N (decreasing weight)
    # At each step, we decide to include or exclude the current element
    
    # Precompute for fast AP checking
    # For a given mask, can we add element x?
    def can_add(x, mask):
        """Check if x can be added to set represented by mask without creating 3-AP."""
        if mask & (1 << x):
            return False
        # For each a in mask, check if adding x creates AP
        m = mask
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            m ^= lsb
            # x is middle: a, x, c where c = 2x - a
            c = 2 * x - a
            if 1 <= c <= N and c != a and (mask & (1 << c)):
                return False
            # a is middle: a is middle of x and c where c = 2a - x
            c2 = 2 * a - x
            if 1 <= c2 <= N and c2 != a and c2 != x and (mask & (1 << c2)):
                return False
        return True
    
    # Score of a mask
    def score_mask(mask):
        s = Fraction(0)
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            s += recip[x]
            m ^= lsb
        return s
    
    # Greedy initialization: add elements in order 1, 2, ..., N if possible
    greedy_mask = 0
    for x in range(1, N + 1):
        if can_add(x, greedy_mask):
            greedy_mask |= (1 << x)
    
    greedy_score = score_mask(greedy_mask)
    best_mask = greedy_mask
    best_score = greedy_score
    
    # Branch and bound
    # Process elements in order of decreasing weight: 1, 2, 3, ..., N
    # At index i, we consider element i+1
    # State: (index, current_mask, current_score)
    # Upper bound: current_score + sum of weights of remaining elements that could potentially be added
    
    # For a tighter upper bound, we can compute the maximum possible additional weight
    # by greedily trying to add remaining elements
    
    def upper_bound(idx, mask, score):
        """Compute an upper bound on the final score."""
        # Simple bound: add all remaining weights
        # This is loose but fast
        remaining_sum = sum(recip[i] for i in range(idx, N + 1))
        return score + remaining_sum
    
    # Better upper bound: greedy fill
    def greedy_upper_bound(idx, mask, score):
        """Try to greedily add remaining elements to get a tighter bound."""
        temp_mask = mask
        temp_score = score
        for i in range(idx, N + 1):
            if can_add(i, temp_mask):
                temp_mask |= (1 << i)
                temp_score += recip[i]
        return temp_score
    
    # Branch and bound search
    # We use iterative deepening or just DFS with pruning
    # Given time constraints, we'll do a limited DFS
    
    nodes_explored = 0
    max_nodes = 500000  # Limit to prevent timeout
    
    def branch_and_bound(idx, mask, score):
        nonlocal best_mask, best_score, nodes_explored
        
        if time.time() - start_time > time_limit - 3:
            return
        
        if idx > N:
            # All elements processed
            if score > best_score:
                best_score = score
                best_mask = mask
            return
        
        nodes_explored += 1
        if nodes_explored > max_nodes:
            return
        
        # Upper bound check
        ub = greedy_upper_bound(idx, mask, score)
        if ub <= best_score:
            return
        
        # Option 1: Include element idx
        if can_add(idx, mask):
            new_mask = mask | (1 << idx)
            new_score = score + recip[idx]
            branch_and_bound(idx + 1, new_mask, new_score)
        
        # Option 2: Exclude element idx
        branch_and_bound(idx + 1, mask, score)
    
    # Start branch and bound from greedy solution
    # We can also try starting from empty set, but greedy is a good initial solution
    branch_and_bound(1, 0, Fraction(0))
    
    # If branch and bound didn't improve, use greedy
    # Extract best set
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        x = lsb.bit_length() - 1
        result.append(x)
        m ^= lsb
    
    if not result:
        # Fallback: return at least one element
        result = [1]
    
    return result
