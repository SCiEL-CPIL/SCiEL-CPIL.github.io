"""Optimized solver for maximum reciprocal weight 3-AP-free subset of [1,256]."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    n = 256
    # Weights: 1/x for x in 1..n
    weights = [Fraction(1, x) for x in range(1, n + 1)]
    
    # Precompute for each element x, the list of pairs (i,j) indices such that
    # {x, i+1, j+1} forms a 3-AP. Actually, let's think in terms of values.
    # For a new element x, it conflicts with existing set S if there exist a,b in S
    # such that {x,a,b} is a 3-AP.
    
    # Alternative: precompute for each x, which other elements y,z would form 3-AP with x
    # i.e., for each x, find all pairs (y,z) such that {x,y,z} is a 3-AP.
    
    # Let's use a different approach: branch and bound with greedy coloring upper bound.
    
    # Order elements by decreasing weight (1/1, 1/2, 1/3, ...)
    # This is already the natural order 1, 2, 3, ...
    
    best_weight = Fraction(0)
    best_set = set()
    
    # Greedy initial solution
    greedy_set = set()
    for x in range(1, n + 1):
        valid = True
        for a in greedy_set:
            for b in greedy_set:
                if a == b:
                    continue
                # Check if {x, a, b} forms a 3-AP
                # x is midpoint: 2x == a+b
                # a is midpoint: 2a == x+b
                # b is midpoint: 2b == x+a
                if 2 * x == a + b or 2 * a == x + b or 2 * b == x + a:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            greedy_set.add(x)
    
    best_weight = sum(weights[x - 1] for x in greedy_set)
    best_set = greedy_set.copy()
    
    if time.time() > deadline - 5:
        return sorted(best_set)
    
    # Branch and bound
    # Process elements in order 1 to n (decreasing weight)
    # State: index, current set, current weight
    # Upper bound: current weight + sum of weights of all remaining elements (optimistic)
    
    # Precompute suffix sums of weights for upper bound
    suffix_sum = [Fraction(0)] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i - 1]
    
    # For faster conflict checking, precompute for each x the set of pairs
    # Actually, let's precompute for each x, a bitmask or set of elements that
    # would conflict. But the conflict depends on the current set.
    
    # Let's use a different upper bound: for each remaining element, check if it
    # could potentially be added (not conflicting with current set). Sum those weights.
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without forming a 3-AP."""
        for a in current_set:
            for b in current_set:
                if a == b:
                    continue
                if 2 * x == a + b or 2 * a == x + b or 2 * b == x + a:
                    return False
        return True
    
    # This is O(k^2) per check where k is size of current set. 
    # For 256 elements, this could be slow. Let's optimize.
    
    # Precompute for each x, the set of "forbidden" pairs. Actually, let's think:
    # x conflicts with current_set if there exist a,b in current_set with a+b=2x or 
    # a+x=2b or b+x=2a.
    # Equivalently: x conflicts if there exists a in current_set such that:
    #   - 2x - a is in current_set (x is midpoint of a and 2x-a)
    #   - a - x is in current_set and a - x > 0... wait no.
    # Let me re-derive: {x,a,b} is 3-AP iff one is midpoint.
    # Case 1: x = (a+b)/2 => a+b = 2x => b = 2x-a
    # Case 2: a = (x+b)/2 => 2a = x+b => b = 2a-x
    # Case 3: b = (x+a)/2 => 2b = x+a => b = (x+a)/2, need x+a even
    #
    # So x conflicts if:
    # - There exists a in S such that 2x-a in S and 2x-a != a (i.e., a != x)
    # - There exists a in S such that 2a-x in S and 2a-x != x and 2a-x > 0
    # - There exists a in S such that (x+a) is even and (x+a)//2 in S and (x+a)//2 != x and (x+a)//2 != a
    #
    # This is still O(k) per check if we use a set for membership.
    
    def can_add_fast(x, current_set):
        """Fast check using set membership."""
        # Case 1: x is midpoint, so a+b=2x, b=2x-a
        for a in current_set:
            b = 2 * x - a
            if b in current_set and b != a:
                return False
        # Case 2: a is midpoint, so 2a=x+b, b=2a-x
        for a in current_set:
            b = 2 * a - x
            if b in current_set and b != x and b > 0:
                return False
        # Case 3: b is midpoint, so 2b=x+a, b=(x+a)/2
        for a in current_set:
            s = x + a
            if s % 2 == 0:
                b = s // 2
                if b in current_set and b != x and b != a:
                    return False
        return True
    
    if time.time() > deadline - 5:
        return sorted(best_set)
    
    # Now branch and bound
    # We'll do DFS processing elements 1 to n
    # To make it faster, let's limit the depth or use a better heuristic
    
    # Actually, let's try a local search approach instead, which might be more effective
    # Start from greedy, try to swap elements to improve weight
    
    # Try: for each element not in set, see if we can add it by removing some conflicts
    # Or: try to replace a low-weight element with a higher-weight one
    
    # Let's do a simple improvement: try to add each missing element, and if it conflicts,
    # see if removing one conflicting element allows adding it with net gain
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try to add each element not in best_set
        for x in range(1, n + 1):
            if x in best_set:
                continue
            if can_add_fast(x, best_set):
                # Can add directly
                best_set.add(x)
                best_weight += weights[x - 1]
                improved = True
            else:
                # Find conflicts and see if we can swap
                # Find all elements a in best_set such that adding x would conflict
                # We need to remove at least one element from each conflicting 3-AP
                # This is a hitting set problem. For simplicity, try removing each
                # conflicting element one at a time
                conflicts = []
                for a in best_set:
                    for b in best_set:
                        if a == b:
                            continue
                        if 2 * x == a + b or 2 * a == x + b or 2 * b == x + a:
                            # {x,a,b} is a 3-AP, we need to break it
                            # Removing a or b would break it
                            if a not in conflicts:
                                conflicts.append(a)
                            if b not in conflicts:
                                conflicts.append(b)
                
                # Try removing each conflicting element
                for c in conflicts:
                    if c == x:
                        continue
                    # If we remove c, can we add x?
                    # Check if x conflicts with best_set - {c}
                    temp_set = best_set - {c}
                    if can_add_fast(x, temp_set):
                        # Net change: add x, remove c
                        delta = weights[x - 1] - weights[c - 1]
                        if delta > 0:
                            best_set = temp_set
                            best_set.add(x)
                            best_weight += delta
                            improved = True
                            break
    
    if time.time() > deadline - 3:
        return sorted(best_set)
    
    # Try more aggressive local search: 2-swap
    # Replace two elements with two others
    # This is more complex, let's skip if time is tight
    
    # Final: try adding any remaining elements
    for x in range(1, n + 1):
        if x not in best_set and can_add_fast(x, best_set):
            best_set.add(x)
            best_weight += weights[x - 1]
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
