"""Optimized solver for maximum reciprocal weight 3-AP-free subset of [1,256]."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    n = 256
    # Weights: 1/x for x in 1..n
    weights = [Fraction(1, x) for x in range(1, n + 1)]
    
    # Precompute for each x, the set of elements that form a 3-AP with x and one other element
    # conflict_pairs[x] = list of (a, b) such that {x, a, b} is a 3-AP
    # Actually, let's precompute for each x, which pairs (y, z) would create a 3-AP with x
    # i.e., x is involved in a 3-AP with y and z
    
    # For faster conflict checking, precompute for each x, the set of "partners"
    # partner[x] = set of y such that there exists z with {x, y, z} forming a 3-AP
    # But we need to check against the current set.
    
    # Better approach: for each x, precompute the list of (a, b) pairs such that
    # {x, a, b} is a 3-AP. Then x conflicts with S if any such pair is subset of S.
    
    # Even better: for each x, precompute the "conflict graph" - which elements in S
    # would prevent adding x.
    
    # Let's use a different strategy: precompute for each x, the list of all y such that
    # adding x would conflict with y (given some other element in S).
    
    # Actually, the cleanest approach: for each x, precompute all pairs (a, b) with a < b
    # such that {x, a, b} is a 3-AP. Store as a set of frozensets or sorted tuples.
    
    # For O(1) membership, we can check: does S contain both a and b for any such pair?
    # This is still O(number_of_pairs) per check.
    
    # Let's precompute for each x, the list of "required pairs" - pairs (a,b) such that
    # if both a and b are in S, then x cannot be added.
    
    required_pairs = {}
    for x in range(1, n + 1):
        pairs = []
        # Case 1: x is midpoint, a + b = 2x, a < x < b or a < b < x etc.
        # Actually, we need all pairs (a, b) with a != b, a,b in [1,n], a != x, b != x
        # such that {x, a, b} is a 3-AP.
        # Case 1: x = (a+b)/2 => a + b = 2x
        for a in range(1, n + 1):
            if a == x:
                continue
            b = 2 * x - a
            if 1 <= b <= n and b != x and b != a:
                pairs.append((min(a, b), max(a, b)))
        # Case 2: a is midpoint, 2a = x + b => b = 2a - x
        for a in range(1, n + 1):
            if a == x:
                continue
            b = 2 * a - x
            if 1 <= b <= n and b != x and b != a:
                pairs.append((min(x, b), max(x, b)))
        # Case 3: b is midpoint, 2b = x + a => b = (x+a)/2
        for a in range(1, n + 1):
            if a == x:
                continue
            s = x + a
            if s % 2 == 0:
                b = s // 2
                if 1 <= b <= n and b != x and b != a:
                    pairs.append((min(x, a), max(x, a)))
        
        # Deduplicate
        required_pairs[x] = set(pairs)
    
    # Now can_add can check: for each pair in required_pairs[x], if both elements are in S, conflict.
    # But this is still O(num_pairs) per check. Let's optimize further.
    
    # Alternative: for each x, precompute the "conflict set" - the set of elements y such that
    # there exists z with {x, y, z} forming a 3-AP. Then x conflicts with S if
    # (conflict_set[x] & S) is non-empty AND for some y in that intersection, the partner z is also in S.
    
    # Actually, let's just use the fast check with set membership, which is O(k) where k = |S|.
    
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
    
    # Greedy initial solution
    greedy_set = set()
    for x in range(1, n + 1):
        if can_add_fast(x, greedy_set):
            greedy_set.add(x)
    
    best_weight = sum(weights[x - 1] for x in greedy_set)
    best_set = greedy_set.copy()
    
    if time.time() > deadline - 5:
        return sorted(best_set)
    
    # Local search: try to add elements or swap
    improved = True
    max_iterations = 50
    iteration = 0
    while improved and time.time() < deadline - 10 and iteration < max_iterations:
        improved = False
        iteration += 1
        
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
                # Find all conflicting elements
                # An element c in best_set conflicts with x if there exists d in best_set
                # such that {x, c, d} is a 3-AP
                conflicts = set()
                for c in best_set:
                    # Check if c is part of a 3-AP with x and some other element in best_set
                    # Case 1: x is midpoint, c + d = 2x => d = 2x - c
                    d = 2 * x - c
                    if d in best_set and d != c and d != x:
                        conflicts.add(c)
                        conflicts.add(d)
                    # Case 2: c is midpoint, 2c = x + d => d = 2c - x
                    d = 2 * c - x
                    if d in best_set and d != x and d != c and d > 0:
                        conflicts.add(c)
                        conflicts.add(d)
                    # Case 3: d is midpoint, 2d = x + c => d = (x+c)/2
                    s = x + c
                    if s % 2 == 0:
                        d = s // 2
                        if d in best_set and d != x and d != c:
                            conflicts.add(c)
                            conflicts.add(d)
                
                # Try removing each conflicting element, sorted by weight (remove lowest weight first)
                for c in sorted(conflicts, key=lambda y: weights[y - 1]):
                    if c == x:
                        continue
                    temp_set = best_set - {c}
                    if can_add_fast(x, temp_set):
                        delta = weights[x - 1] - weights[c - 1]
                        if delta > 0:
                            best_set = temp_set
                            best_set.add(x)
                            best_weight += delta
                            improved = True
                            break
    
    if time.time() > deadline - 3:
        return sorted(best_set)
    
    # Final polish: try adding any remaining elements
    for x in range(1, n + 1):
        if x not in best_set and can_add_fast(x, best_set):
            best_set.add(x)
            best_weight += weights[x - 1]
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
