"""Optimized solver for maximum reciprocal weight 3-AP-free subset of [1,256]."""
import time
from fractions import Fraction
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    n = 256
    # Weights: 1/x for x in 1..n
    weights = [Fraction(1, x) for x in range(1, n + 1)]
    float_weights = [0.0] + [1.0 / i for i in range(1, n + 1)]

    # Precompute for each x, the set of elements y such that there exists z with {x, y, z} forming a 3-AP
    # This helps in fast conflict checking.
    # Actually, for a set S, x conflicts if there exist a, b in S such that {x, a, b} is a 3-AP.
    # We can precompute for each x, the list of "partner" pairs? No, that's too much.
    # Let's stick to the O(|S|) check which is fast enough for |S| ~ 100-200.
    
    def can_add_fast(x, current_set):
        """Fast check if x can be added to current_set without creating a 3-AP."""
        # Case 1: x is midpoint, a + b = 2x, so b = 2x - a
        for a in current_set:
            b = 2 * x - a
            if b in current_set and b != a:
                return False
        # Case 2: a is midpoint, 2a = x + b, so b = 2a - x
        for a in current_set:
            b = 2 * a - x
            if b in current_set and b != x and b > 0:
                return False
        # Case 3: b is midpoint, 2b = x + a, so b = (x+a)/2
        for a in current_set:
            s = x + a
            if s % 2 == 0:
                b = s // 2
                if b in current_set and b != x and b != a:
                    return False
        return True

    # Greedy initial solution: add elements in order of decreasing weight (increasing index)
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
