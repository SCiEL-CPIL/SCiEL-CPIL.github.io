"""Optimized solver for maximum reciprocal weight 3-AP-free subset of [1,256]."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    n = 256
    # Weights: 1/x for x in 1..n
    weights = [Fraction(1, x) for x in range(1, n + 1)]
    
    # Precompute AP relationships for faster conflict checking
    # For each x, precompute which pairs (a,b) form a 3-AP with x
    # Store as: for each x, a set of "partner" elements that would conflict
    # Actually, let's precompute for each pair (a,b) whether they form a 3-AP with some x
    
    # Better: precompute for each x, the list of y such that {x,y} is part of some 3-AP
    # For conflict checking: x conflicts with S if there exist a,b in S with {x,a,b} being 3-AP
    
    # Fast conflict check using set membership
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
    
    # Local search with 1-swap and 2-swap improvements
    improved = True
    max_iterations = 100  # Safety limit
    iteration = 0
    
    while improved and time.time() < deadline - 10 and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Phase 1: Try to add each missing element directly
        for x in range(1, n + 1):
            if x in best_set:
                continue
            if can_add_fast(x, best_set):
                best_set.add(x)
                best_weight += weights[x - 1]
                improved = True
        
        # Phase 2: 1-swap - try removing one element and adding another
        for x in range(1, n + 1):
            if x in best_set:
                continue
            # Find conflicts
            conflicts = set()
            for a in best_set:
                for b in best_set:
                    if a == b:
                        continue
                    if 2 * x == a + b or 2 * a == x + b or 2 * b == x + a:
                        conflicts.add(a)
                        conflicts.add(b)
            
            # Try removing each conflicting element
            for c in list(conflicts):
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
        
        # Phase 3: 2-swap - try removing two elements and adding one or two others
        # This is more expensive but can escape local optima
        if time.time() < deadline - 15:
            # Try removing pairs of elements and see if we can add better ones
            # For efficiency, only try promising pairs
            elements_in_set = sorted(best_set)
            # Try removing low-weight elements first
            for i in range(len(elements_in_set)):
                for j in range(i + 1, len(elements_in_set)):
                    c1 = elements_in_set[i]
                    c2 = elements_in_set[j]
                    # Only consider removing if they're among the lower-weight elements
                    if c1 > n // 2 and c2 > n // 2:
                        continue
                    
                    temp_set = best_set - {c1, c2}
                    removed_weight = weights[c1 - 1] + weights[c2 - 1]
                    
                    # Try adding each missing element
                    added_any = False
                    for x in range(1, n + 1):
                        if x in temp_set or x == c1 or x == c2:
                            continue
                        if can_add_fast(x, temp_set):
                            delta = weights[x - 1] - removed_weight
                            if delta > 0:
                                temp_set.add(x)
                                # Check if we can add another element
                                for y in range(1, n + 1):
                                    if y in temp_set or y == c1 or y == c2 or y == x:
                                        continue
                                    if can_add_fast(y, temp_set):
                                        delta2 = weights[y - 1] - removed_weight
                                        if delta2 > 0:
                                            temp_set.add(y)
                                            delta = delta + weights[y - 1]
                                            break
                                if delta > 0:
                                    best_set = temp_set
                                    best_weight += delta
                                    improved = True
                                    added_any = True
                                    break
                    
                    if added_any:
                        break
                if improved:
                    break
            if improved:
                break
    
    # Final polish: try adding any remaining valid elements multiple times
    for _ in range(3):
        for x in range(1, n + 1):
            if x not in best_set and can_add_fast(x, best_set):
                best_set.add(x)
                best_weight += weights[x - 1]
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
