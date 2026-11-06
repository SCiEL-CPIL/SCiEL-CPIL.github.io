"""Optimized solver for maximum reciprocal weight 3-AP-free subset of [1,256]."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    n = 256
    weights = [Fraction(1, x) for x in range(1, n + 1)]
    
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
    
    # Aggressive local search with multi-swap
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        
        # Phase 1: Try to add each element not in best_set
        for x in range(1, n + 1):
            if x in best_set:
                continue
            if can_add_fast(x, best_set):
                best_set.add(x)
                best_weight += weights[x - 1]
                improved = True
        
        if time.time() > deadline - 15:
            break
        
        # Phase 2: Single-swap (remove one, add one)
        for x in range(1, n + 1):
            if x in best_set:
                continue
            if can_add_fast(x, best_set):
                continue  # Already handled in phase 1
            
            # Find all elements in best_set that conflict with x
            conflicts = set()
            for a in best_set:
                for b in best_set:
                    if a == b:
                        continue
                    if 2 * x == a + b or 2 * a == x + b or 2 * b == x + a:
                        conflicts.add(a)
                        conflicts.add(b)
            
            # Try removing each conflicting element
            for c in conflicts:
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
        
        if time.time() > deadline - 15:
            break
        
        # Phase 3: Two-swap (remove up to 2, add up to 2) - more aggressive
        # Try removing pairs of elements and see if we can add more valuable ones
        for x in range(1, n + 1):
            if x in best_set:
                continue
            if can_add_fast(x, best_set):
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
            
            # Try removing pairs from conflicts
            conflict_list = list(conflicts)
            for i in range(len(conflict_list)):
                for j in range(i + 1, len(conflict_list)):
                    c1, c2 = conflict_list[i], conflict_list[j]
                    temp_set = best_set - {c1, c2}
                    if can_add_fast(x, temp_set):
                        delta = weights[x - 1] - weights[c1 - 1] - weights[c2 - 1]
                        if delta > 0:
                            best_set = temp_set
                            best_set.add(x)
                            best_weight += delta
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    if time.time() > deadline - 5:
        return sorted(best_set)
    
    # Final polish: try adding any remaining elements
    for x in range(1, n + 1):
        if x not in best_set and can_add_fast(x, best_set):
            best_set.add(x)
            best_weight += weights[x - 1]
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
