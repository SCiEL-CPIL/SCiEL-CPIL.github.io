"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def construct():
    """
    Find large sum-free subsets for three instances.
    Uses multi-start greedy with diverse orderings plus local search refinement.
    """
    start_time = time.time()
    DEADLINE = 120  # seconds, leave margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_sum_free(elements, order_key=None, rng=None):
        """
        Greedy construction: add elements in given order if they maintain sum-freeness.
        order_key: function to sort by, or None for original order
        """
        if order_key is not None:
            elements = sorted(elements, key=order_key)
        
        S = []
        S_set = set()
        
        for x in elements:
            # Check if adding x violates sum-freeness
            can_add = True
            # x + y in S or y + x in S for some y in S
            for y in S_set:
                if (x + y) in S_set or (y + x) in S_set:
                    can_add = False
                    break
            # Also check 2*x in S
            if 2 * x in S_set:
                can_add = False
            
            if can_add:
                S.append(x)
                S_set.add(x)
        
        return S

    def local_search_improve(S, elements, max_iters=500, rng=None):
        """
        Try to improve S by removing and re-adding elements.
        Returns improved S.
        """
        S = list(S)
        S_set = set(S)
        elements_set = set(elements)
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            # Try removing each element and see if we can add a better one
            for i in range(len(S)):
                if time.time() - start_time > DEADLINE:
                    break
                
                x = S[i]
                # Remove x
                S_set.remove(x)
                S.pop(i)
                
                # Try to add elements from elements not in S
                candidates = [e for e in elements if e not in S_set]
                # Sort by some heuristic
                candidates.sort(key=lambda e: abs(e), reverse=True)
                
                best_add = None
                for c in candidates:
                    can_add = True
                    for y in S_set:
                        if (c + y) in S_set or (y + c) in S_set:
                            can_add = False
                            break
                    if 2 * c in S_set:
                        can_add = False
                    if can_add:
                        best_add = c
                        break
                
                if best_add is not None and len(S) + 1 > len(S) - 1:
                    # We removed 1, can add 1, so size stays same or we need net gain
                    # Actually we need to find if we can add MORE than we removed
                    pass
                
                # Put x back
                S.insert(i, x)
                S_set.add(x)
            
            # Alternative: try to add elements directly
            candidates = [e for e in elements if e not in S_set]
            candidates.sort(key=lambda e: abs(e), reverse=True)
            
            for c in candidates:
                can_add = True
                for y in S_set:
                    if (c + y) in S_set or (y + c) in S_set:
                        can_add = False
                        break
                if 2 * c in S_set:
                    can_add = False
                if can_add:
                    S.append(c)
                    S_set.add(c)
                    improved = True
                    break
            
            if not improved:
                break
        
        return S

    def multi_start_greedy(elements, n_starts=20, rng=None):
        """
        Run greedy multiple times with different orderings, return best.
        """
        best_S = []
        
        for i in range(n_starts):
            if time.time() - start_time > DEADLINE:
                break
            
            if i == 0:
                # Original order
                S = greedy_sum_free(elements, order_key=None, rng=rng)
            elif i == 1:
                # Sort by absolute value
                S = greedy_sum_free(elements, order_key=lambda x: abs(x), rng=rng)
            elif i == 2:
                # Sort by value
                S = greedy_sum_free(elements, order_key=lambda x: x, rng=rng)
            elif i == 3:
                # Sort by -abs value
                S = greedy_sum_free(elements, order_key=lambda x: -abs(x), rng=rng)
            else:
                # Random shuffle
                shuffled = list(elements)
                rng.shuffle(shuffled)
                S = greedy_sum_free(shuffled, order_key=None, rng=rng)
            
            if len(S) > len(best_S):
                best_S = S
        
        return best_S

    def solve_instance(elements):
        """
        Solve one instance: find large sum-free subset.
        """
        if not elements:
            return []
        
        rng = random.Random(12345)
        
        # Start with multi-start greedy
        S = multi_start_greedy(elements, n_starts=30, rng=rng)
        
        # Try local search improvements
        S = local_search_improve(S, elements, max_iters=200, rng=rng)
        
        return S

    # Solve the three instances
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > DEADLINE:
            # Return what we have so far
            results.append([])
            continue
        
        elements = instance(size, seed)
        S = solve_instance(elements)
        results.append(S)
    
    return results
# EVOLVE-BLOCK-END
