"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
    # Base: {66, ..., 100} - size 35
    # Min 3-sum = 201 > 200, Max 2-sum = 199 < 200
    # So no subset sums to 200.
    base_set = set(range(66, 101))
    best_size = len(base_set)
    best_set = set(base_set)
    
    # Try to improve by local search: remove elements from base, add smaller ones
    # that don't create a subset sum of 200
    
    def has_sum_200(s):
        """Check if any subset of s sums to 200. Uses DP for small sets."""
        n = len(s)
        if n < 3:
            return False
        # Use bitset DP
        reachable = 1  # bit 0 set
        for x in s:
            if x > 200:
                continue
            reachable |= (reachable << x)
            if reachable >> 200 & 1:
                return True
            # Early exit: if we can already reach 200
        return (reachable >> 200) & 1
    
    def count_conflicts(s, x):
        """Count how many subsets of s sum to 200-x (i.e., adding x creates sum 200)."""
        target = 200 - x
        if target <= 0:
            return 0
        # Count subsets of s that sum to target
        n = len(s)
        # DP counting
        dp = [0] * (target + 1)
        dp[0] = 1
        for v in s:
            if v <= target:
                for j in range(target, v - 1, -1):
                    dp[j] += dp[j - v]
        return dp[target]
    
    # Local search: try removing one element from base and adding one or more from {1,...,65}
    # We'll do iterative improvement
    
    current = set(base_set)
    current_size = len(current)
    
    # Precompute: for each candidate x in 1..65, check if adding it to current creates sum 200
    # and count conflicts
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try: remove one element r from current, add one element a from 1..65 not in current
        # Check if result is valid and same size (no improvement in size directly, but may enable future)
        # Better: try to replace 2 elements with 3, or 1 with 2
        
        # Strategy: try replacing k elements from current with k+1 elements from 1..65
        # Start with k=1: replace 1 element with 2 elements
        for r in list(current):
            if time.time() > deadline:
                break
            for a in range(1, 66):
                if a in current:
                    continue
                for b in range(a + 1, 66):
                    if b in current:
                        continue
                    new_set = current - {r} | {a, b}
                    if has_sum_200(new_set):
                        continue
                    if len(new_set) > current_size:
                        current = new_set
                        current_size = len(current)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current)
            continue
        
        # Try: replace 2 elements with 3
        elems = list(current)
        for i in range(len(elems)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(elems)):
                if time.time() > deadline:
                    break
                for a in range(1, 66):
                    if a in current:
                        continue
                    for b in range(a + 1, 66):
                        if b in current:
                            continue
                        for c in range(b + 1, 66):
                            if c in current:
                                continue
                            new_set = current - {elems[i], elems[j]} | {a, b, c}
                            if has_sum_200(new_set):
                                continue
                            if len(new_set) > current_size:
                                current = new_set
                                current_size = len(current)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current)
            continue
        
        # Try: replace 1 element with 3
        for r in list(current):
            if time.time() > deadline:
                break
            for a in range(1, 66):
                if a in current:
                    continue
                for b in range(a + 1, 66):
                    if b in current:
                        continue
                    for c in range(b + 1, 66):
                        if c in current:
                            continue
                        new_set = current - {r} | {a, b, c}
                        if has_sum_200(new_set):
                            continue
                        if len(new_set) > current_size:
                            current = new_set
                            current_size = len(current)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current)
            continue
        
        # No improvement found with these moves; try random restarts with different bases
        # or break
        break
    
    # Also try direct construction with different strategies
    # Strategy: take largest numbers, greedily add smaller ones
    
    if time.time() < deadline - 5:
        # Greedy from top
        current2 = set()
        reachable2 = {0}
        for x in range(100, 0, -1):
            new_sums = {s + x for s in reachable2 if s + x <= 200}
            if 200 not in new_sums:
                current2.add(x)
                reachable2 |= new_sums
        if len(current2) > best_size:
            best_size = len(current2)
            best_set = current2
    
    if time.time() < deadline - 5:
        # Try a mixed approach: start with large numbers, try to insert small ones
        # Use {66..100} as base, try to swap
        current3 = set(range(66, 101))
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved3 = False
            # Try replacing one large number with two small numbers
            for r in list(current3):
                if time.time() > deadline - 5:
                    break
                for a in range(1, 66):
                    if a in current3:
                        continue
                    for b in range(a + 1, 66):
                        if b in current3:
                            continue
                        new_set = current3 - {r} | {a, b}
                        if not has_sum_200(new_set):
                            current3 = new_set
                            improved3 = True
                            break
                    if improved3:
                        break
                if improved3:
                    break
            if not improved3:
                break
        if len(current3) > best_size:
            best_size = len(current3)
            best_set = current3
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
