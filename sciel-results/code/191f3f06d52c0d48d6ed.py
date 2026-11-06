"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    
    # Precompute LCM and integer values
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, 25):
        L = lcm(L, i)
        
    val = {i: L // i for i in range(1, 25)}
    
    def is_subsum_free(S):
        """Check if set S is subsum-free using integer arithmetic."""
        S_list = sorted(S)
        n = len(S_list)
        for i, x in enumerate(S_list):
            target = val[x]
            # Only elements y > x can form 1/x
            # In sorted S_list, these are S_list[i+1:]
            candidates = [val[y] for y in S_list[i+1:]]
            
            # Check if target is a subset sum of candidates
            # Prune sums > target
            achievable = {0}
            for c in candidates:
                if c > target:
                    continue
                new_sums = set()
                for s in achievable:
                    ns = s + c
                    if ns <= target:
                        new_sums.add(ns)
                achievable |= new_sums
                if target in achievable:
                    return False
        return True

    def greedy_remove(S):
        """Iteratively remove subsumed elements until stable."""
        S = set(S)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Sort to ensure deterministic order, smallest first
            for x in sorted(S):
                if time.time() >= deadline:
                    break
                # Check if x is subsumed by others in S
                # Only y > x matter
                others = [y for y in S if y > x]
                if not others:
                    continue
                    
                target = val[x]
                candidates = [val[y] for y in others]
                
                achievable = {0}
                found = False
                for c in candidates:
                    if c > target:
                        continue
                    new_sums = set()
                    for s in achievable:
                        ns = s + c
                        if ns <= target:
                            new_sums.add(ns)
                    achievable |= new_sums
                    if target in achievable:
                        found = True
                        break
                
                if found:
                    S.remove(x)
                    changed = True
        return S
    
    # Start with all elements and remove subsumed ones
    S = greedy_remove(set(range(1, 25)))
    
    best = S
    
    # Try greedy construction from different orderings
    # Since the property is "x is subsumed by larger elements",
    # constructing by adding smaller elements first might be risky?
    # Or adding larger elements first makes them safe, then we check smaller ones against them.
    
    orders = [
        range(1, 25),
        range(24, 0, -1),
        # Interleaved
        [24, 1, 23, 2, 22, 3, 21, 4, 20, 5, 19, 6, 18, 7, 17, 8, 16, 9, 15, 10, 14, 11, 13, 12],
        # Evens then odds
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
        # Odds then evens
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    ]
    
    for order in orders:
        if time.time() >= deadline:
            break
        S_greedy = set()
        for x in order:
            if time.time() >= deadline:
                break
            candidate = S_greedy | {x}
            if is_subsum_free(candidate):
                S_greedy = candidate
        if len(S_greedy) > len(best):
            best = S_greedy
            
    # Local search / Repair
    # Try to add back elements from [1,24] \ best
    remaining = set(range(1, 25)) - best
    # Sort remaining by some heuristic? Maybe larger ones are easier to add?
    # Or smaller ones?
    # Let's try adding in descending order (larger first, safer)
    for x in sorted(remaining, reverse=True):
        if time.time() >= deadline:
            break
        candidate = best | {x}
        if is_subsum_free(candidate):
            best = candidate
            
    # More aggressive local search: try all single additions and swaps
    all_elements = set(range(1, 25))
    
    # Try adding each remaining element (both orders)
    for order in [sorted(remaining, reverse=True), sorted(remaining)]:
        for x in order:
            if time.time() >= deadline:
                break
            candidate = best | {x}
            if is_subsum_free(candidate):
                best = candidate
                remaining = all_elements - best
    
    # Try swaps: remove one from best, try to add two from remaining
    best_list = sorted(best)
    remaining_list = sorted(all_elements - best)
    
    for i, y in enumerate(best_list):
        if time.time() >= deadline:
            break
        temp = best - {y}
        # Try to add back two elements
        for x1 in remaining_list:
            if time.time() >= deadline:
                break
            cand1 = temp | {x1}
            if not is_subsum_free(cand1):
                continue
            for x2 in remaining_list:
                if x2 == x1 or x2 <= y:
                    continue
                if time.time() >= deadline:
                    break
                cand2 = cand1 | {x2}
                if len(cand2) > len(best) and is_subsum_free(cand2):
                    best = cand2
                    best_list = sorted(best)
                    remaining_list = sorted(all_elements - best)
                    break
            if time.time() >= deadline:
                break
    
    # Backtracking search for larger sets
    # Use the integer representation for exact checks
    def backtrack_search(current_set, available, depth, best_size):
        """Search for larger subsum-free sets using backtracking."""
        nonlocal best, deadline
        
        if len(current_set) >= best_size:
            return
        
        if time.time() >= deadline:
            return
        
        # Sort available to try larger elements first (safer)
        for x in sorted(available, reverse=True):
            if time.time() >= deadline:
                return
            
            candidate = current_set | {x}
            if is_subsum_free(candidate):
                if len(candidate) > len(best):
                    best = candidate
                    best_size = len(candidate)
                
                # Recurse with remaining available elements
                new_available = available - {x}
                backtrack_search(candidate, new_available, depth + 1, best_size)
    
    # Start backtracking from the current best
    if time.time() < deadline:
        available = all_elements - best
        backtrack_search(best, available, 0, len(best))
    
    # Final verification and return
    if not best:
        best = {24}  # Fallback: always valid
    
    return list(best)
# EVOLVE-BLOCK-END
