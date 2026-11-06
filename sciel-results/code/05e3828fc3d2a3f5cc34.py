"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import lcm

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# Precompute LCM of 1..24
_L = 1
for i in range(1, 25):
    _L = lcm(_L, i)

# Precompute integer reciprocals: R[x] = _L // x
_R = [0] * 25
for i in range(1, 25):
    _R[i] = _L // i

def _is_subsum_free(S):
    """
    Check if S is reciprocal-subsum-free: no reciprocal of a selected element 
    equals the sum of reciprocals of a subset of the OTHER selected elements.
    Uses integer arithmetic for speed.
    """
    n = len(S)
    if n == 0:
        return True
    
    # Precompute reciprocal values for elements in S
    # S is a list of integers
    recips = [_R[x] for x in S]
    
    # For each element k in S, check if 1/k can be formed by subset of others
    for i in range(n):
        k = S[i]
        target = _R[k]
        
        # Create list of others
        others_idx = [j for j in range(n) if j != i]
        others_recips = [recips[j] for j in others_idx]
        m = len(others_recips)
        
        # If sum of all others < target, no subset can equal it
        total_others = sum(others_recips)
        if total_others < target:
            continue
        
        # Check if target is a subset sum of others_recips
        # Use bitset DP for subset sum
        # Max sum is total_others. We only care if target is reachable.
        # Since target is small relative to LCM, but total_others can be large.
        # However, we can cap the DP array size at target.
        
        # dp[s] = True if sum s is reachable
        # We only need to know if target is reachable.
        # Initialize dp[0] = True
        # For each r in others_recips:
        #   new_dp = dp | (dp << r)
        #   mask to keep only bits up to target
        
        # Using a Python integer as a bitset
        # bit i is set if sum i is reachable
        dp = 1  # bit 0 set
        
        # Mask to limit size to target
        # We don't strictly need to mask if we just check bit target at the end,
        # but the integer can grow large. 
        # Max sum is total_others. 
        # Let's just let it grow and check the bit at the end.
        # But shifting large ints is slow.
        # Let's mask to target.
        mask = (1 << (target + 1)) - 1
        
        for r in others_recips:
            # If r > target, it can't contribute to forming target (since all positive)
            # Wait, if r > target, then adding it makes sum > target.
            # So we can ignore r > target?
            # Yes, because all numbers are positive.
            if r > target:
                continue
            dp = (dp | (dp << r)) & mask
            # Early exit if target bit is set?
            # We can check (dp >> target) & 1
            if (dp >> target) & 1:
                break
        
        if (dp >> target) & 1:
            # Target is reachable by a non-empty subset?
            # dp[0] is always reachable. 
            # If target == 0, then empty subset works. But target = 1/k > 0.
            # So if target bit is set, it's by a non-empty subset (since target > 0).
            return False
            
    return True

def construct():
    """
    Find the largest subset of [1, 24] that is reciprocal-subsum-free.
    """
    # Strategy: Greedy construction from largest to smallest.
    # This tends to maximize cardinality because larger elements (smaller reciprocals)
    # are less likely to form subsums with each other.
    
    S = []
    # Try adding from 24 down to 1
    for k in range(24, 0, -1):
        if _time_remaining() < 5:
            break
            
        # Tentatively add k
        new_S = S + [k]
        
        # Check if new_S is subsum-free
        # Optimization: We only need to check if adding k creates a conflict.
        # Conflict can be:
        # 1. 1/k == sum(1/x for x in T subset of S)
        # 2. For some x in S, 1/x == sum(1/y for y in T subset of (S \ {x}) U {k})
        #    This means 1/x == sum(1/y for y in T1) + 1/k  OR  1/x == sum(1/y for y in T2)
        #    Since S was already subsum-free, 1/x != sum(1/y for y in T2 subset of S\{x}).
        #    So the only new conflict is if 1/x == sum(1/y for y in T1 subset of S\{x}) + 1/k
        #    which implies 1/k = 1/x - sum(1/y for y in T1).
        #    This is harder to check incrementally.
        
        # Given N=24 is small, full check _is_subsum_free(new_S) is fast enough.
        # The DP inside _is_subsum_free is O(N * Target).
        # Target is ~ LCM/24. LCM(1..24) is ~ 5342931457063200.
        # This is too big for a bitset in Python int?
        # Python ints are arbitrary precision, but shifting a 5e15-bit integer is impossible.
        
        # RE-EVALUATE DP APPROACH.
        # LCM(1..24) is huge. Bitset DP won't work directly on the LCM scaled values.
        
        # We must go back to Fraction or a different scaling.
        # Or, use the fact that we are checking subset sum of reciprocals.
        # 1/k = sum(1/x_i).
        # Multiply by LCM(k, x_1, ..., x_m).
        # This is complex.
        
        # Alternative: Use Fraction and a set of reachable sums?
        # Number of subsets is 2^(N-1). For N=24, 2^23 ~ 8M. Too slow in Python.
        
        # Let's stick to the DFS with pruning from Inspiration 1, but optimized.
        # Or, simply verify the known good set {7, 9..24} and try to extend it?
        # The inspiration says {7, 9..24} scores 1.417.
        # Let's try to find a better set using a local search or greedy with backtracking.
        
        pass

    # Since the bitset approach failed due to LCM size, let's use the Fraction-based DFS
    # but with a tighter deadline and early exit.
    # Or, just return the known good set if we can't find better quickly.
    
    # Let's try a greedy approach with a fast check for "is 1/k a subset sum of S".
    # We can use a memoized DFS or just check the specific case.
    
    # Actually, let's just use the set {7, 9, 10, ..., 24} as a strong baseline.
    # It has size 17.
    # Can we add 8?
    # Can we add 6?
    # Can we add 5?
    
    # Let's implement a fast check using Fraction and a set of possible sums,
    # but limit the size of the set.
    
    def _can_form_target(target, elements):
        # elements: list of Fractions
        # target: Fraction
        # Returns True if target is a subset sum of elements
        # Use a set of reachable sums.
        reachable = {Fraction(0)}
        for e in elements:
            # Add e to all existing sums
            new_sums = []
            for s in reachable:
                ns = s + e
                if ns <= target: # Prune sums > target
                    new_sums.append(ns)
                    if ns == target:
                        return True
            reachable.update(new_sums)
            # Optional: prune reachable set if it gets too large?
            # If len(reachable) > 10000, switch to DFS?
            # For now, let's see.
        return target in reachable

    # Greedy construction
    S = []
    for k in range(24, 0, -1):
        if _time_remaining() < 5:
            break
            
        # Check if 1/k can be formed by subset of S
        if S:
            recips_S = [Fraction(1, x) for x in S]
            if _can_form_target(Fraction(1, k), recips_S):
                continue # Skip k if it creates a conflict where 1/k is sum of others
        
        # Check if adding k creates a conflict for existing elements
        # i.e., for any x in S, is 1/x = sum(1/y for y in T subset of (S\{x}) U {k})?
        # This is equivalent to: is 1/x - 1/k a subset sum of S\{x}?
        # Or 1/x is a subset sum of (S\{x}) U {k}?
        # Since S is subsum-free, 1/x is NOT a subset sum of S\{x}.
        # So we only need to check if 1/x - 1/k is a subset sum of S\{x} (and positive).
        
        conflict = False
        for x in S:
            target_diff = Fraction(1, x) - Fraction(1, k)
            if target_diff <= 0:
                continue
            others = [y for y in S if y != x]
            if not others:
                continue
            recips_others = [Fraction(1, y) for y in others]
            if _can_form_target(target_diff, recips_others):
                conflict = True
                break
        
        if not conflict:
            S.append(k)
            
    if not S:
        return list(range(13, 25))
        
    return sorted(S)

# EVOLVE-BLOCK-END
