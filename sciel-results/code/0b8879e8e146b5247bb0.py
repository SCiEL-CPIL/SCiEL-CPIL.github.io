from fractions import Fraction
from itertools import combinations
import time

def _subset_sums(elements):
    """Returns a set of all possible subset sums of the given list of Fractions."""
    sums = {Fraction(0)}
    for e in elements:
        new_sums = set()
        for s in sums:
            new_sums.add(s + e)
        sums.update(new_sums)
    return sums

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # We want to select a subset S of [1, 24]
    # Condition: For all i in S, 1/i is not a subset sum of {1/j for j in S, j > i}
    
    # Greedy construction from largest to smallest
    # Start with all numbers, and remove those that cause conflicts?
    # Or build up?
    
    # Let's try building up from largest to smallest.
    # Candidate pool: [24, 23, ..., 1]
    # Current set: []
    # For each candidate c in descending order:
    #   Check if adding c to current set S is valid.
    #   Validity check:
    #     1. For c: 1/c must not be a subset sum of {1/j for j in S, j > c}.
    #        Since we are adding in descending order, all j in S are > c.
    #        So we just need to check if 1/c is in subset_sums({1/j for j in S}).
    #     2. For existing j in S: Does adding c create a conflict for j?
    #        Conflict for j exists if 1/j is a subset sum of {1/k for k in S U {c}, k > j}.
    #        Previously, 1/j was not a sum of {1/k for k in S, k > j}.
    #        The new set is {1/k for k in S, k > j} U {1/c if c > j}.
    #        Since we add c < all existing j (because we go descending), c < j.
    #        So c is NOT > j.
    #        Therefore, {1/k for k in S U {c}, k > j} is exactly {1/k for k in S, k > j}.
    #        So existing elements are NOT affected by adding a smaller element c.
        
    # This is a key insight!
    # If we build the set S by adding elements in strictly decreasing order,
    # adding a new element c (which is smaller than all existing elements)
    # does NOT introduce any new terms to the subset sums used to check validity
    # of the existing elements.
    # It only introduces a new check for c itself:
    # Is 1/c a subset sum of the reciprocals of the existing (larger) elements?
    
    # So the algorithm is simply:
    # S = []
    # for c in [24, 23, ..., 1]:
    #   if 1/c is not in subset_sums([1/j for j in S]):
    #     S.append(c)
    # return S
    
    # This greedy algorithm is deterministic and fast.
    # Let's test this logic.
    
    S = []
    reciprocals_in_S = [] # List of Fractions 1/j for j in S
    
    for c in range(24, 0, -1):
        if time.time() > deadline:
            break
            
        target = Fraction(1, c)
        # Check if target is a subset sum of reciprocals_in_S
        # Since |S| <= 24, subset sums are manageable.
        # But we can optimize by caching or using a set.
        
        # To check if target is in subset sums, we can generate all subset sums.
        # However, doing this from scratch for each c is O(2^|S| * |S|).
        # With |S| up to 24, 2^24 is 16M, which might be slow in Python if repeated 24 times.
        # But we can maintain the set of subset sums incrementally?
        # No, the set of subset sums grows.
        
        # Let's just compute subset sums of reciprocals_in_S.
        # Optimization: If sum(reciprocals_in_S) < target, it's impossible.
        # If target is very small, it might be possible.
        
        # Let's try a simple check first.
        # If the greedy approach yields a large set, great.
        
        # To speed up, we can keep a set of all subset sums of the current S.
        # Let SS be the set of subset sums of reciprocals_in_S.
        # Initially SS = {0}.
        # When adding c, we check if 1/c in SS.
        # If not, we add c to S, and update SS.
        # New SS = SS U {s + 1/c for s in SS}.
        
    # Let's re-implement with this optimization.
    
    S = []
    subset_sums = {Fraction(0)}
    
    for c in range(24, 0, -1):
        if time.time() > deadline:
            break
            
        target = Fraction(1, c)
        
        if target not in subset_sums:
            S.append(c)
            # Update subset sums
            # new_sums = {s + target for s in subset_sums}
            # subset_sums.update(new_sums)
            # This doubles the size of the set each time.
            # Max size 2^24. This might be too slow/memory heavy.
            
            # Alternative: Just check validity against the current S without maintaining full subset sums?
            # Or use a heuristic to prune?
            
            # Let's try to keep the subset sums set but check size.
            # If len(subset_sums) gets too large, we might need a different check.
            # But for N=24, 2^24 is ~16 million.
            # A set of 16 million Fractions might be slow or memory intensive.
            
            # Let's try a different check:
            # Is target a subset sum of reciprocals_in_S?
            # This is a subset sum problem.
            # We can solve it using DP or backtracking.
            # Since the numbers are Fractions, we can scale them to integers?
            # LCM of 1..24 is large.
            
            # Let's stick to the incremental subset sums set but monitor performance.
            # If it's too slow, we fall back.
            
            new_sums = set()
            for s in subset_sums:
                new_sums.add(s + target)
            subset_sums.update(new_sums)
            
            # If the set grows too large, we might hit performance issues.
            # Let's add a check.
            if len(subset_sums) > 1000000:
                # Fallback: Just accept the current set? Or use a slower check?
                # For now, let's assume it's fast enough for 24 elements.
                pass
                
    return S
