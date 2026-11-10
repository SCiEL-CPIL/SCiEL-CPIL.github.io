"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from functools import reduce
from fractions import Fraction

def _is_dissociated_fast(elems):
    """Check if all subset sums of reciprocals are distinct using integer arithmetic with early termination."""
    if not elems:
        return True
    if len(elems) == 1:
        return True
    
    n = len(elems)
    # Calculate LCM
    L = reduce(math.lcm, elems)
    
    # Convert to scaled integers: w_i = L / e_i
    weights = [L // e for e in elems]
    
    # Use a set to track all subset sums
    # Start with {0}
    sums = {0}
    
    for w in weights:
        # Generate new sums by adding w to each existing sum
        new_sums = set()
        for s in sums:
            ns = s + w
            new_sums.add(ns)
        
        # Check for collisions: if any new sum is already in sums, we have a collision
        if new_sums & sums:
            return False
        
        # Merge
        sums = sums | new_sums
        
    # All subset sums should be distinct: 2^n total
    return len(sums) == (1 << n)

def _is_dissociated(elems):
    """Check if all subset sums of reciprocals are distinct using integer arithmetic."""
    return _is_dissociated_fast(elems)

def _compute_conflict_pairs(universe):
    """
    Compute pairs (a, b) such that there exist c, d in universe (possibly a or b)
    where 1/a + 1/b = 1/c + 1/d with {a,b} != {c,d}.
    This identifies potential conflicts.
    """
    n = len(universe)
    conflicts = set()
    
    # Precompute reciprocals as Fractions for exact comparison
    recip = {e: Fraction(1, e) for e in universe}
    
    # For each pair, check if their sum equals another pair's sum
    for i in range(n):
        for j in range(i+1, n):
            a, b = universe[i], universe[j]
            s_ab = recip[a] + recip[b]
            
            for k in range(n):
                for l in range(k+1, n):
                    c, d = universe[k], universe[l]
                    if {a, b} == {c, d}:
                        continue
                    s_cd = recip[c] + recip[d]
                    if s_ab == s_cd:
                        conflicts.add((a, b))
                        conflicts.add((c, d))
    
    return conflicts

def _branch_and_bound(universe, deadline, time_limit_margin=10):
    """
    Branch and bound search for maximum dissociated subset.
    Uses conflict information to guide the search.
    """
    start_time = time.time()
    
    # Compute conflicts
    conflicts = _compute_conflict_pairs(universe)
    
    # Build adjacency: for each element, which other elements conflict with it
    adj = {e: set() for e in universe}
    for a, b in conflicts:
        adj[a].add(b)
        adj[b].add(a)
    
    best_set = []
    
    # Sort universe by degree (conflict count) ascending - try less conflicting first
    sorted_universe = sorted(universe, key=lambda x: len(adj[x]))
    
    def bb_search(current, remaining, current_size):
        nonlocal best_set
        
        if time.time() > deadline - time_limit_margin:
            return
        
        # Update best
        if current_size > len(best_set):
            best_set = list(current)
        
        if not remaining:
            return
        
        # Upper bound: current_size + len(remaining)
        if current_size + len(remaining) <= len(best_set):
            return
        
        # Try adding the first element in remaining
        e = remaining[0]
        rest = remaining[1:]
        
        # Option 1: Include e
        if _is_dissociated_fast(current + [e]):
            # Check if e conflicts with any in current (should be fine if dissociated)
            bb_search(current + [e], rest, current_size + 1)
        
        # Option 2: Exclude e
        bb_search(current, rest, current_size)
    
    # Initial call
    bb_search([], sorted_universe, 0)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 60  # 60s limit, keep safety margin
    
    universe = list(range(1, 23))
    
    # Greedy function with a given order
    def greedy(order):
        cur = []
        for e in order:
            test_set = cur + [e]
            if _is_dissociated_fast(test_set):
                cur = test_set
        return cur
    
    best_set = []
    
    # 1. Try deterministic orders
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        list(universe)
    ]
    
    for order in orders:
        if time.time() > deadline - 15:
            break
        result = greedy(order)
        if len(result) > len(best_set):
            best_set = result
    
    # 2. Local search: try to add any missing element
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        remaining = [e for e in universe if e not in best_set]
        random.shuffle(remaining)
        for e in remaining:
            test_set = best_set + [e]
            if _is_dissociated_fast(test_set):
                best_set = test_set
                improved = True
                break
    
    # 3. If we have time, try branch-and-bound from a good starting point
    if time.time() < deadline - 20:
        # Use the current best as a starting point for B&B
        # Or try B&B from scratch with time limit
        bb_result = _branch_and_bound(universe, deadline, time_limit_margin=10)
        if len(bb_result) > len(best_set):
            best_set = bb_result
    
    # 4. Random restarts with greedy + local search
    while time.time() < deadline - 15:
        random.shuffle(universe)
        result = greedy(universe)
        if len(result) > len(best_set):
            best_set = result
            # Try to improve this new best
            improved = True
            while improved and time.time() < deadline - 15:
                improved = False
                remaining = [e for e in range(1, 23) if e not in best_set]
                random.shuffle(remaining)
                for e in remaining:
                    if _is_dissociated_fast(best_set + [e]):
                        best_set.append(e)
                        improved = True
                        break
        
        # If we found a very large set, stop early
        if len(best_set) >= 18:
            break
    
    # Final verification
    if best_set and _is_dissociated_fast(best_set):
        return sorted(best_set)
    
    # Fallback to a known valid set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
