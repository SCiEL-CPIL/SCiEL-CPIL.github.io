"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 128
    
    # Precompute lcm for all pairs
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Find all bad triples
    # A triple (a,b,c) with a<b<c is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    # For efficiency, we can iterate L from 1 to n, and check divisors.
    # But direct iteration over all triples is also fast enough for n=128.
    # Number of triples is ~350k.
    # 350k * 3 LCM lookups is ~1M ops. Very fast.
    
    # Let's stick to direct iteration for simplicity and correctness, 
    # but optimize by breaking early if possible? No, LCM is fast.
    
    for a in range(1, n+1):
        if time.time() > deadline - 10:
            break
        for b in range(a+1, n+1):
            Lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if time.time() > deadline - 10:
                    break
                # Check if all three are equal
                if lcm_table[a][c] == Lab and lcm_table[b][c] == Lab:
                    bad_triples.append((a, b, c))
    
    if time.time() > deadline - 5:
        # Fallback: powers of 2
        return [1, 2, 4, 8, 16, 32, 64, 128]

    if not bad_triples:
        return list(range(1, n+1))

    # Convert to set for faster lookup if needed, but we need to manage removal.
    # We will use a greedy vertex cover approach.
    
    # Map each element to the set of indices of bad triples it belongs to
    # Or simpler: maintain a list of active triples.
    
    # To make removal efficient, let's count frequencies.
    # Since the number of bad triples is likely small (let's say < 10,000),
    # we can just filter the list.
    
    active_triples = bad_triples
    
    removed = set()
    
    # Greedy: repeatedly pick the element that appears in the most active triples
    # and remove it.
    
    # Optimization: Pre-compute for each element, the list of triples it's in.
    elem_to_triples = {i: [] for i in range(1, n+1)}
    for idx, t in enumerate(bad_triples):
        for x in t:
            elem_to_triples[x].append(idx)
    
    # We need to track which triples are still active.
    # active = [True] * len(bad_triples)
    # But filtering is easier.
    
    # Let's use a set of triple tuples for active triples.
    active_set = set(bad_triples)
    
    # Count frequencies in active_set
    # This is O(|active_set| * 3)
    
    # To avoid recomputing counts from scratch every time (which is O(N * T)),
    # we can maintain counts.
    
    counts = {}
    for t in active_set:
        for x in t:
            counts[x] = counts.get(x, 0) + 1
    
    # Main loop
    while active_set:
        if time.time() > deadline - 2:
            break
        
        # Find element with max count
        best_x = None
        max_count = 0
        for x, c in counts.items():
            if c > max_count:
                max_count = c
                best_x = x
        
        if best_x is None or max_count == 0:
            break
            
        # Remove best_x from the set
        removed.add(best_x)
        
        # Update active_set and counts
        # Find all triples containing best_x
        to_remove = []
        for t in active_set:
            if best_x in t:
                to_remove.append(t)
        
        for t in to_remove:
            active_set.remove(t)
            for x in t:
                counts[x] -= 1
                if counts[x] == 0:
                    # Optional: remove from counts dict to keep it small
                    pass # leaving 0 is fine, just check > 0
    
    # Current solution is all elements NOT in removed
    result_set = set(range(1, n+1)) - removed
    
    # Local Search: Try to add back removed elements
    # For each x in removed, check if it can be added.
    # x can be added if there is no triple {x, u, v} in bad_triples 
    # such that u and v are both in result_set.
    
    # Precompute for each x, the list of pairs (u,v) such that {x,u,v} is bad.
    # We can derive this from bad_triples.
    conflicts = {i: set() for i in range(1, n+1)}
    for t in bad_triples:
        a, b, c = t
        # For a, the conflicting pair is (b,c)
        conflicts[a].add((b,c))
        conflicts[b].add((a,c))
        conflicts[c].add((a,b))
    
    # Sort removed elements by some heuristic, e.g., number of conflicts with current set?
    # Or just iterate.
    # To maximize gain, we should try to add elements that have fewest conflicts.
    
    # Let's try adding them back.
    # We iterate through removed elements.
    # If we add one, it might block others, so order matters.
    # Greedy: pick the one that creates the fewest new conflicts?
    # Or simply: pick the one that allows the most additions?
    
    # Simple approach: try to add all that can be added.
    # Repeat until no more can be added.
    
    changed = True
    while changed:
        changed = False
        if time.time() > deadline - 1:
            break
            
        candidates = []
        for x in removed:
            if x in result_set:
                continue
            # Check if x can be added
            can_add = True
            for u, v in conflicts[x]:
                if u in result_set and v in result_set:
                    can_add = False
                    break
            if can_add:
                candidates.append(x)
        
        if not candidates:
            break
        
        # Add all candidates that are still valid?
        # Adding one might invalidate another.
        # Let's add them one by one, prioritizing those with fewer conflicts?
        # Or just add the first one found?
        # Better: Add the one that is in the most "potential" triples?
        # Actually, if we can add multiple disjoint ones, we should.
        
        # Let's just add the first valid one, then restart the check.
        # This is safe but might be slow if many.
        # Given n=128, it's fine.
        
        # Pick candidate with min number of conflicts in result_set?
        # conflicts[x] contains pairs. We want x such that few pairs are in result_set.
        # Actually, we already checked that NO pairs are in result_set.
        # So any candidate is valid.
        
        # Let's pick the one with the smallest index to be deterministic?
        # Or largest?
        # Let's pick the one that is in the most bad triples originally?
        # It doesn't matter much.
        
        x_to_add = candidates[0]
        result_set.add(x_to_add)
        removed.discard(x_to_add)
        changed = True

    result = sorted(result_set)
    
    if not result:
        result = [1]
        
    return result
# EVOLVE-BLOCK-END
