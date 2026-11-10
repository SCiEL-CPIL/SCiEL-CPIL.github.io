"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    
    # Precompute lcm for all pairs
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Find all bad triples
    # A triple (a,b,c) with a<b<c is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = set()
    
    for a in range(1, n+1):
        if time.time() > deadline - 20:
            break
        for b in range(a+1, n+1):
            Lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if time.time() > deadline - 20:
                    break
                if lcm_table[a][c] == Lab and lcm_table[b][c] == Lab:
                    bad_triples.add((a, b, c))
    
    if time.time() > deadline - 10:
        # Fallback: powers of 2
        return [1, 2, 4, 8, 16, 32, 64, 128]

    # Convert to list for easier indexing if needed, but set is fine
    # Map each element to the set of bad triples it belongs to
    elem_to_triples = {i: set() for i in range(1, n+1)}
    for t in bad_triples:
        for x in t:
            elem_to_triples[x].add(t)

    # Helper to check if a set is valid (no bad triples)
    # This is O(|S|^3) in worst case, but we can be smarter.
    # For a candidate addition x to S, we only need to check triples involving x.
    # A triple (x, u, v) is bad if it is in bad_triples and u, v in S.
    
    def can_add(x, current_set):
        # Check all triples containing x
        for t in elem_to_triples[x]:
            # t is a tuple of 3 elements, one of which is x
            # Find the other two
            others = [y for y in t if y != x]
            if len(others) == 2:
                u, v = others
                if u in current_set and v in current_set:
                    return False
        return True

    def is_valid_set(S):
        # Check all triples in S
        # For each x in S, check if any bad triple is fully contained in S
        # This is redundant if we build incrementally, but useful for verification.
        for x in S:
            for t in elem_to_triples[x]:
                if all(y in S for y in t):
                    return False
        return True

    # Greedy Vertex Cover to get initial solution
    # We want to remove minimum elements to hit all bad triples.
    # This leaves a valid set.
    
    def greedy_cover():
        active_triples = set(bad_triples)
        removed = set()
        
        # Count frequencies
        counts = {i: 0 for i in range(1, n+1)}
        for t in active_triples:
            for x in t:
                counts[x] += 1
        
        while active_triples:
            if time.time() > deadline - 5:
                break
            
            # Find element with max count in active_triples
            # Optimization: only check elements with count > 0
            best_x = None
            max_count = 0
            for x in range(1, n+1):
                if counts[x] > max_count:
                    max_count = counts[x]
                    best_x = x
            
            if best_x is None or max_count == 0:
                break
            
            removed.add(best_x)
            
            # Update active_triples and counts
            to_remove = []
            for t in active_triples:
                if best_x in t:
                    to_remove.append(t)
            
            for t in to_remove:
                active_triples.discard(t)
                for x in t:
                    counts[x] -= 1
        
        result_set = set(range(1, n+1)) - removed
        return result_set

    best_set = greedy_cover()
    best_size = len(best_set)
    
    # Local Search / Randomized Restart
    # Try to improve by adding elements or swapping.
    
    current_set = set(best_set)
    current_size = best_size
    
    # Precompute conflicts for faster checking?
    # For each x, store the list of pairs (u,v) such that (x,u,v) is bad.
    conflicts = {i: [] for i in range(1, n+1)}
    for t in bad_triples:
        a, b, c = t
        conflicts[a].append((b,c))
        conflicts[b].append((a,c))
        conflicts[c].append((a,b))
    
    def try_add_to_set(S, x):
        """Check if x can be added to S"""
        for u, v in conflicts[x]:
            if u in S and v in S:
                return False
        return True

    # Phase 1: Greedy Add
    # Try to add any element that is valid
    while True:
        added = False
        # Sort candidates by number of conflicts? Or just iterate
        for x in range(1, n+1):
            if x not in current_set and try_add_to_set(current_set, x):
                current_set.add(x)
                current_size += 1
                added = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
        if not added:
            break
            
    # Phase 2: Randomized Local Search
    # We perform random removals and then greedy re-adds.
    
    for _ in range(1000): # Max iterations
        if time.time() > deadline - 2:
            break
        
        # Randomly remove 1-3 elements
        k = random.randint(1, 3)
        if len(current_set) < k:
            continue
        to_remove = random.sample(list(current_set), k)
        
        # Save state
        old_set = set(current_set)
        old_size = current_size
        
        for x in to_remove:
            current_set.discard(x)
        
        # Greedily add back elements
        # We want to try to add the removed ones and any others.
        # Let's shuffle all candidates not in current_set
        candidates = [x for x in range(1, n+1) if x not in current_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 2:
                break
            if try_add_to_set(current_set, x):
                current_set.add(x)
        
        new_size = len(current_set)
        
        if new_size > best_size:
            best_size = new_size
            best_set = set(current_set)
        elif new_size < old_size:
            # Revert if we got smaller than before the move
            # Actually, we should compare to best_size.
            # If new_size < best_size, we might want to revert to best_set
            # Or just keep current_set if it's still good?
            # Let's revert to the previous state if it was worse than the pre-move state
            # to avoid drifting down.
            current_set = old_set
            current_size = old_size
        else:
            # If same size, keep current (might be different configuration)
            pass
            
        # If we hit a new best, we might want to restart the "greedy add" phase
        # but the loop handles it naturally.

    result = sorted(best_set)
    if not result:
        result = [1]
        
    return result
# EVOLVE-BLOCK-END
