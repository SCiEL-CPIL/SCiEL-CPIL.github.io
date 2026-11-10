"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
from collections import Counter, defaultdict

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
    
    # Find all bad triples: (a,b,c) with a<b<c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = set()
    for a in range(1, n+1):
        if time.time() > deadline - 10:
            break
        for b in range(a+1, n+1):
            Lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if lcm_table[a][c] == Lab and lcm_table[b][c] == Lab:
                    bad_triples.add((a,b,c))
    
    if time.time() > deadline - 8:
        # Fallback: return a simple valid set
        return [1, 2, 4, 8, 16, 32, 64, 128]
    
    if not bad_triples:
        return list(range(1, n+1))
    
    # Build adjacency: for each element, the set of bad triples it participates in
    elem_triples = defaultdict(set)
    for t in bad_triples:
        for x in t:
            elem_triples[x].add(t)
    
    # Phase 1: Greedy removal to get a valid set
    removed = set()
    uncovered = set(bad_triples)
    
    iteration = 0
    while uncovered and iteration < 300:
        if time.time() > deadline - 5:
            break
        iteration += 1
        # Find element with most coverage in uncovered triples
        best_x = None
        best_size = 0
        for x in elem_triples:
            if x in removed:
                continue
            size = len(elem_triples[x] & uncovered)
            if size > best_size:
                best_size = size
                best_x = x
        if best_x is None or best_size == 0:
            break
        removed.add(best_x)
        uncovered -= elem_triples[best_x]
    
    result = [x for x in range(1, n+1) if x not in removed]
    result_set = set(result)
    
    if not result:
        result = [1]
        result_set = {1}
    
    # Phase 2: Try to add back removed elements
    for x in sorted(removed, key=lambda v: -len(elem_triples[v])):
        if time.time() > deadline - 3:
            break
        can_add = True
        for t in bad_triples:
            if x in t:
                others = [y for y in t if y != x]
                if len(others) == 2 and others[0] in result_set and others[1] in result_set:
                    can_add = False
                    break
        if can_add:
            result.append(x)
            result_set.add(x)
            removed.discard(x)
    
    result.sort()
    
    if not result:
        result = [1]
    
    # Phase 3: Local search with swaps - try swapping one element out for one in
    # to see if we can increase the set size
    improved = True
    swap_rounds = 0
    while improved and swap_rounds < 20:
        if time.time() > deadline - 2:
            break
        improved = False
        swap_rounds += 1
        result_set = set(result)
        
        # Try removing each element and adding a different one
        for x in sorted(result_set):
            if time.time() > deadline - 2:
                break
            # Temporarily remove x
            result_set.discard(x)
            
            # Try adding each removed element
            best_add = None
            for y in removed:
                can_add = True
                for t in bad_triples:
                    if y in t:
                        others = [z for z in t if z != y]
                        if len(others) == 2 and others[0] in result_set and others[1] in result_set:
                            can_add = False
                            break
                if can_add:
                    best_add = y
                    break
            
            if best_add is not None:
                # Perform the swap
                result_set.add(best_add)
                removed.discard(best_add)
                removed.add(x)
                improved = True
            
            # Restore x if we didn't add best_add
            if best_add is None:
                result_set.add(x)
        
        result = sorted(result_set)
    
    # Phase 4: More aggressive local search - try multiple removals and additions
    if time.time() < deadline - 2:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            
            # Try removing up to 2 elements and adding up to 2 others
            result_set = set(result)
            best_size = len(result)
            best_result = list(result)
            
            # Try single swaps more thoroughly
            for x in sorted(result_set):
                if time.time() > deadline - 2:
                    break
                temp_set = result_set - {x}
                
                candidates = []
                for y in removed:
                    can_add = True
                    for t in bad_triples:
                        if y in t:
                            others = [z for z in t if z != y]
                            if len(others) == 2 and others[0] in temp_set and others[1] in temp_set:
                                can_add = False
                                break
                    if can_add:
                        candidates.append(y)
                
                if candidates:
                    # Try each candidate
                    for y in candidates:
                        new_set = temp_set | {y}
                        if len(new_set) > best_size:
                            best_size = len(new_set)
                            best_result = sorted(new_set)
            
            if best_size > len(result):
                result = best_result
                result_set = set(result)
                # Update removed set
                old_removed = set(removed)
                new_removed = old_removed - set(result)
                removed = new_removed
    
    result.sort()
    
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
