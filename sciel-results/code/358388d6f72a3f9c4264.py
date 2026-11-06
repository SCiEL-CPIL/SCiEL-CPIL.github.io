"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 70  # Safety margin under 90s kill
    N = 128
    elements = list(range(1, N + 1))
    elem_idx = {x: i for i, x in enumerate(elements)}  # 1-based to 0-based
    
    # Precompute LCM for all pairs
    lcm = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            lcm[i][j] = i * j // gcd(i, j)
            lcm[j][i] = lcm[i][j]
            
    # Find all bad triples and build conflict masks
    # conflict_mask[x] is a bitmask of elements y such that (x,y,z) forms a bad triple for some z
    # More precisely: for each element x, we store a set of (y,z) pairs where {x,y,z} is a bad triple
    # For bitmask efficiency: conflict_pairs[x] = set of (y,z) sorted tuples
    
    # Build: for each x, which pairs (y,z) with y<z from elements create bad triple with x
    # We'll use: bad_triples_by_element[x] = list of (y,z) where sorted(x,y,z) in bad_triples
    bad_triples = set()
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = lcm[a][b]
            for c in range(b+1, N+1):
                if lcm[a][c] == lab and lcm[b][c] == lab:
                    bad_triples.add((a, b, c))
    
    # For each element, precompute which pairs of other elements would form a bad triple with it
    # conflict_with[x] = set of frozenset({y,z}) such that {x,y,z} is bad
    # Better: for bitmask, we need to check if adding x to S creates a conflict
    # i.e., are there y,z in S such that {x,y,z} is a bad triple?
    
    # Precompute: for each x, a list of (y, z) pairs (y<z, both != x) where {x,y,z} is bad
    # Then can_add(x, S) checks if any (y,z) in S
    conflicts_with = [[] for _ in range(N+1)]
    for a, b, c in bad_triples:
        # a < b < c
        conflicts_with[a].append((b, c))
        conflicts_with[b].append((a, c))
        conflicts_with[c].append((a, b))
    
    # For fast lookup, convert to sets of frozensets or use bitmask approach
    # Let's use: for each x, a set of (min(y,z), max(y,z)) pairs
    conflict_pairs_set = [set() for _ in range(N+1)]
    for x in range(1, N+1):
        for y, z in conflicts_with[x]:
            conflict_pairs_set[x].add((min(y,z), max(y,z)))
    
    # Precompute degrees
    degrees = [0] * (N + 1)
    for a, b, c in bad_triples:
        degrees[a] += 1
        degrees[b] += 1
        degrees[c] += 1

    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i+1, n):
                y, z = S_list[i], S_list[j]
                if y > z:
                    y, z = z, y
                if (y, z) in conflict_pairs_set[x]:
                    return False
        return True

    def is_valid(S):
        """Verify that S contains no bad triple."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if a > b: a, b = b, a
                    if b > c: b, c = c, b
                    if a > b: a, b = b, a
                    if (a, b, c) in bad_triples:
                        return False
        return True

    best_set = set()
    best_size = 0
    
    def greedy_fill(S, order):
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)
        return S

    def polish(S):
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for x in elements:
                if x not in S and can_add(x, S):
                    S.add(x)
                    changed = True
        return S

    # Simulated annealing local search
    def simulated_annealing(S, max_iter=2000):
        """Try to improve S by removing one element and adding another."""
        S = set(S)
        current_size = len(S)
        best_local = set(S)
        best_local_size = current_size
        
        start_temp = 1.0
        end_temp = 0.01
        cooling = (end_temp / start_temp) ** (1.0 / max(max_iter, 1))
        temp = start_temp
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            if current_size == 0:
                break
                
            # Randomly pick an element to remove
            to_remove = random.choice(list(S))
            S.remove(to_remove)
            
            # Try to add a different element
            candidates = [x for x in elements if x not in S]
            if not candidates:
                S.add(to_remove)
                continue
                
            # Try a few random candidates
            tried = random.sample(candidates, min(5, len(candidates)))
            added = None
            for cand in tried:
                if can_add(cand, S):
                    S.add(cand)
                    added = cand
                    break
            
            if added is None:
                # Couldn't add anything, put back
                S.add(to_remove)
            else:
                new_size = len(S)
                delta = new_size - current_size
                if delta > 0 or random.random() < temp ** (1.0 / max(1, abs(delta))):
                    current_size = new_size
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local = set(S)
                else:
                    # Reject: revert
                    S.discard(added)
                    S.add(to_remove)
            
            temp *= cooling
        
        return best_local

    # Run multiple strategies
    strategies = []
    
    # 1. Natural order
    strategies.append(elements[:])
    
    # 2. Reverse natural order
    strategies.append(elements[::-1])
    
    # 3. Sorted by degree (ascending)
    strategies.append(sorted(elements, key=lambda x: degrees[x]))
    
    # 4. Sorted by degree (descending)
    strategies.append(sorted(elements, key=lambda x: -degrees[x]))
    
    # 5-10. Random shuffles
    for _ in range(5):
        random.shuffle(elements)
        strategies.append(elements[:])
    
    for order in strategies:
        if time.time() > deadline - 15:
            break
        S = set()
        greedy_fill(S, order)
        S = polish(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Try simulated annealing on this result
        if time.time() < deadline - 20:
            S_sa = simulated_annealing(set(S), max_iter=1500)
            if len(S_sa) > best_size:
                best_size = len(S_sa)
                best_set = set(S_sa)
    
    # Final polish on best set
    if time.time() < deadline - 5:
        best_set = polish(best_set)
    
    # Final verification
    if not is_valid(best_set):
        best_set = {1}
        
    return list(best_set)
# EVOLVE-BLOCK-END
