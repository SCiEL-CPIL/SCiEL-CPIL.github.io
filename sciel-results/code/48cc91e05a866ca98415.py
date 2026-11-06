"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # 85s safety margin for 90s limit
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    # Precompute all bad triples: (a, b, c) where lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Store as sorted tuples for efficiency
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.add((a, b, c))
    
    # Precompute: for each x, the list of (y, z) pairs such that (x,y,z) is a bad triple
    # This makes can_add faster: just check if both y and z are in S
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        for y, z in conflict_pairs[x]:
            if y in S and z in S:
                return False
        return True
    
    def greedy_fill(S):
        """Try to add any remaining elements that can be added."""
        improved = True
        while improved:
            improved = False
            for x in elements:
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
        return S
    
    best = set()
    best_size = 0
    
    # Multiple randomized greedy passes with local search
    num_starts = 2000
    for start in range(num_starts):
        if time.time() > deadline - 15:
            break
        
        # Randomize the order
        order = elements[:]
        random.shuffle(order)
        
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        
        # Local search: try to improve
        if time.time() > deadline - 5:
            break
        
        # Try add-one moves until no more can be added
        greedy_fill(S)
        
        # Try 1-out 1-in swap moves with lightweight evaluation
        swap_attempts = 0
        max_swaps = 50
        while swap_attempts < max_swaps and time.time() < deadline - 5:
            swap_attempts += 1
            if not S:
                break
            
            # Pick a random element to remove
            to_remove = random.choice(list(S))
            S.remove(to_remove)
            
            # Try to find a replacement that improves the set
            best_new_size = len(S)
            best_S = set(S)
            
            # Try adding each possible replacement
            for x in elements:
                if x == to_remove or x in S:
                    continue
                if not can_add(x, S):
                    continue
                S.add(x)
                # Quick fill to see potential
                S_copy = set(S)
                greedy_fill(S_copy)
                if len(S_copy) > best_new_size:
                    best_new_size = len(S_copy)
                    best_S = S_copy
                S.remove(x)
            
            if best_new_size > len(S):
                S = best_S
            else:
                S.add(to_remove)  # restore
                break
        
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)
    
    # Final: try to add any remaining elements
    if time.time() < deadline - 2:
        greedy_fill(best)
    
    if not best:
        best = {1}
    
    return list(best)
# EVOLVE-BLOCK-END
