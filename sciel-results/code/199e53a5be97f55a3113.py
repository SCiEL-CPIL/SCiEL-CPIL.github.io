"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s kill, 5s safety margin

    # Precompute all forbidden triples: (a,b,c) distinct, a<b<c, 
    # where a+b, a+c, b+c are all in 1..128
    # A set is valid iff it contains no such triple
    # We want maximum subset of {1..128} with no forbidden triple
    
    # Build conflict structure: for each pair (a,b), the "sum" a+b
    # A triple (a,b,c) is forbidden iff a+b, a+c, b+c are all in the set
    # So for a given set S, we check: for all distinct a,b,c in S,
    # not all of a+b, a+c, b+c are in S
    
    # Key insight: if we precompute which triples are "dangerous" (all three pair-sums in range 1..128),
    # then we need a subset with no dangerous triple.
    
    # Precompute dangerous triples
    N = 128
    all_vals = list(range(1, N+1))
    
    # A triple (a,b,c) with a<b<c is dangerous if a+b <= N, a+c <= N, b+c <= N
    # Since a<b<c, b+c is the largest sum, so we need b+c <= N
    # Also a+b < b+c and a+c < b+c, so only need b+c <= N
    
    # For branch and bound, use bitset representation
    # vertices = 1..128, indexed 0..127
    
    # Precompute for each vertex v, the set of other vertices u such that 
    # there exists a third vertex w where {v,u,w} is a dangerous triple
    # Actually, let's think differently.
    
    # A set S is valid iff for every triple (a,b,c) of distinct elements in S,
    # at least one of a+b, a+c, b+c is NOT in S.
    
    # Equivalently: S is valid iff there do NOT exist distinct a,b,c in S 
    # such that a+b in S AND a+c in S AND b+c in S.
    
    # Let's precompute: for each pair (a,b) with a<b, define sum_ab = a+b
    # Then a triple (a,b,c) is forbidden in S iff:
    #   c in S, a+b in S, a+c in S, b+c in S
    #   AND a,b,c are all distinct
    #   AND a+b, a+c, b+c are all in [1,128] (i.e., <= 128)
    
    # For efficiency in branch-and-bound, let's use a different approach:
    # Greedy + local search with fast validity checking.
    
    # Fast validity check for adding x to set S:
    # Adding x creates a forbidden triple iff there exist distinct a,b in S (a≠b, a≠x, b≠x)
    # such that a+b in S∪{x}, a+x in S∪{x}, b+x in S∪{x}
    # Since x is new, a+x in S∪{x} means a+x in S (since a+x ≠ x unless a=0 which isn't in range)
    # Similarly b+x in S
    # And a+b in S∪{x} means a+b in S or a+b == x
    
    # So: adding x is invalid iff ∃ distinct a,b in S, a≠b, such that:
    #   (a+b in S or a+b == x) AND (a+x in S) AND (b+x in S)
    #   AND a, b, x are all distinct (a≠x, b≠x already since a,b in S and x not in S)
    #   AND a≠b
    
    # Also need a+x <= 128, b+x <= 128, a+b <= 128 for the sums to be in range
    
    def can_add(S_set, x):
        """Check if x can be added to S_set without creating a forbidden triple."""
        if x in S_set:
            return False
        # For each pair (a,b) in S_set, a<b, check if adding x creates a forbidden config
        # Need: a+x in S_set, b+x in S_set, and (a+b in S_set or a+b == x)
        # Also a+x <= N, b+x <= N, a+b <= N
        S_list = sorted(S_set)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            ax = a + x
            if ax > N:
                continue
            if ax not in S_set:
                continue
            for j in range(i+1, n):
                b = S_list[j]
                bx = b + x
                if bx > N:
                    continue
                if bx not in S_set:
                    continue
                ab = a + b
                if ab > N:
                    continue
                if ab in S_set or ab == x:
                    return False
        return True

    # Better: precompute for speed. Use a set-based approach.
    # For a given x, the pairs (a,b) that would be problematic are those where:
    # a+x in S, b+x in S, and (a+b in S or a+b==x)
    # i.e., a in S-x (shifted), b in S-x (shifted), and a+b in S or a+b==x
    
    def can_add_fast(S_set, S_list, x):
        """Faster check."""
        # a+x must be in S, so a must be in {s-x for s in S if s-x >= 1}
        # Similarly for b
        # Let's iterate over possible a values
        for a in S_list:
            if a == x:
                continue
            ax = a + x
            if ax > N or ax not in S_set:
                continue
            # Now need b in S, b != a, b != x, b+x in S, and (a+b in S or a+b==x)
            for b in S_list:
                if b <= a:
                    continue
                if b == x:
                    continue
                bx = b + x
                if bx > N or bx not in S_set:
                    continue
                ab = a + b
                if ab > N:
                    continue
                if ab in S_set or ab == x:
                    return False
        return True

    # Start with a good base set
    # {65..128} is sum-free (min pair sum = 131 > 128)
    # Can add small numbers
    base_A = set(range(65, 129))  # 64 elements
    # Try to add elements greedily
    
    best_set = set(base_A)
    best_size = len(best_set)
    
    # Greedy: try adding each element from 1 to 64
    for x in range(1, 65):
        if time.time() > deadline - 10:
            break
        if can_add_fast(best_set, sorted(best_set), x):
            best_set.add(x)
    
    best_size = len(best_set)
    
    # Try multiple random restarts with different orderings
    rng = random.Random(12345)
    
    for restart in range(200):
        if time.time() > deadline - 15:
            break
        
        # Start with base_A plus some random small elements
        S = set(base_A)
        S_list = sorted(S)
        
        # Try adding elements in random order
        candidates = list(range(1, 65))
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 15:
                break
            if x in S:
                continue
            if can_add_fast(S, S_list, x):
                S.add(x)
                S_list.append(x)
                S_list.sort()
        
        # Local search: try to swap elements
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
        
        # Try to improve by removing and re-adding
        if len(S) >= best_size and time.time() < deadline - 20:
            S_list = sorted(S)
            not_in_S = [x for x in range(1, N+1) if x not in S]
            
            improved = True
            while improved and time.time() < deadline - 20:
                improved = False
                for x in S_list:
                    if time.time() > deadline - 20:
                        break
                    S_temp = S - {x}
                    S_temp_list = [v for v in S_list if v != x]
                    
                    for y in not_in_S:
                        if time.time() > deadline - 20:
                            break
                        if y in S_temp:
                            continue
                        if can_add_fast(S_temp, S_temp_list, y):
                            S_new = S_temp | {y}
                            S_new_list = sorted(S_new)
                            
                            # Try to add more
                            for z in not_in_S:
                                if z == y or z in S_new:
                                    continue
                                if can_add_fast(S_new, S_new_list, z):
                                    S_new.add(z)
                                    S_new_list.append(z)
                                    S_new_list.sort()
                            
                            if len(S_new) > best_size:
                                best_set = S_new
                                best_size = len(S_new)
                                S = S_new
                                S_list = S_new_list
                                not_in_S = [x for x in range(1, N+1) if x not in S]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
    
    # Final polish: try all single-element removals + additions
    if time.time() < deadline - 5:
        S_list = sorted(best_set)
        not_in_S = [x for x in range(1, N+1) if x not in best_set]
        
        for x in S_list:
            if time.time() > deadline - 5:
                break
            S_temp = best_set - {x}
            S_temp_list = [v for v in S_list if v != x]
            
            for y in not_in_S:
                if time.time() > deadline - 5:
                    break
                if y in S_temp:
                    continue
                if can_add_fast(S_temp, S_temp_list, y):
                    S_new = S_temp | {y}
                    S_new_list = sorted(S_new)
                    
                    # Try greedy additions
                    for z in not_in_S:
                        if z == y or z in S_new:
                            continue
                        if can_add_fast(S_new, S_new_list, z):
                            S_new.add(z)
                            S_new_list.append(z)
                            S_new_list.sort()
                    
                    if len(S_new) > best_size:
                        best_set = S_new
                        best_size = len(S_new)
    
    result = [int(x) for x in sorted(best_set)]
    return result
# EVOLVE-BLOCK-END
