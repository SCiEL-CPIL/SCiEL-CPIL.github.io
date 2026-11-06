"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    N = 192
    MAX_SIZE = 48
    
    # Precompute for each x in [1,N] and each pair (a,b) with a<b,
    # the set of c values (c != x, c > b) such that x | (a+b+c)
    # Actually, let's precompute more directly:
    # For each x, for each pair (a,b) with a<b, a,b != x:
    #   We need to check if there exists c in S\{x,a,b} such that (a+b+c) % x == 0
    #   i.e., c % x == (-a-b) % x
    # So for each x, we can precompute for each residue r mod x, 
    #   which pairs (a,b) have (a+b) % x == (-r) % x... 
    # This is getting complex. Let's use a simpler but efficient approach.
    
    # Key insight: For a set S and element x in S, x violates if there exist
    # a,b,c in S\{x} with x | (a+b+c).
    # For efficiency, when checking if we can add x to S:
    # 1. Check if any triple in S sums to multiple of x
    # 2. For each y in S, check if any pair (a,b) in S\{y} with x gives (x+a+b) % y == 0
    
    # Precompute: for each y in [1,N], for each pair (a,b) with a<b, a,b != y,
    # store (a+b) % y. Then when adding x, for each y in S, we need to check
    # if there exist a,b in S\{y} such that (x+a+b) % y == 0, i.e., (a+b) % y == (-x) % y.
    
    # Let's precompute for each y, a dictionary mapping residue to list of pairs
    # Actually, for speed, let's precompute for each y and each residue r in [0,y-1],
    # the count of pairs (a,b) in [1,N]\{y} with a<b and (a+b) % y == r.
    # But we need to know if the pairs are in S specifically.
    
    # Alternative approach: Use bitsets for the set S.
    # For each y, precompute for each residue r, a bitset of elements c such that c % y == r.
    # Then for a given S (as a bitset), for each y in S, we can check:
    #   Does there exist a,b in S\{y} with (a+b) % y == (-x) % y?
    #   This is equivalent to: is there a pair in the intersection of S with the 
    #   appropriate residue classes that sums to the right value mod y?
    
    # This is still complex. Let's use a hybrid approach:
    # Precompute pair sums modulo each y for all pairs, then use numpy for fast checks.
    
    import numpy as np
    
    # Precompute: for each y in [2,N] (y=1 always divides everything),
    # create a 2D array or use a different structure.
    
    # Simpler: precompute for each y, an array pair_mod[y][a][b] = (a+b) % y for a<b
    # That's N * N^2 / 2 = 192 * 192^2 / 2 ≈ 3.5M entries. Too much memory.
    
    # Let's use a different strategy: 
    # For each y, precompute for each a in [1,N]\{y}, the set of b > a, b != y, 
    # such that (a+b) % y == r for each r. Store as lists.
    
    # Actually, the most practical approach for N=192:
    # Precompute all triple sums and their divisors.
    # For each triple (a,b,c) with a<b<c, compute s = a+b+c.
    # For each divisor d of s with d in [1,N] and d not in {a,b,c}, 
    # record that d conflicts with this triple.
    # Then for a set S, d in S is valid iff no triple in S\{d} conflicts with d.
    
    # Number of triples: C(192,3) = 1,165,920. For each, factorize the sum.
    # This is feasible.
    
    # Let's precompute: conflict[d] = set of triples (a,b,c) with a<b<c, 
    # a,b,c != d, and d | (a+b+c)
    
    # But storing all triples for each d is memory heavy. Instead:
    # For each d, store a set of "pair signatures" or use a different check.
    
    # Most efficient for our use case: 
    # Precompute for each d in [2,N], for each pair (a,b) with a<b, a,b != d,
    # the residue (a+b) % d. Then when checking if d can be in S given S\{d}:
    #   We need to check if there exist a,b,c in S\{d} with (a+b+c) % d == 0
    #   i.e., c % d == (-(a+b)) % d
    #   So for each pair (a,b) in S\{d}, we check if there's c in S\{d,a,b} 
    #   with c % d == (-(a+b)) % d.
    
    # For speed, let's precompute for each d, for each residue r, the sorted list
    # of elements in [1,N]\{d} with element % d == r.
    
    residue_lists = [None] * (N + 1)
    for d in range(2, N + 1):
        residue_lists[d] = [[] for _ in range(d)]
        for v in range(1, N + 1):
            if v != d:
                residue_lists[d][v % d].append(v)
    
    def can_add(S, x, S_set):
        """Check if adding x to S keeps it valid. S is a list, x not in S_set."""
        if x in S_set:
            return False
        
        # Check 1: Does x divide any triple sum in S?
        # For each triple (a,b,c) in S, check (a+b+c) % x == 0
        if len(S) >= 3:
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # Check 2: For each y in S, does any triple in (S\{y}) + {x} sum to multiple of y?
        # New triples involving x: (x, a, b) where a,b in S\{y}, a != b
        for y in S:
            if y == 1:
                # 1 divides everything, so any triple sum works
                # Check if there's a,b in S\{1} with x != a,b
                # Triple (x,a,b): sum = x+a+b, 1 divides it always
                # So if |S\{1}| >= 2 and x not in S, we have a violation
                others = [v for v in S if v != 1]
                if len(others) >= 2:
                    return False
                continue
            
            target_res = (-x) % y
            # We need a,b in S\{y} such that (a+b) % y == target_res
            # And a,b != x (but x not in S yet, so a,b in S are fine)
            # Iterate over pairs in S\{y}
            S_without_y = [v for v in S if v != y]
            found = False
            for i in range(len(S_without_y)):
                a = S_without_y[i]
                needed_b_res = (target_res - a) % y
                # Check if there's b in S_without_y[i+1:] with b % y == needed_b_res
                for j in range(i + 1, len(S_without_y)):
                    b = S_without_y[j]
                    if b % y == needed_b_res:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
        
        return True
    
    def is_valid_set(S):
        """Check if set S is valid."""
        S_set = set(S)
        for x in S:
            others = [v for v in S if v != x]
            if len(others) < 3:
                continue
            for a, b, c in combinations(others, 3):
                if (a + b + c) % x == 0:
                    return False
        return True
    
    # Greedy construction: start with large numbers
    def greedy_construct():
        S = []
        S_set = set()
        for x in range(N, 0, -1):
            if len(S) >= MAX_SIZE:
                break
            if can_add(S, x, S_set):
                S.append(x)
                S_set.add(x)
        return S
    
    best = greedy_construct()
    best_size = len(best)
    
    # Try to add more elements
    S = list(best)
    S_set = set(S)
    for x in range(1, N + 1):
        if x in S_set:
            continue
        if can_add(S, x, S_set):
            S.append(x)
            S_set.add(x)
            if len(S) > best_size:
                best = list(S)
                best_size = len(S)
            if len(S) >= MAX_SIZE:
                break
    
    # Local search with swaps
    if time.time() < deadline - 10:
        S = list(best)
        S_set = set(S)
        
        # Try random restarts with different greedy orders
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Randomized greedy
            candidates = list(range(N, 0, -1))
            random.shuffle(candidates)
            S_r = []
            S_r_set = set()
            for x in candidates:
                if len(S_r) >= MAX_SIZE:
                    break
                if can_add(S_r, x, S_r_set):
                    S_r.append(x)
                    S_r_set.add(x)
            
            if len(S_r) > best_size:
                best = list(S_r)
                best_size = len(S_r)
                S = list(S_r)
                S_set = set(S)
        
        # Local search: try to improve current best
        S = list(best)
        S_set = set(S)
        
        for iteration in range(1000):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try adding elements
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if can_add(S, x, S_set):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    if len(S) > best_size:
                        best = list(S)
                        best_size = len(S)
                    if len(S) >= MAX_SIZE:
                        break
            
            if len(S) >= MAX_SIZE:
                break
            
            # Try swaps
            if not improved:
                for x in range(1, N + 1):
                    if x in S_set:
                        continue
                    for y in list(S):
                        if time.time() > deadline - 5:
                            break
                        S.remove(y)
                        S_set.discard(y)
                        if can_add(S, x, S_set):
                            S.append(x)
                            S_set.add(x)
                            improved = True
                            if len(S) > best_size:
                                best = list(S)
                                best_size = len(S)
                            break
                        else:
                            S.append(y)
                            S_set.add(y)
                    if improved:
                        break
            
            if not improved:
                break
    
    # Final verification
    if not is_valid_set(best):
        # Fallback: rebuild greedily
        best = greedy_construct()
    
    return best[:MAX_SIZE]
# EVOLVE-BLOCK-END
