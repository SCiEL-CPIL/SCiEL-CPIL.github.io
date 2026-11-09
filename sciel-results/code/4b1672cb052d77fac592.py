"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from collections import defaultdict

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    a_dim = n * m  # 16
    b_dim = m * p  # 36
    c_dim = p * n  # 36
    
    # Build the set of non-zero entries in the multiplication tensor
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some valid i,j,k
    nonzeros = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzeros.add((a, b, c))
    
    # Greedy hypergraph matching: iteratively select rank-1 terms
    # A rank-1 term is defined by (u, v, w) where u is support on A-dim, v on B-dim, w on C-dim
    # The term covers all (a,b,c) in u_support × v_support × w_support that are in remaining
    
    remaining = set(nonzeros)
    terms = []  # list of (u_dict, v_dict, w_dict)
    
    # Precompute for each (a,b) pair, the set of c values
    ab_to_cs = defaultdict(set)
    for (a, b, c) in nonzeros:
        ab_to_cs[(a, b)].add(c)
    
    # Precompute for each (a,c) pair, the set of b values
    ac_to_bs = defaultdict(set)
    for (a, b, c) in nonzeros:
        ac_to_bs[(a, c)].add(b)
    
    # Precompute for each (b,c) pair, the set of a values
    bc_to_as = defaultdict(set)
    for (a, b, c) in nonzeros:
        bc_to_as[(b, c)].add(a)
    
    # For greedy selection, we'll try different types of rank-1 terms:
    # 1. Single entry: u={a}, v={b}, w={c} - covers 1 entry
    # 2. Pair in one dim: e.g., u={a1,a2}, v={b}, w={c} - covers up to 2 entries
    # 3. Pairs in two dims: e.g., u={a1,a2}, v={b1,b2}, w={c} - covers up to 4 entries
    # 4. Pairs in all three dims: u={a1,a2}, v={b1,b2}, w={c1,c2} - covers up to 8 entries
    
    # For efficiency, we'll focus on terms with small support sizes
    # and use the structure of the tensor to guide our search
    
    while remaining:
        best_term = None
        best_count = 0
        best_u = None
        best_v = None
        best_w = None
        
        # Try to find a good rank-1 term by examining the remaining entries
        # Strategy: pick a random-ish starting point and expand
        
        # For each remaining (a,b,c), consider extending it
        # We'll try a few heuristics:
        
        # Heuristic 1: Find (a,b) with most remaining c's, then pick c
        # Heuristic 2: Find (a,c) with most remaining b's, then pick b
        # Heuristic 3: Find (b,c) with most remaining a's, then pick a
        
        # Let's try a more systematic approach:
        # For small support sizes, enumerate possible rank-1 terms
        
        # Support size 1 in each dim: just pick a remaining entry
        # This always works but is suboptimal
        
        # Let's try to find terms with support size 2 in at least one dim
        
        # For each pair of a values that share a common b or c pattern
        # This is getting complex. Let's use a simpler greedy:
        
        # Pick the (a,b) pair with the most remaining c's
        best_ab = None
        best_ab_count = 0
        for (a, b), cs in ab_to_cs.items():
            count = len(cs & {c for (_, _, c) in remaining})
            if count > best_ab_count:
                best_ab_count = count
                best_ab = (a, b)
        
        if best_ab is not None:
            a, b = best_ab
            # Now find c values for this (a,b) that are in remaining
            cs = [c for (a2, b2, c) in remaining if a2 == a and b2 == b]
            
            if len(cs) == 1:
                # Single entry
                c = cs[0]
                u = {a: 1}
                v = {b: 1}
                w = {c: 1}
                count = 1
                if count > best_count:
                    best_count = count
                    best_u = u
                    best_v = v
                    best_w = w
            elif len(cs) >= 2:
                # Try to use multiple c values
                # u={a}, v={b}, w={c1,c2,...}
                c_subset = cs[:min(4, len(cs))]  # Limit to 4 for efficiency
                u = {a: 1}
                v = {b: 1}
                w = {c: 1 for c in c_subset}
                # Count how many entries this covers
                count = sum(1 for c in c_subset if (a, b, c) in remaining)
                if count > best_count:
                    best_count = count
                    best_u = u
                    best_v = v
                    best_w = w
        
        # Also try: for each a, find b,c patterns
        for a in range(a_dim):
            # Find all (b,c) pairs for this a in remaining
            bc_pairs = [(b, c) for (a2, b, c) in remaining if a2 == a]
            if not bc_pairs:
                continue
            
            # Group by b
            b_to_cs = defaultdict(set)
            for b, c in bc_pairs:
                b_to_cs[b].add(c)
            
            # For each b with multiple c's
            for b, cs in b_to_cs.items():
                if len(cs) >= 2:
                    c_subset = list(cs)[:min(4, len(cs))]
                    u = {a: 1}
                    v = {b: 1}
                    w = {c: 1 for c in c_subset}
                    count = sum(1 for c in c_subset if (a, b, c) in remaining)
                    if count > best_count:
                        best_count = count
                        best_u = u
                        best_v = v
                        best_w = w
        
        # If we found a term, use it
        if best_u is not None:
            terms.append((best_u, best_v, best_w))
            # Remove covered entries
            u_set = set(best_u.keys())
            v_set = set(best_v.keys())
            w_set = set(best_w.keys())
            to_remove = set()
            for (a, b, c) in remaining:
                if a in u_set and b in v_set and c in w_set:
                    to_remove.add((a, b, c))
            remaining -= to_remove
        else:
            # Fallback: pick a single remaining entry
            a, b, c = next(iter(remaining))
            u = {a: 1}
            v = {b: 1}
            w = {c: 1}
            terms.append((u, v, w))
            remaining.discard((a, b, c))
    
    # Build the factor matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(a_dim)]
    V = [[0] * rank for _ in range(b_dim)]
    W = [[0] * rank for _ in range(c_dim)]
    
    for col, (u, v, w) in enumerate(terms):
        for a, val in u.items():
            U[a][col] = val
        for b, val in v.items():
            V[b][col] = val
        for c, val in w.items():
            W[c][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
