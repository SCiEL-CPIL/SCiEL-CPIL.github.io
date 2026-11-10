"""
Strategy: Integer Linear Programming (ILP) via scipy.optimize.milp.
Problem: Maximize sum(x_i) subject to x_i + x_j + x_k <= 2 for all bad triples (i,j,k).
This is a maximum independent set problem on a 3-uniform hypergraph.
"""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify all bad triples (a, b, c) with a < b < c
    # such that lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a - 1, b - 1, c - 1)) # 0-indexed for matrix

    # If no bad triples, return all
    if not bad_triples:
        return numbers

    # 3. Set up ILP
    # Variables x_1 ... x_128, binary.
    # Objective: Maximize sum(x_i) -> Minimize -sum(x_i)
    c = -np.ones(n)
    
    # Constraints: For each bad triple (i, j, k), x_i + x_j + x_k <= 2
    # A matrix of size (num_triples, n)
    # A is sparse
    A = lil_matrix((len(bad_triples), n), dtype=np.int8)
    for idx, (i, j, k) in enumerate(bad_triples):
        A[idx, i] = 1
        A[idx, j] = 1
        A[idx, k] = 1
    A = A.tocsr()
    
    # Constraint: A @ x <= 2 * ones(len(bad_triples))
    lb = np.zeros(len(bad_triples))
    ub = np.full(len(bad_triples), 2)
    
    constraints = LinearConstraint(A, lb, ub)
    integrality = np.ones(n) # All variables are integer (binary)
    bounds = Bounds(0, 1)
    
    # Solve
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0 and res.x is not None:
            # res.x contains the optimal solution
            # Due to floating point, round to nearest integer
            x_opt = np.round(res.x).astype(int)
            # Ensure binary (0 or 1)
            x_opt = (x_opt > 0.5).astype(int)
            
            # Construct result set
            selected = [i + 1 for i in range(n) if x_opt[i] == 1]
            
            # Verify validity (just in case)
            if selected:
                sel_set = set(selected)
                valid = True
                # Optimization: only check triples where all 3 are in sel_set
                # We can iterate through bad_triples again
                for i, j, k in bad_triples:
                    if i in sel_set and j in sel_set and k in sel_set:
                        valid = False
                        break
                if valid:
                    return sorted(selected)
                else:
                    # Fallback if verification fails (should not happen with exact solver)
                    # Use greedy fallback
                    pass
            else:
                return [1] # Fallback non-empty
        else:
            # Solver failed or infeasible? Fallback to greedy
            pass
    except Exception:
        pass

    # Fallback: Simple Greedy if ILP fails or times out (though ILP is usually fast for this size)
    # Re-use logic from previous steps if needed, but ILP should be robust.
    # Let's implement a quick greedy fallback to ensure non-empty return.
    
    conflict = {i: set() for i in range(1, n + 1)}
    # Rebuild conflict map for greedy if needed
    # Actually, let's just use the bad_triples list
    # Greedy: Add elements if they don't form a bad triple with existing ones
    current_set = set()
    # Order by frequency in bad triples (least constrained first)
    freq = [0]*(n+1)
    for i, j, k in bad_triples:
        freq[i+1] += 1
        freq[j+1] += 1
        freq[k+1] += 1
    order = sorted(numbers, key=lambda x: freq[x])
    
    for x in order:
        valid = True
        # Check if adding x creates a bad triple
        # We need to check pairs (u, v) in current_set such that (u, v, x) is a bad triple
        # This is O(|current_set|^2) per x, which is 128^3 ~ 2M ops. Acceptable for fallback.
        # But we can be smarter: precompute for each x, the set of pairs (u,v) that conflict.
        pass
        
    # Better fallback:
    # We already have bad_triples.
    # Let's just do a simple random restart greedy if ILP fails.
    import random
    best_size = 0
    best_set = []
    
    for _ in range(20):
        if time.time() > deadline - 2: break
        current_set = set()
        # Random order
        rand_order = numbers[:]
        random.shuffle(rand_order)
        for x in rand_order:
            # Check validity
            ok = True
            # Iterate over all bad triples involving x?
            # Too slow.
            # Let's use the conflict map built from bad_triples
            # We didn't build it in this scope efficiently for O(1) lookup of pairs.
            # Let's build it once.
            break # This fallback is getting complicated, let's rely on ILP success.
    
    # If we reach here, ILP likely failed. Return a single element to satisfy "nonempty".
    # But we should try a bit harder.
    
    # Simple Greedy with precomputed conflicts
    conflict_pairs = {i: [] for i in range(1, n+1)}
    for i, j, k in bad_triples:
        a, b, c = i+1, j+1, k+1
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    current_set = set()
    for x in order:
        valid = True
        for u, v in conflict_pairs[x]:
            if u in current_set and v in current_set:
                valid = False
                break
        if valid:
            current_set.add(x)
            
    if current_set:
        return sorted(list(current_set))
    return [1]

# EVOLVE-BLOCK-END
