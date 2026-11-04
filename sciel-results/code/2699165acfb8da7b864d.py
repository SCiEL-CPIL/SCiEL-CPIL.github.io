"""Matrix multiplication 4x5x5 - structured rank reduction search."""
DIMENSIONS = [4, 5, 5]
import time
import numpy as np
from itertools import product, combinations

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    schoolbook_rank = n * m * p  # 100
    
    deadline = time.time() + 140  # Safety margin
    
    # Build schoolbook as baseline
    def make_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = make_schoolbook()
    
    # Verify schoolbook
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Strategy: Try to find rank reductions by identifying algebraic dependencies
    # For matrix multiplication, we can sometimes use "Strassen-like" tricks
    # but for exact integer/half-integer decompositions, this is very constrained
    
    # Key insight: If we have two terms (i1,j1,k1) and (i2,j2,k2) that share
    # the same i, j, or k index, we might be able to merge them.
    # 
    # More systematically: try to find U, V, W of rank R < 100 such that
    # the tensor equation is satisfied. We'll use a greedy approach:
    # Start with schoolbook, try to eliminate one term at a time by adjusting
    # other terms.
    
    # For each term to eliminate, check if we can redistribute its contribution
    # to other terms using half-integer adjustments
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try rank = 99: can we eliminate one term?
    # Term (i,j,k) contributes 1 to the (i,j,k) constraint.
    # If we set U[i*m+j, c] = 0 for that term's column, we need other terms
    # to compensate. This requires other terms to have non-zero values at
    # (i*m+j, j*p+k, k*n+i) positions.
    
    # Let's try a different approach: use the fact that the tensor is "separable"
    # in certain ways. For 4x5x5, try to find a decomposition using the
    # structure of the indices.
    
    # Actually, let's try a more principled approach:
    # The multiplication tensor for C = AB has a known rank. For small matrices,
    # we can try to find the exact rank by solving a system of equations.
    
    # Given the time constraints and the difficulty of finding rank < 100 for
    # 4x5x5 (which is likely close to optimal), let's try a few targeted
    # modifications and return the best we find.
    
    # Try: can we use a "block" structure? Split the 5x5 into blocks.
    # For example, split B into two parts: B = [B1 | B2] where B1 is 5x2, B2 is 5x3
    # Then C = A*B1 + A*B2, which might allow some sharing.
    
    # But this doesn't directly reduce the tensor rank.
    
    # Let's try a randomized local search: start with schoolbook, randomly
    # perturb entries (by +/-0.5) and check if we can still satisfy the
    # constraints with fewer non-zero columns.
    
    # Actually, the most promising approach for exact decompositions is to
    # look for "linear dependencies" in the constraint system.
    
    # The constraints are: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of 100 equations in the variables U, V, W.
    
    # For rank R, we have R*20 + R*25 + R*20 = 65R variables, and 100 equations.
    # For R >= 2, we have more variables than equations, so solutions exist.
    # The question is whether integer/half-integer solutions exist for R < 100.
    
    # Let's try R = 99 and see if we can find a solution by starting from
    # schoolbook and making a small perturbation.
    
    # For R = 99, we need to eliminate one column from the schoolbook.
    # Say we eliminate column c corresponding to (i0, j0, k0).
    # We need: for all (i,j,k), the sum over r=0..98 of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # where the original column c is removed, and we adjust other columns.
    
    # This is a nonlinear system. Let's try a simple heuristic:
    # For the eliminated term (i0,j0,k0), we need to add its contribution
    # to other terms. The simplest way: find another term (i1,j1,k1) that
    # shares indices and adjust its coefficient.
    
    # But for exact integer/half-integer entries, this is very restrictive.
    
    # Given the difficulty, let's try a few specific structured modifications
    # and return the best valid result.
    
    # Modification 1: Try to use "symmetry" in the 5x5 part.
    # If we can find a decomposition that exploits the square structure...
    
    # Actually, let's just do a quick randomized search for small rank reductions
    # and return the best we find within the time budget.
    
    import random
    random.seed(42)
    
    # Try rank 99 with a few random attempts
    for target_rank in [99, 98, 97]:
        if time.time() > deadline - 10:
            break
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Start with schoolbook, remove one column, try to fix
            U = [row[:] for row in U_sb]
            V = [row[:] for row in V_sb]
            W = [row[:] for row in W_sb]
            
            # Remove a random column
            col_to_remove = random.randint(0, schoolbook_rank - 1)
            
            # Determine which (i,j,k) this column corresponds to
            i0 = col_to_remove // (m*p)
            j0 = (col_to_remove // p) % m
            k0 = col_to_remove % p
            
            # Zero out this column
            for r in range(nm):
                U[r][col_to_remove] = 0
            for r in range(mp):
                V[r][col_to_remove] = 0
            for r in range(pn):
                W[r][col_to_remove] = 0
            
            # Now we need to compensate. The missing contribution is 1 at (i0,j0,k0).
            # Try to add 0.5 to two other terms that can cover this.
            
            # Find two terms that share at least one index with (i0,j0,k0)
            candidates = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i,j,k) != (i0,j0,k0):
                            # Check if this term can help
                            # It shares i if i == i0, shares j if j == j0, shares k if k == k0
                            shares = 0
                            if i == i0: shares += 1
                            if j == j0: shares += 1
                            if k == k0: shares += 1
                            if shares > 0:
                                candidates.append((i,j,k))
            
            if len(candidates) >= 2:
                # Pick two candidates
                c1, c2 = random.sample(candidates, 2)
                
                # Try to adjust: add 0.5 to each of these terms
                # This requires finding their column indices
                col1 = c1[0]*m*p + c1[1]*p + c1[2]
                col2 = c2[0]*m*p + c2[1]*p + c2[2]
                
                # But we need to be careful: adding 0.5 to U, V, W for these columns
                # will affect other constraints too.
                
                # This is getting complex. Let's just check if the modified
                # decomposition is valid (it probably won't be, but let's try).
                
                U2 = [row[:] for row in U]
                V2 = [row[:] for row in V]
                W2 = [row[:] for row in W]
                
                # Add 0.5 to the selected columns
                for r in range(nm):
                    U2[r][col1] += 0.5
                    U2[r][col2] += 0.5
                for r in range(mp):
                    V2[r][col1] += 0.5
                    V2[r][col2] += 0.5
                for r in range(pn):
                    W2[r][col1] += 0.5
                    W2[r][col2] += 0.5
                
                # Check validity
                if verify(U2, V2, W2, schoolbook_rank):
                    # Valid! But we still have rank 100 with some zeros.
                    # We need to actually reduce the rank by removing the zero column.
                    # Create new matrices without the zero column.
                    U3 = [[U2[r][c] for c in range(schoolbook_rank) if c != col_to_remove] for r in range(nm)]
                    V3 = [[V2[r][c] for c in range(schoolbook_rank) if c != col_to_remove] for r in range(mp)]
                    W3 = [[W2[r][c] for c in range(schoolbook_rank) if c != col_to_remove] for r in range(pn)]
                    
                    if verify(U3, V3, W3, schoolbook_rank - 1):
                        best_rank = schoolbook_rank - 1
                        best_U, best_V, best_W = U3, V3, W3
                        break
        
        if best_rank < schoolbook_rank:
            break
    
    # If no improvement found, return schoolbook
    if best_rank == schoolbook_rank:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
