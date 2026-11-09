"""Combined program: structured rank-59 decomposition with verification."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    # 1. Baseline: Schoolbook Decomposition (Rank 60)
    rank_baseline = n * m * p
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j][col] = 1
                V_base[j * p + k][col] = 1
                W_base[k * n + i][col] = 1
                col += 1

    # 2. Attempt Rank 59 Decomposition
    # The schoolbook rank is n*m*p = 60.
    # We try to reduce this by 1 by sharing one rank term between two (i,k) pairs.
    # Specifically, we take the term for (i=0, k=0) and (i=0, k=1) and try to merge them.
    # In the schoolbook, term (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # Let's try to construct a rank 59 solution directly.
    
    # Strategy:
    # Keep 59 terms.
    # 58 terms are standard schoolbook terms for specific (i,j,k) tuples.
    # 1 term is a "merged" term that covers the missing (i,j,k) tuples.
    #
    # Actually, a simpler constructive approach for rank 59:
    # The tensor T_{i,j,k} = 1.
    # We need sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1.
    #
    # Let's try a specific algebraic construction.
    # Consider the decomposition where we use rank R.
    # For R=59, we can try to set up the system and solve for one variable.
    # However, random search for R=59 with integer/half-integer constraints is hard.
    
    # Let's try a specific known lower rank if it exists, or a constructive one.
    # For 3x4x5, the border rank is 59. The rank might be 59 or 60.
    # If rank is 59, there exists a decomposition.
    
    # Let's try to construct a rank 59 decomposition manually or via a small solver.
    # Since we need integer/half-integer entries, we can scale by 2 to work with integers.
    # Let U', V', W' be integer matrices such that sum_c U'_{i,j,c} V'_{j,k,c} W'_{k,i,c} = 2.
    # Then U = U'/2, etc. are half-integers.
    
    # Let's try to find a rank 59 decomposition using a specific pattern.
    # One common technique is to start with rank 60 and perturb one term.
    
    # Alternative: Use the fact that the tensor is symmetric in some sense? No.
    
    # Let's try a direct search for R=59 with a smarter initialization.
    # Initialize U, V, W with the schoolbook basis, then modify one column.
    # Or, use a linear algebra approach.
    
    # Given the time constraint and the difficulty of finding an exact rank 59 solution
    # with integer/half-integer entries via random search, we will attempt a structured
    # search for R=59.
    
    # If we cannot find a valid R=59 solution quickly, we return the R=60 baseline.
    
    # Let's try a specific construction:
    # We want to eliminate one rank term.
    # Let the 60th term be the one for i=0, j=0, k=0.
    # We modify the other terms to compensate.
    # This is generally not possible with simple integer modifications.
    
    # Let's try a random search for R=59 with entries in {-1, 0, 1} and W entries in {-1, -0.5, 0, 0.5, 1}.
    # But the current program's search was inefficient.
    
    # Instead, let's try to verify if a rank 59 solution exists by solving a system.
    # We fix 58 of the rank terms to be standard schoolbook terms for 58 distinct (i,j,k) triples.
    # The remaining 1 term must satisfy the equations for the 2 remaining (i,j,k) triples.
    # This is a system of 2 equations for 3*4 + 4*5 + 5*3 = 12 + 20 + 15 = 47 variables.
    # This system is underdetermined, so a solution likely exists.
    # We just need to find one with integer/half-integer entries.
    
    # Let's pick the 2 missing triples: (0,0,0) and (0,0,1).
    # The other 58 triples are all (i,j,k) except these two.
    # For the 58 standard terms, U, V, W are 1 at the appropriate positions.
    # For the 59th term (index c=58), let u_{j} = U[0*m+j, 58], v_{k} = V[0*p+k, 58], w_{i} = W[i, 58] (since i=0 for both missing triples, W[k*n+0, 58] is the relevant part, but W depends on k and i. Wait, W index is k*n+i.
    # The missing triples are (i=0, j=0, k=0) and (i=0, j=0, k=1).
    # Equation 1 (for i=0, j=0, k=0):
    # Sum_{c=0}^{58} U[0*m+0, c] V[0*p+0, c] W[0*n+0, c] = 1
    # The standard terms for (i,j,k) contribute 1 if (i,j,k) is in the standard set.
    # (0,0,0) is NOT in the standard set. So the sum over c=0..57 is 0.
    # So we need: U[0, 58] * V[0, 58] * W[0, 58] = 1.
    
    # Equation 2 (for i=0, j=0, k=1):
    # Sum_{c=0}^{58} U[0*m+0, c] V[0*p+1, c] W[1*n+0, c] = 1
    # (0,0,1) is NOT in the standard set. So sum over c=0..57 is 0.
    # So we need: U[0, 58] * V[1, 58] * W[15, 58] = 1.  (Note: W index is k*n+i = 1*3+0 = 3? No, n=3, so 1*3+0=3. Wait, W shape is p*n = 5*3=15. Index is k*n+i. For k=1, i=0, index is 3.)
    
    # So we have:
    # U0 * V0 * W0 = 1
    # U0 * V1 * W3 = 1
    # where U0 = U[0, 58], V0 = V[0, 58], V1 = V[1, 58], W0 = W[0, 58], W3 = W[3, 58].
    # We also need to ensure that the 59th term does NOT contribute to any other (i,j,k) triple.
    # For any other (i,j,k) in the standard set, the sum must be 1.
    # The standard term for (i,j,k) contributes 1.
    # The 59th term contributes U[i*m+j, 58] * V[j*p+k, 58] * W[k*n+i, 58].
    # We need this contribution to be 0 for all (i,j,k) in the standard set.
    
    # This imposes many constraints.
    # Let's try to choose U[0, 58] = 1, V[0, 58] = 1, W[0, 58] = 1.
    # Then U0 * V1 * W3 = 1 => 1 * V1 * W3 = 1 => V1 * W3 = 1.
    # Let V[1, 58] = 1, W[3, 58] = 1.
    # Now we need U[i*m+j, 58] * V[j*p+k, 58] * W[k*n+i, 58] = 0 for all (i,j,k) != (0,0,0), (0,0,1).
    
    # Let's check (i=0, j=1, k=0):
    # U[0*m+1, 58] * V[1*p+0, 58] * W[0*n+0, 58] = U[1, 58] * V[4, 58] * W[0, 58]
    # We set W[0, 58] = 1. So we need U[1, 58] * V[4, 58] = 0.
    # We can set U[1, 58] = 0.
    
    # Let's check (i=0, j=0, k=2):
    # U[0, 58] * V[2, 58] * W[6, 58] = 1 * V[2, 58] * W[6, 58] = 0.
    # Set V[2, 58] = 0.
    
    # This seems feasible. We can set most entries of the 59th column to 0, except where necessary.
    # Let's try to construct this explicitly.
    
    R = 59
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Fill 58 standard terms
    # Missing triples: (0,0,0) and (0,0,1)
    missing = {(0,0,0), (0,0,1)}
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) in missing:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # col should be 58 now.
    # Set up the 59th term (index 58)
    U[0][58] = 1
    V[0][58] = 1
    W[0][58] = 1
    
    V[1][58] = 1
    W[3][58] = 1  # k=1, i=0 => index 3
    
    # Now check if this satisfies all equations.
    # We need to verify that for all (i,j,k):
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(R):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        # We found a rank 59 solution!
        return {"U": U, "V": V, "W": W}
    else:
        # If the simple construction failed, return baseline
        return {"U": U_base, "V": V_base, "W": W_base}

