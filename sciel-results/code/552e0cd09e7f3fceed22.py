import numpy as np
import time

def construct():
    deadline = time.time() + 110.0
    n, m, p = 2, 4, 4
    
    # Try to find a rank-24 decomposition
    # The equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Use a known construction or find one computationally.
    
    # Let me try a constructive approach with 24 terms.
    # I'll use the following idea:
    # 
    # Write the product as C = A*B where A is 2x4, B is 4x4.
    # Use 24 rank-1 terms.
    #
    # Here's a specific construction attempt:
    
    rank = 24
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Let me try to use a pattern based on Strassen with sharing.
    # 
    # For 2x2x2 Strassen (7 terms):
    # M1=(a00+a11)(b00+b11) -> c00,c11
    # M2=(a10+a11)b00 -> c10,c11
    # M3=a00(b01-b11) -> c01,c11
    # M4=a11(b10-b00) -> c00,c10
    # M5=(a00+a01)b11 -> c00,c01
    # M6=(a10-a00)(b00+b01) -> c11
    # M7=(a01-a11)(b10+b11) -> c00
    #
    # Now for 2x4 x 4x4, let me define 24 terms.
    
    # I'll use a different approach: solve the system numerically.
    # Fix the structure of U, V, W to have small integer entries and solve.
    
    # Actually, let me try a specific known construction.
    # Based on the literature, rank(2,4,4) = 24.
    
    # Here's a construction:
    # We use 24 terms where the U, V, W matrices have entries in {-1, 0, 1}.
    
    # Let me define the terms explicitly. I'll use a pattern:
    
    # Term indices 0-23
    # For each term, specify which U, V, W entries are nonzero.
    
    terms = []
    
    # I'll use a systematic construction. Let me define:
    # For the "top-left" 2x2x2 block (A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]):
    # Use Strassen terms 0-6 (7 terms)
    
    # For "top-right" (A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]):
    # Share some terms with top-left? No, different B.
    # Use 7 terms: 7-13
    
    # For "bottom-left" (A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]):
    # Use 7 terms: 14-20
    
    # For "bottom-right" (A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]):
    # Need 7 terms but only have 24-21=3 left.
    
    # So 7+7+7+3 = 24. The last group only gets 3 terms. This won't work.
    
    # Alternative: 6+6+6+6 = 24. But 2x2x2 has rank 7, not 6.
    
    # Alternative: Use a different decomposition of 2x2x2 that allows sharing.
    
    # Key insight: If two 2x2x2 products share the same A or same B, we can share terms.
    
    # Let me try: 
    # Group 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (7 terms)
    # Group 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (7 terms, but share A factors)
    # 
    # If I use the same "left" Strassen factors for A[0:2,0:2] in both groups,
    # I can potentially save terms. But the B factors are different.
    
    # Actually, in Strassen, the 7 terms involve 7 different "linear combinations" of A
    # multiplied by 7 different "linear combinations" of B. If the A combinations are
    # the same, I still need 7 multiplications because the B combinations differ.
    
    # So sharing A doesn't help reduce the number of terms.
    
    # Let me think about this differently. What if I use a non-Strassen decomposition
    # of 2x2x2 that has more terms but allows better sharing?
    
    # Or what if I use a completely different approach?
    
    # Here's another idea: Use the fact that 2x4 x 4x4 can be seen as
    # (2x4 x 4x1) * 4 repeated, but that doesn't help.
    
    # Let me try to just solve the system numerically.
    
    # I'll parameterize U, V, W with small integer values and solve.
    
    # Approach: 
    # 1. Fix W to be a specific pattern (e.g., identity-like)
    # 2. Solve for U and V
    
    # Or: Use iterative refinement.
    
    # Let me try a simple approach: assume all entries are in {-1, 0, 1} and
    # search for a valid assignment.
    
    # This is a constraint satisfaction problem. With 768 binary variables and
    # 32 constraints, it might be tractable with smart search.
    
    # But that's too many variables. Let me reduce.
    
    # Alternative: Fix the rank to 24 and use a specific structure.
    
    # Let me try: each term r has at most 2 nonzeros in U, at most 2 in V, at most 2 in W.
    # This is like the Strassen pattern.
    
    # With this constraint, I have 24 terms, each with up to 6 nonzero entries.
    # Total nonzeros: up to 144.
    
    # Let me try to construct this by hand or with a small search.
    
    # Actually, let me just return the rank-28 solution for now, as finding
    # a valid rank-24 construction is complex.
    
    # But wait - let me try one more idea. What if rank 24 is achievable
    # with a specific algebraic identity?
    
    # Here's my best attempt at a rank-24 construction:
    
    # I'll use 24 terms with the following structure:
    # 4 groups of 6 terms each, where each group handles a 2x2 x 2x2 -> 2x2
    # product using a rank-6 decomposition (which doesn't exist for general 2x2x2).
    
    # Since rank-6 for 2x2x2 doesn't exist, this won't work.
    
    # Let me try yet another approach: use the "subproduct" method.
    
    # I give up trying to find rank 24 by hand. Let me just return rank 28.
    
    # Build rank-28 Strassen decomposition
    U = np.zeros((n*m, 28))
    V = np.zeros((m*p, 28))
    W = np.zeros((p*n, 28))
    
    def add_strassen(a_row, a_col, b_row, b_col, c_row, c_col, start):
        a00 = a_row * m + a_col
        a01 = a_row * m + a_col + 1
        a10 = (a_row + 1) * m + a_col
        a11 = (a_row + 1) * m + a_col + 1
        
        b00 = b_row * p + b_col
        b01 = b_row * p + b_col + 1
        b10 = (b_row + 1) * p + b_col
        b11 = (b_row + 1) * p + b_col + 1
        
        c00 = c_col * n + c_row
        c01 = (c_col + 1) * n + c_row
        c10 = c_col * n + (c_row + 1)
        c11 = (c_col + 1) * n + (c_row + 1)
        
        r = start
        U[a00][r] += 1; U[a11][r] += 1
        V[b00][r] += 1; V[b11][r] += 1
        W[c00][r] += 1; W[c11][r] += 1
        r += 1
        
        U[a10][r] += 1; U[a11][r] += 1
        V[b00][r] += 1
        W[c10][r] += 1; W[c11][r] += -1
        r += 1
        
        U[a00][r] += 1
        V[b01][r] += 1; V[b11][r] += -1
        W[c01][r] += 1; W[c11][r] += 1
        r += 1
        
        U[a11][r] += 1
        V[b10][r] += 1; V[b00][r] += -1
        W[c00][r] += 1; W[c10][r] += 1
        r += 1
        
        U[a00][r] += 1; U[a01][r] += 1
        V[b11][r] += 1
        W[c00][r] += -1; W[c01][r] += 1
        r += 1
        
        U[a10][r] += 1; U[a00][r] += -1
        V[b00][r] += 1; V[b01][r] += 1
        W[c11][r] += 1
        r += 1
        
        U[a01][r] += 1; U[a11][r] += -1
        V[b10][r] += 1; V[b11][r] += 1
        W[c00][r] += 1
        r += 1
        
        return r
    
    r = add_strassen(0, 0, 0, 0, 0, 0, 0)
    r = add_strassen(0, 2, 2, 0, 0, 0, r)
    r = add_strassen(0, 0, 0, 2, 0, 2, r)
    r = add_strassen(0, 2, 2, 2, 0, 2, r)
    
    assert r == 28
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
