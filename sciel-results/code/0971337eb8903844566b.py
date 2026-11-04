"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

import time
import numpy as np
import sys

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5  # A: n x m, B: m x p, C: n x p
    
    # We want to find U, V, W of rank r such that:
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # where T[i,j,k] is the matrix multiplication tensor.
    # This is equivalent to C = A @ B.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor notation: C[i,k] = sum_j U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Wait, the standard decomposition is:
    # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_k V[j,k,r] B[j,k]) ... no.
    # The tensor T is defined by T[i,j,k] = 1 if i,j,k correspond to the multiplication.
    # Actually, the task defines U, V, W shapes:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # And the condition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k?
    # No, that's not how matrix multiplication tensor decomposition works.
    # The matrix multiplication tensor T_{i,j,k} has entries 1.
    # A rank-1 decomposition of T is U_{i,j} V_{j,k} W_{k,i} ? No.
    # Let's look at the standard definition.
    # T = sum_r u_r \otimes v_r \otimes w_r
    # T[i,j,k] = sum_r u_r[i,j] v_r[j,k] w_r[k,i] ? No, indices must match.
    # Usually T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] is not right either because of the cyclic indices in W.
    # Let's check the interface:
    # U: (n*m, rank) -> index i*m+j
    # V: (m*p, rank) -> index j*p+k
    # W: (p*n, rank) -> index k*n+i
    # Condition: "exactly decompose the matrix multiplication tensor"
    # The tensor for matrix multiplication C = A B is T[i,j,k] = 1.
    # The decomposition is: 1 = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] ?
    # Let's test this. If U[i*m+j, r] = A[i,j], V[j*p+k, r] = B[j,k], W[k*n+i, r] = C[k,i] ?
    # No, U, V, W are fixed coefficients.
    # The condition is: For all i, j, k:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # This implies that the tensor T[i,j,k] = 1 for all i,j,k?
    # But matrix multiplication tensor has T[i,j,k] = 1 only if the indices correspond to the product?
    # No, the matrix multiplication tensor T is a 3-tensor of size n x m x p where T[i,j,k] = 1.
    # Wait, the tensor T_{i,j,k} represents the coefficient of A[i,j] B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k.
    # Yes, the tensor is all ones.
    # So we need: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # Let's verify the schoolbook solution in the current code:
    # rank = n*m*p = 30
    # For each (i,j,k), col = i*m*p + j*p + k (or similar)
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    # Sum over r: only one term is non-zero for a specific (i,j,k) if the mapping is unique?
    # In the current code:
    # col increments for each i,j,k.
    # U[i*m+j, col] = 1. Note that U index is i*m+j.
    # V[j*p+k, col] = 1. V index is j*p+k.
    # W[k*n+i, col] = 1. W index is k*n+i.
    # For a fixed (i,j,k), let col_c be the column corresponding to this triple.
    # Then U[i*m+j, col_c] = 1, V[j*p+k, col_c] = 1, W[k*n+i, col_c] = 1.
    # For other columns col_d corresponding to (i',j',k'),
    # U[i*m+j, col_d] is 1 only if i'm'+j' = i*m+j AND i'!=i or j'!=j?
    # Since i<m and j<m, i*m+j uniquely identifies (i,j).
    # So U[i*m+j, col_d] = 1 implies i'=i and j'=j.
    # Similarly V[j*p+k, col_d] = 1 implies j'=j and k'=k.
    # So for a fixed (i,j,k), the only non-zero term in the sum is the one where (i',j',k') = (i,j,k).
    # The value is 1*1*1 = 1.
    # So the schoolbook solution works.
    
    # Now, can we do better?
    # Rank 28 is claimed possible.
    # Let's try to construct a rank-28 solution.
    # We can split the problem into two parts:
    # 1. Multiply 2x3 A by 3x3 B1 (columns 0-2 of B). Result C1 is 2x3.
    # 2. Multiply 2x3 A by 3x2 B2 (columns 3-4 of B). Result C2 is 2x2.
    # Total rank = rank(2x3x3) + rank(2x3x2).
    
    # What is the minimal rank for 2x3x3?
    # Schoolbook: 2*3*3 = 18.
    # Can we do better?
    # 2x2x2 has rank 7 (Strassen).
    # 2x2x3?
    # Let's look for known results or try to construct.
    # Actually, let's just use the schoolbook for now to ensure correctness, then try to optimize if time permits.
    # But the goal is to improve the score. Score 1.0 corresponds to rank 30.
    # If we find rank 29, score = 30/29 > 1.
    # If we find rank 28, score = 30/28 > 1.
    
    # Let's try to find a rank-29 or rank-28 decomposition.
    # One strategy: Use Strassen's algorithm where possible.
    # 2x2x2 multiplication has rank 7.
    # 2x3x3 can be viewed as two 2x2x2 multiplications plus corrections?
    # A = [A1 A2] where A1 is 2x2, A2 is 2x1.
    # B1 = [B11 B12] where B11 is 2x2, B12 is 2x1.
    # C1 = A1 B11 + A2 B12? No.
    # A (2x3) * B1 (3x3) = C1 (2x3).
    # Let A = [[a00, a01, a02], [a10, a11, a12]]
    # B1 = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C1[0,0] = a00 b00 + a01 b10 + a02 b20
    # C1[0,1] = a00 b01 + a01 b11 + a02 b21
    # C1[0,2] = a00 b02 + a01 b12 + a02 b22
    # C1[1,0] = a10 b00 + a11 b10 + a12 b20
    # C1[1,1] = a10 b01 + a11 b11 + a12 b21
    # C1[1,2] = a10 b02 + a11 b12 + a12 b22
    
    # This is 6 terms, each a sum of 3 products.
    # Total 18 products.
    
    # Can we reduce the rank?
    # Let's try to use a generic solver for small ranks?
    # The number of variables is huge.
    # rank r: U (6 x r), V (15 x r), W (10 x r).
    # Equations: 2*3*5 = 30 equations.
    # Unknowns: 6r + 15r + 10r = 31r.
    # For r=28, 31*28 = 868 unknowns, 30 equations. Underdetermined.
    # We can set some variables to 0 or 1 and solve the rest?
    # This is a system of polynomial equations.
    
    # Let's try a heuristic search.
    # Start with rank 30 (schoolbook).
    # Try to merge terms or find linear dependencies.
    
    # Alternative: Use the known rank-7 algorithm for 2x2x2 (Strassen) to handle parts of the multiplication.
    # 2x3x3:
    # Split A into A1 (2x2) and A2 (2x1).
    # Split B1 into B11 (2x2), B12 (2x1), and B13 (2x1)? No, B1 is 3x3.
    # B1 rows: b0, b1, b2.
    # C1 = a0 b0 + a1 b1 + a2 b2.
    # a0, a1, a2 are row vectors of length 3.
    # b0, b1, b2 are column vectors of length 3.
    # This is not standard matrix multiplication.
    
    # Let's stick to the tensor decomposition definition.
    # T[i,j,k] = 1.
    # We want to minimize r such that T = sum_r u_r \otimes v_r \otimes w_r.
    # u_r is a vector of length n*m = 6.
    # v_r is a vector of length m*p = 15.
    # w_r is a vector of length p*n = 10.
    
    # Let's try to construct a solution with rank 29.
    # Maybe we can save 1 term by exploiting symmetry or zero patterns?
    # In schoolbook, each term is a basis vector.
    # If we can find a linear combination that covers two terms with one, we reduce rank.
    # This is equivalent to finding a rank-1 update that satisfies the equations.
    
    # Let's use a simple iterative improvement.
    # Start with schoolbook rank 30.
    # Randomly perturb and project?
    
    # Given the time limit, I will implement a verified rank-28 decomposition if I can find one,
    # otherwise fall back to rank 30.
    # Actually, let's try to hardcode a rank-28 solution if possible.
    # Since I don't have one, I will try to generate one using a solver.
    # But solving polynomial systems is hard.
    
    # Let's try a different approach:
    # Use the fact that 2x2x2 has rank 7.
    # 2x3x5 can be embedded in 3x3x5?
    # Or 2x3x5 is a sub-tensor of 3x3x5.
    # Rank(3x3x5) >= Rank(2x3x5).
    # Schoolbook 3x3x5 is 45.
    
    # Let's just return the schoolbook solution for now, but optimized for speed.
    # The current code is already fast.
    
    # Wait, the inspiration says rank 28 is possible.
    # Let's try to construct it by splitting into 2x3x3 and 2x3x2.
    # Rank(2x3x3) + Rank(2x3x2).
    # Schoolbook: 18 + 12 = 30.
    # If we can find Rank(2x3x3) = 17 and Rank(2x3x2) = 11, total 28.
    
    # Is Rank(2x3x2) = 11 possible?
    # 2x3x2 tensor has 12 entries.
    # Schoolbook rank 12.
    # Can we do rank 11?
    # Let's try to solve for rank 11.
    # 11 unknowns for U, V, W? No, 11 columns each.
    # U: 6x11, V: 15x11, W: 10x11.
    # 6*11 + 15*11 + 10*11 = 31*11 = 341 unknowns.
    # 12 equations.
    # We can fix many variables to 0 or 1.
    
    # Let's try a greedy algorithm to reduce rank.
    # Start with rank 30.
    # Try to set one column to zero and see if the system is still solvable?
    # No, the columns are the terms.
    
    # Let's just return the schoolbook solution. It's safe.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
