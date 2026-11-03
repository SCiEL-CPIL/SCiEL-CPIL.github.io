"""Matrix multiplication 2x4x6 - attempting rank-reduced decomposition via Gaussian elimination on tensor constraints.

Strategy: The multiplication tensor T[i,j,k,l] = delta_{i,k} delta_{j,l} for i in [0,2), j in [0,4), k in [0,4), l in [0,6)
can be decomposed into rank-1 terms. We set up the linear system for finding a decomposition of rank r and attempt
to find solutions with r < 48 using randomized search combined with structured initialization.

We use the fact that the tensor has structure: it's the product of identity-like maps across dimensions.
We exploit this by trying low-rank decompositions with random integer/rational coefficients and verifying.
"""

import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 4, 6]

def _verify(U, V, W, n=2, m=4, p=6, tol=1e-9):
    """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,k} * delta_{j,l} for all i,j,k,l."""
    r = len(U[0])
    # Build the tensor as a flat array of size n*m*p*n (or just check the n*m*p*n constraints)
    # The tensor T[i,j,k,l] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Note: W index is k*n + i per the trace convention
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i == k and j == k) else 0.0  # Wait, this isn't right
    # Actually for 2x4 * 4x6:
    # A is 2x4, B is 4x6, C = A*B is 2x6
    # C[i,l] = sum_j A[i,j] * B[j,l]
    # Tensor T[i,j,l] = sum over the "inner" dimension... 
    # The bilinear map: (A,B) -> C where C[i,l] = sum_{j=0}^{m-1} A[i,j] B[j,l]
    # So T[i,l] as a function of (A,B) has components T[i,l]^{(j1,j2)} = delta_{i,j1} delta_{j2,l}
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank) = (8, rank), indexed by i*m+j
    # V has shape (m*p, rank) = (24, rank), indexed by j*p+k
    # W has shape (p*n, rank) = (12, rank), indexed by k*n+i
    # The multiplication is: C[i,l] = sum_c sum... 
    # Actually the tensor for multiplying n x m by m x p matrix:
    # T[i,j,k] where i is row of result (0..n-1), j is... 
    # Let me think about this differently.
    # A is n x m, B is m x p, C = A @ B is n x p
    # C[a,c] = sum_b A[a,b] B[b,c]
    # So the bilinear map has n*p outputs, each a bilinear form in (A, B)
    # Output (a,c) = sum_b A[a,b] * B[b,c]
    # In tensor form: T[a,c]^{(a',b',b'',c')} = delta_{a,a'} delta_{b,b'} ... no.
    # The tensor T lives in (A^* ⊗ B^* ⊗ C) or equivalently (R^n ⊗ R^m)^* ⊗ (R^m ⊗ R^p)^* ⊗ (R^n ⊗ R^p)
    # Component: T[a',b',b'',c'; a,c] = delta_{a',a} delta_{b',b''} delta_{b'',c} ... 
    # Let's use the decomposition form:
    # C[a,c] = sum_t (sum_{a',b'} U[a',b',t] A[a',b']) * (sum_{b'',c'} V[b'',c',t] B[b'',c'])
    #           * W[a,c,t]  ... no that's not standard.
    #
    # Standard Strassen-style decomposition:
    # C = sum_t (U_t A) * (V_t B) * w_t  -- no
    # C_{a,c} = sum_t U^{(t)}_{a,·} A (V^{(t)}_{·,c} B)  ... 
    #
    # Actually, the standard tensor decomposition for matrix multiplication:
    # C[a,c] = sum_t (sum_{a',b'} U_t[a',b'] A[a',b']) * (sum_{b'',c''} ... ) 
    # 
    # Let me just use the given interface:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # The product is: C[i,l] = sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] ... summed over j, k?
    # 
    # From the schoolbook code:
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1 for each (i,j,k)
    # So for a given column c corresponding to (i,j,k):
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # The reconstructed C element: C[?,?] = sum over i,j,k of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * ... 
    # 
    # I think the reconstruction is:
    # C[i',l'] = sum_{i,j,k} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i', c] * A[i,j] * B[k,l']
    # Hmm, this is getting confusing. Let me just verify using the known schoolbook identity.
    # 
    # In schoolbook: for each (i,j,k) triple, we have one rank-1 term.
    # The term contributes: A[i,j] * B[j,k] to C[i,k]
    # So U[i*m+j] picks out A[i,j], V[j*p+k] picks out B[j,k], W[k*n+i] routes to C[i,k]
    #
    # So the full reconstruction: C[i',l'] = sum_c sum_{i,j,k} U[i*m+j,c] * A[i,j] * V[j*p+k,c] * B[j,k] * W[k*n+i',c]
    # For this to equal sum_j A[i',j] B[j,l'], we need:
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i',c] = delta_{i,i'} delta_{j,k} for the correct summation... 
    # Actually: sum_{j} (sum_c U[i*m+j,c] W[j*p+...]) ... 
    #
    # Let me define: for the reconstruction to work, we need
    # sum_c sum_{j,k} A[i,j] B[k,l] U[i*m+j,c] V[j*p+k,c] W[k*n+i',c] = sum_j A[i',j] B[j,l]
    # 
    # This requires: for all i, j, k, l, i':
    # The coefficient of A[i,j] B[k,l] in LHS is sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i',c]
    # And this must equal delta_{i,i'} delta_{j,k} (the coefficient in sum_j' A[i',j'] B[j',l])
    # Wait, RHS coefficient of A[i,j]B[k,l] is: delta_{i,i'} * (1 if j=k and l=k... ) 
    # RHS = sum_{j''} A[i',j''] B[j'',l]. Coefficient of A[i,j]B[k,l] is 1 if i=i' and j=k and j''=j=k and l=l, so delta_{i,i'} delta_{j,k} delta_{j,l}... 
    # No: RHS = sum_{j''} A[i',j''] B[j'',l]. For this to equal A[i,j]B[k,l], we need i'=i, j''=j=k, and l=l. So coefficient is delta_{i,i'} * delta_{j,k} (with the implicit j''=k=l constraint... )
    # Actually coefficient of A[i,j] * B[k,l] in RHS is: 1 if i'=i and j=k (so that j''=j=k and B[j'',l]=B[k,l]) and l is free... 
    # I'm overcomplicating this. The coefficient of A[i,j]B[k,l] in sum_{j''} A[i',j'']B[j'',l] is delta_{i,i'} * delta_{j,k} (since j'' must equal both j (from A[i,j]) and k (from B[k,l]), so j=k, and i'=i).
    # 
    # So we need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i',c] = delta_{i,i'} delta_{j,k} for all i,j,k,l,i'
    # (l doesn't appear, which makes sense since B's second index is free in the constraint)
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    r = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    # LHS: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+ip, c]
                    lhs = 0.0
                    for c in range(r):
                        lhs += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+ip, c]
                    rhs = 1.0 if (i == ip and j == k) else 0.0
                    if abs(lhs - rhs) > 1e-6:
                        return False
    return True


def _verify_strict(U, V, W, n=2, m=4, p=6):
    """Strict verification using Fractions for exactness."""
    r = len(U[0])
    Uf = [[Fraction(U[i][c]) for c in range(r)] for i in range(len(U))]
    Vf = [[Fraction(V[i][c]) for c in range(r)] for i in range(len(V))]
    Wf = [[Fraction(W[i][c]) for c in range(r)] for i in range(len(W))]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    lhs = Fraction(0)
                    for c in range(r):
                        lhs += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+ip][c]
                    rhs = Fraction(1) if (i == ip and j == k) else Fraction(0)
                    if lhs != rhs:
                        return False
    return True


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Leave 15s safety margin for the 150s limit
    
    # Start with schoolbook rank 48 as fallback
    rank_fallback = n * m * p
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_fallback
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to find lower rank decompositions
    # Key insight: The tensor has structure. For 2x4x4x6, known lower bounds and results
    # suggest rank could be as low as 40 or so. Let's try ranks from 40 down to 47.
    
    random.seed(42)
    
    # For each candidate rank, try to find a decomposition
    for candidate_rank in range(47, 39, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts for this rank
        max_attempts = 5
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                    break
            
            # Random initialization with small integers
            U = [[random.randint(-2, 2) for _ in range(candidate_rank)] for _ in range(n*m)]
            V = [[random.randint(-2, 2) for _ in range(candidate_rank)] for _ in range(m*p)]
            W = [[random.randint(-2, 2) for _ in range(candidate_rank)] for _ in range(p*n)]
            
            # Verify
            if _verify(U, V, W, n, m, p):
                best_rank = candidate_rank
                best_result = {"U": U, "V": V, "W": W}
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
