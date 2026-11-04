"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return the 7-term Strassen decomposition for 2x2 * 2x2.
    Each term is (u, v, w) where u, v, w are length-4 vectors (flattened 2x2).
    u[i*2+j], v[j*2+k], w[k*2+i]
    """
    # Strassen's 7 products for 2x2*2x2
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 + M7
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # U vectors (for A), V vectors (for B), W vectors (for C)
    # Flattened: index = row*2 + col
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    terms = []
    # M1: (A00+A11)*(B00+B11)
    # u = [1,0,0,1], v = [1,0,0,1], w = [1,0,0,1]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2: (A10+A11)*B00
    # u = [0,0,1,1], v = [1,0,0,0], w = [0,0,1,0]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # M3: A00*(B01-B11)
    # u = [1,0,0,0], v = [0,1,0,-1], w = [0,1,0,0]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # M4: A11*(B10-B00)
    # u = [0,0,0,1], v = [0,0,1,-1], w = [0,0,1,1]
    terms.append(([0,0,0,1], [0,0,1,-1], [0,0,1,1]))
    # M5: (A00+A01)*B11
    # u = [1,1,0,0], v = [0,0,0,1], w = [-1,0,0,-1]
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,0,-1]))
    # M6: (A10-A00)*(B00+B01)
    # u = [-1,0,1,0], v = [1,1,0,0], w = [0,0,0,1]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7: (A01-A11)*(B10+B11)
    # u = [0,1,0,-1], v = [0,0,1,1], w = [0,0,0,-1]
    terms.append(([0,1,0,-1], [0,0,1,1], [0,0,0,-1]))
    
    return terms

def _kron_decomp_4x4x4():
    """Build 4x4*4x4 decomposition from Strassen⊗Strassen, rank 49.
    Returns list of (U_vec, V_vec, W_vec) where each is length 16.
    """
    s2 = _strassen_2x2()
    terms = []
    for (u1, v1, w1) in s2:
        for (u2, v2, w2) in s2:
            # Kronecker product of vectors
            # u = u1 ⊗ u2, v = v1 ⊗ v2, w = w1 ⊗ w2
            u = [u1[a] * u2[b] for a in range(4) for b in range(4)]
            v = [v1[a] * v2[b] for a in range(4) for b in range(4)]
            w = [w1[a] * w2[b] for a in range(4) for b in range(4)]
            terms.append((u, v, w))
    return terms

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Build initial decomposition:
    # 4x4x4 part: Strassen⊗Strassen, 49 terms
    # 4x4x3 part: schoolbook, 4*4*3 = 48 terms
    # Total: 97 terms
    
    terms_444 = _kron_decomp_4x4x4()
    
    # Schoolbook for 4x4x3 (output columns 4,5,6)
    terms_443 = []
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k=0,1,2 correspond to output columns 4,5,6
                u = [0] * (n*m)
                v = [0] * (m*p)
                w = [0] * (p*n)
                u[i*m + j] = 1
                v[j*p + (k+4)] = 1  # output column k+4
                w[(k+4)*n + i] = 1
                terms_443.append((u, v, w))
    
    # Combine: for the 4x4x4 part, w vectors are length p*n = 28, but we only have 16 entries
    # Need to pad w vectors to length 28
    all_terms = []
    for (u, v, w) in terms_444:
        # u is length 16 (4*4), v is length 16 (4*4), w is length 16 (4*4)
        # Pad to: u length n*m=16, v length m*p=28, w length p*n=28
        u_pad = u[:]  # already length 16
        v_pad = v[:16] + [0]*12  # v for 4x4 part, pad with 0s for columns 4-6
        w_pad = w[:16] + [0]*12  # w for 4x4 part, pad with 0s for columns 4-6
        all_terms.append((u_pad, v_pad, w_pad))
    
    for (u, v, w) in terms_443:
        all_terms.append((u, v, w))
    
    rank_init = len(all_terms)
    
    # Now apply peeling: try to remove linearly dependent terms
    # A term c is redundant if there exist coefficients alpha_d (d != c) such that
    # sum_d alpha_d * (u_d ⊗ v_d ⊗ w_d) = u_c ⊗ v_c ⊗ w_c
    
    # More practically: check if the tensor T can be reconstructed without term c.
    # Build the full tensor T of shape (n*m, m*p, p*n) where T[a,b,c] = sum over all terms
    # and check if removing any term still gives exact T.
    
    # Actually, for peeling, we check if the c-th term vector (u_c, v_c, w_c) is in the
    # span of the other terms. This is a 3-linear dependency check.
    
    # Simpler approach: iteratively try to remove each term and verify the decomposition
    # still exactly reconstructs the multiplication tensor.
    
    # Build target tensor T: T[i*m+j, j'*p+k, k'*n+i'] ... wait, let me be careful.
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k)
    # Actually no: for matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is: for each (i,j,k), the contribution is 1 (it's always included)
    # The decomposition must satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # Let's verify the initial decomposition is correct
    def verify(terms):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for (u, v, w) in terms:
                        total += Fraction(u[i*m+j]) * Fraction(v[j*p+k]) * Fraction(w[k*n+i])
                    if total != 1:
                        return False
        return True
    
    if not verify(all_terms):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Peeling: try to remove each term
    current_terms = list(all_terms)
    removed = set()
    
    # Try to remove terms one by one (greedy)
    # For each term, check if the remaining terms still form a valid decomposition
    for idx in range(len(current_terms)):
        if idx in removed:
            continue
        # Try removing term idx
        trial_terms = [t for i, t in enumerate(current_terms) if i != idx and i not in removed]
        if verify(trial_terms):
            removed.add(idx)
    
    final_terms = [t for i, t in enumerate(current_terms) if i not in removed]
    rank_final = len(final_terms)
    
    # Build output matrices
    U = [[0] * rank_final for _ in range(n * m)]
    V = [[0] * rank_final for _ in range(m * p)]
    W = [[0] * rank_final for _ in range(p * n)]
    for c, (u, v, w) in enumerate(final_terms):
        for a in range(n * m):
            U[a][c] = u[a]
        for b in range(m * p):
            V[b][c] = v[b]
        for d in range(p * n):
            W[d][c] = w[d]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
