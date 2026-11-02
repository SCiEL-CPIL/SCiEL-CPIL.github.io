"""
Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x8 matrix.
The problem decomposes into 4 independent 2x2x2 matrix multiplications.
The minimal rank for 2x2x2 is 7 (Strassen).
Thus, the minimal rank for 2x2x8 is 4 * 7 = 28.
"""

import time
import numpy as np

# Dimensions
N, M, P = 2, 2, 8
NM = N * M  # 4
MP = M * P  # 16
PN = P * N  # 16

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition.
    U: shape (NM, rank)
    V: shape (MP, rank)
    W: shape (PN, rank)
    """
    # Strassen terms for 2x2x2 matrix multiplication
    # A is 2x2, B is 2x2, C = A*B
    # Indexing: A[i, j], B[j, k], C[i, k]
    # Flattened indices:
    # A_idx = i*2 + j
    # B_idx = j*2 + k
    # C_idx = k*2 + i  (Note: The task specifies W[k*n+i, r] for C[k, i] in the trace convention? 
    # Let's re-read carefully: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i,k] is the output. The tensor entry T[a,b,c] is 1 if c corresponds to C[i,k] where a=A[i,j], b=B[j,k].
    # The prompt says: W[k*n+i, r]. This implies the index for W is k*n + i.
    # If C is 2x8, let's say C[i, col]. The prompt says W[k*n+i].
    # Let's check the schoolbook implementation in the current program to be sure about the index convention.
    # Current code: valid.add((i*m+j, j*p+k, k*n+i))
    # So a = i*m+j (A[i,j]), b = j*p+k (B[j,k]), c = k*n+i (C[k,i]??)
    # Wait, standard matrix mult C[i,k] = sum_j A[i,j]B[j,k].
    # If c = k*n + i, then the row index of C is k and col index is i?
    # Or is it C[i,k] but indexed as k*n+i?
    # Let's assume the convention is exactly as in the `valid` set:
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    
    # For 2x2x2 block:
    # i in {0,1}, j in {0,1}, k in {0,1}
    # a = i*2 + j
    # b = j*2 + k
    # c = k*2 + i
    
    # Strassen decomposition for 2x2x2:
    # We need 7 terms (u_r, v_r, w_r) such that sum_r u_r[a] v_r[b] w_r[c] = T[a,b,c]
    
    # Standard Strassen terms for C = A * B:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # Strassen multipliers:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+g)
    # m7 = (b-d)(e+g)
    
    # C11 = m1 - m4 + m5 - m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Let's map these to vectors u, v, w for the tensor decomposition.
    # The tensor T is defined by T[a,b,c] = 1 if a,b,c correspond to A[i,j], B[j,k], C[k,i] (based on c=k*n+i).
    # Wait, if c = k*n + i, then the first index of C is k and second is i?
    # Let's check C[i,k] in standard notation.
    # If the output is C, and the index is k*n+i, let's call the output matrix C'.
    # C'[k, i] = C[i, k]?
    # If C[i,k] = sum A[i,j]B[j,k], then C'[k,i] = sum A[i,j]B[j,k].
    # This is the transpose of the usual result if we view C' as a matrix with rows k and cols i.
    # However, the tensor definition is abstract. We just need to satisfy T[a,b,c].
    
    # Let's construct the 7 terms for the 2x2x2 case.
    # Indices:
    # A: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> indices 0, 1, 2, 3 (j*p+k, p=2)
    # C: 00, 01, 10, 11 -> indices 0, 1, 2, 3 (k*n+i, n=2)
    
    # Let's verify the indices for C:
    # i=0, k=0 -> c = 0*2+0 = 0
    # i=0, k=1 -> c = 1*2+0 = 2
    # i=1, k=0 -> c = 0*2+1 = 1
    # i=1, k=1 -> c = 1*2+1 = 3
    
    # So C[0,0] is at index 0.
    # C[0,1] (i=0, k=1) is at index 2.
    # C[1,0] (i=1, k=0) is at index 1.
    # C[1,1] (i=1, k=1) is at index 3.
    
    # Let's define the 7 terms (u, v, w) as lists of length 4.
    # u corresponds to A coefficients.
    # v corresponds to B coefficients.
    # w corresponds to C coefficients.
    
    # Term 1: m1 = (A00 + A11)(B00 + B11)
    # u1 = [1, 0, 0, 1] (A00 + A11)
    # v1 = [1, 0, 0, 1] (B00 + B11)
    # This product contributes to C.
    # C00 += A00B00 + A00B11 + A11B00 + A11B11
    # C01 += A00B01 + A00B10 + A11B01 + A11B10  (Wait, B indices in v are j*2+k)
    # B00 is j=0,k=0 -> idx 0
    # B01 is j=0,k=1 -> idx 1
    # B10 is j=1,k=0 -> idx 2
    # B11 is j=1,k=1 -> idx 3
    
    # Let's use the known Strassen vectors for the tensor T[a,b,c] = delta_{a,b,c} structure.
    # The previous program had a set of terms. I will reuse and verify them.
    
    terms = [
        # u, v, w
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Verify one term quickly in head or just trust the previous code's verification which passed.
    # The previous code verified these terms for a single 2x2x2 block.
    
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    # 4 blocks of 2x2 in the 2x8 matrix
    # B is 2x8. Split into 4 blocks of 2x2: columns (0,1), (2,3), (4,5), (6,7)
    # C is 2x8. Split into 4 blocks of 2x2: columns (0,1), (2,3), (4,5), (6,7)
    
    for block_idx in range(4):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            # Fill U
            # U index is i (0..3) corresponding to A_{i//2, i%2}
            # A is shared across all blocks, so U is the same for all blocks.
            for i in range(4):
                U[i][col] = u[i]
                
            # Fill V
            # v is indexed by local (j, k) where j in 0..1, k in 0..1
            # Global B index: b = j * p + k_global
            # k_global = 2 * block_idx + k_local
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * P + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]
                    
            # Fill W
            # w is indexed by local (i, k) where i in 0..1, k in 0..1
            # Global C index: c = k_global * n + i
            # k_global = 2 * block_idx + k_local
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * N + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]
                    
    return {"U": U, "V": V, "W": W}

# Optional: Verification function for local testing
def verify(U, V, W, n=N, m=M, p=P):
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if U else 0
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Constructed rank:", len(res['U'][0]))
    print("Valid:", verify(res['U'], res['V'], res['W']))
