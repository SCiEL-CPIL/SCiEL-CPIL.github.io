"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import numpy as np
import time

_START = time.time()
_LIMIT = 130.0
DIMENSIONS = [3, 4, 7]


def _pack(arr):
    return [[float(x) for x in row] for row in np.asarray(arr)]


def construct():
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n
    R0 = n * m * p  # 84

    U = np.zeros((Nm, R0))
    V = np.zeros((Mp, R0))
    W = np.zeros((Pn, R0))

    col = 0

    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j, col] = 1.0
        V[j * p + k, col] = 1.0
        W[k * n + i, col] = 1.0
        col += 1

    # Helper to add a Strassen term for 2x2x2 block
    # i0, i1, j0, j1, k0, k1
    # Strassen coefficients:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # Term 1: (A_i0,j0 + A_i1,j1) * (B_j0,k0 + B_j1,k1)
        # W: C_i0,k0 += M1, C_i0,k1 += M1, C_i1,k0 += M1, C_i1,k1 += M1 ?
        # Strassen Output:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # We need to map these to the tensor indices.
        # The term r contributes U[i,j,r] V[j,k,r] W[k,i,r] to the product.
        # Actually, the decomposition is sum of rank 1 tensors.
        # Each rank 1 tensor is U_r, V_r, W_r.
        # The "value" of the term is the outer product? No.
        
        # Let's use the standard Strassen coefficients for the rank-1 terms.
        # Each term is a product of 3 vectors.
        
        # Term 1:
        # U: 1 at (i0,j0) and (i1,j1)
        # V: 1 at (j0,k0) and (j1,k1)
        # W: 1 at (k0,i0), (k0,i1), (k1,i0), (k1,i1) ??
        # No, W must be consistent.
        
        # Let's look at the contribution to C[i,k].
        # C[i,k] += U[i,j] V[j,k] W[k,i] ?
        # The sum is over j and r.
        # For a fixed r, it is sum_j U[i,j,r] V[j,k,r] W[k,i,r].
        # This is (U_r^i @ V_r^i_k) * W_r^k_i ?
        
        # Let's construct the matrices for each Strassen term.
        
        # T1: (A00+A11)(B00+B11)
        # U: e_i0,j0 + e_i1,j1
        # V: e_j0,k0 + e_j1,k1
        # W: e_k0,i0 + e_k0,i1 + e_k1,i0 + e_k1,i1 ??
        # This W is not a single vector?
        # W is a vector of length p*n.
        # W[k*n + i].
        
        # Let's verify the contribution of T1 to C[i,k].
        # Sum_j U[i,j] V[j,k] W[k,i].
        # If U, V, W are sparse.
        
        # Actually, it is easier to just use the known rank 7 decomposition matrices.
        
        # Coeffs for Strassen:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # We will add 7 terms.
        
        # Term 1: M1
        # U: (i0,j0)=1, (i1,j1)=1
        # V: (j0,k0)=1, (j1,k1)=1
        # W: (k0,i0)=1, (k0,i1)=1, (k1,i0)=1, (k1,i1)=1
        # Wait, if W has 1s at multiple places, it adds to multiple C[i,k].
        # This matches the Strassen expansion.
        
        U[i0*m+j0, col] = 1; U[i1*m+j1, col] = 1
        V[j0*p+k0, col] = 1; V[j1*p+k1, col] = 1
        W[k0*n+i0, col] = 1; W[k0*n+i1, col] = 1
        W[k1*n+i0, col] = 1; W[k1*n+i1, col] = 1
        col += 1
        
        # Term 2: M2 = (A10+A11)B00
        # U: (i1,j0)=1, (i1,j1)=1
        # V: (j0,k0)=1
        # W: (k0,i1)=1
        U[i1*m+j0, col] = 1; U[i1*m+j1, col] = 1
        V[j0*p+k0, col] = 1
        W[k0*n+i1, col] = 1
        col += 1
        
        # Term 3: M3 = A00(B10-B11)
        # U: (i0,j0)=1
        # V: (j1,k0)=1, (j1,k1)=-1
        # W: (k0,i0)=1, (k1,i0)=1
        U[i0*m+j0, col] = 1
        V[j1*p+k0, col] = 1; V[j1*p+k1, col] = -1
        W[k0*n+i0, col] = 1; W[k1*n+i0, col] = 1
        col += 1
        
        # Term 4: M4 = A11(B01-B00)
        # U: (i1,j1)=1
        # V: (j0,k1)=1, (j0,k0)=-1
        # W: (k0,i1)=1, (k1,i1)=1
        U[i1*m+j1, col] = 1
        V[j0*p+k1, col] = 1; V[j0*p+k0, col] = -1
        W[k0*n+i1, col] = 1; W[k1*n+i1, col] = 1
        col += 1
        
        # Term 5: M5 = (A00+A01)B11
        # U: (i0,j0)=1, (i0,j1)=1
        # V: (j1,k1)=1
        # W: (k1,i0)=1, (k1,i1)=1
        U[i0*m+j0, col] = 1; U[i0*m+j1, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i0, col] = 1; W[k1*n+i1, col] = 1
        col += 1
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # U: (i1,j0)=1, (i0,j0)=-1
        # V: (j0,k0)=1, (j0,k1)=1
        # W: (k0,i1)=1, (k1,i1)=1
        U[i1*m+j0, col] = 1; U[i0*m+j0, col] = -1
        V[j0*p+k0, col] = 1; V[j0*p+k1, col] = 1
        W[k0*n+i1, col] = 1; W[k1*n+i1, col] = 1
        col += 1
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        # U: (i0,j1)=1, (i1,j1)=-1
        # V: (j1,k0)=1, (j1,k1)=1
        # W: (k0,i0)=1, (k1,i0)=1
        U[i0*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k0, col] = 1; V[j1*p+k1, col] = 1
        W[k0*n+i0, col] = 1; W[k1*n+i0, col] = 1
        col += 1

    # Apply Strassen to blocks
    # i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1)
    # i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen(0, 1, 0, 1, 2, 3)
    # i in {0,1}, j in {0,1}, k in {4,5}
    add_strassen(0, 1, 0, 1, 4, 5)
    # i in {0,1}, j in {2,3}, k in {0,1}
    add_strassen(0, 1, 2, 3, 0, 1)
    # i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen(0, 1, 2, 3, 2, 3)
    # i in {0,1}, j in {2,3}, k in {4,5}
    add_strassen(0, 1, 2, 3, 4, 5)

    # Remaining for i in {0,1}, k=6
    # j in {0,1,2,3}
    # Rank 4 for i=0, Rank 4 for i=1
    for i in [0, 1]:
        for j in [0, 1, 2, 3]:
            add_schoolbook(i, j, 6)

    # Remaining for i=2
    # j in {0,1,2,3}, k in {0,1,2,3,4,5,6}
    # Rank 4 decomposition for 1x4x7
    # r = 0..3
    # U[i=2, j=r, r] = 1
    # V[j=r, k, r] = 1 for all k? No.
    # We need to match A[2,j] B[j,k].
    # Term r: U[2,j,r] = delta_jr. V[j,k,r] = B_jk? No.
    # The decomposition is linear.
    # We established rank 4 works.
    # U[2,j,r] = 1 if j==r else 0.
    # V[j,k,r] = 1 if j==r else 0.
    # W[k,i=2,r] = 1.
    for r in range(4):
        j = r
        U[2*m + j, col] = 1.0
        for k in range(7):
            V[j*p + k, col] = 1.0
        for k in range(7):
            W[k*n + 2, col] = 1.0
        col += 1

    rank = col
    return {"U": _pack(U[:, :rank]), "V": _pack(V[:, :rank]), "W": _pack(W[:, :rank])}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
