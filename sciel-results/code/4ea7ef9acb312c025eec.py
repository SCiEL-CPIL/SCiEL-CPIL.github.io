"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the 7 rank-1 terms for 2x2 matrix multiplication.
    Each term is (u, v, w) where:
      u[i,j] coefficient for A[i,j]
      v[j,k] coefficient for B[j,k]
      w[i,k] coefficient for C[i,k]
    The product sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] should equal C[i,k] = sum_j A[i,j]*B[j,k].
    
    We use the standard Strassen 7-term formula.
    """
    # Indices: i,j,k in {0,1}
    # A is 2x2, B is 2x2, C = A*B is 2x2
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen's 7 products:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11) B00
    # P3 = A00 (B01 - B11)
    # P4 = A11 (B10 - B00)
    # P5 = (A00 + A01) B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each term r has u_r (2x2 for A), v_r (2x2 for B), w_r (2x2 for C)
    # such that sum_r u_r[i,j]*v_r[j,k]*w_r[i,k] = C[i,k]
    
    terms = []
    
    # P1 = (A00+A11)(B00+B11)
    # u: A00+A11 => u[0,0]=1, u[1,1]=1
    # v: B00+B11 => v[0,0]=1, v[1,1]=1
    # w: contributes to C00 with +1, C11 with +1
    # C00 gets +P1, C11 gets +P1
    terms.append({
        'u': {(0,0):1, (1,1):1},
        'v': {(0,0):1, (1,1):1},
        'w': {(0,0):1, (1,1):1}
    })
    
    # P2 = (A10+A11)B00
    # u: A10+A11 => u[1,0]=1, u[1,1]=1
    # v: B00 => v[0,0]=1
    # w: C10 gets +P2, C11 gets -P2
    terms.append({
        'u': {(1,0):1, (1,1):1},
        'v': {(0,0):1},
        'w': {(1,0):1, (1,1):-1}
    })
    
    # P3 = A00(B01-B11)
    # u: A00 => u[0,0]=1
    # v: B01-B11 => v[0,1]=1, v[1,1]=-1
    # w: C01 gets +P3, C11 gets +P3
    terms.append({
        'u': {(0,0):1},
        'v': {(0,1):1, (1,1):-1},
        'w': {(0,1):1, (1,1):1}
    })
    
    # P4 = A11(B10-B00)
    # u: A11 => u[1,1]=1
    # v: B10-B00 => v[1,0]=1, v[0,0]=-1
    # w: C00 gets +P4, C10 gets +P4
    terms.append({
        'u': {(1,1):1},
        'v': {(1,0):1, (0,0):-1},
        'w': {(0,0):1, (1,0):1}
    })
    
    # P5 = (A00+A01)B11
    # u: A00+A01 => u[0,0]=1, u[0,1]=1
    # v: B11 => v[1,1]=1
    # w: C00 gets -P5, C01 gets +P5
    terms.append({
        'u': {(0,0):1, (0,1):1},
        'v': {(1,1):1},
        'w': {(0,0):-1, (0,1):1}
    })
    
    # P6 = (A10-A00)(B00+B01)
    # u: A10-A00 => u[1,0]=1, u[0,0]=-1
    # v: B00+B01 => v[0,0]=1, v[0,1]=1
    # w: C11 gets +P6
    terms.append({
        'u': {(1,0):1, (0,0):-1},
        'v': {(0,0):1, (0,1):1},
        'w': {(1,1):1}
    })
    
    # P7 = (A01-A11)(B10+B11)
    # u: A01-A11 => u[0,1]=1, u[1,1]=-1
    # v: B10+B11 => v[1,0]=1, v[1,1]=1
    # w: C00 gets +P7
    terms.append({
        'u': {(0,1):1, (1,1):-1},
        'v': {(1,0):1, (1,1):1},
        'w': {(0,0):1}
    })
    
    return terms

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # We use Strassen on the 2x2x2 core, then extend to 2x2x15.
    # The matrix multiplication is C = A * B where A is 2x2, B is 2x15, C is 2x15.
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1}, k in {0,...,14}
    
    # We can think of B as 15 columns, each a 2x1 vector.
    # For each column k, we compute C[:,k] = A * B[:,k] using Strassen with 7 multiplications.
    # But Strassen is for 2x2 * 2x2. For 2x2 * 2x1, we can use a variant.
    
    # Actually, let's use the direct approach: for each pair (i,k), C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # That's 2*2*15 = 60 terms for schoolbook.
    
    # With Strassen-like decomposition for 2x2 * 2x1:
    # We can use 4 terms per column (since we only need 2 outputs per column, and Strassen saves for 2x2*2x2).
    # Actually for 2x2 * 2x1, the minimal rank is 4 (since it's just two dot products of length 2).
    # Wait, C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] and C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # These are independent, so we need 4 terms per column? No, each term is rank-1 in the tensor.
    
    # Let me think more carefully. The tensor T[i,j,k] = 1 if the multiplication C[i,k] = sum_j A[i,j]B[j,k] uses this.
    # Actually T is defined by: for each (i,j,k), the coefficient is 1 if j is the summation index.
    # T[i,j,k] = 1 for all i,j,k? No.
    
    # The matrix multiplication tensor for (n,m,p) = (2,2,15) has entries:
    # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k]).
    # Wait, that's not right either. The tensor is T[i,j,k] = delta_{i,i} * delta_{j,j} * ... 
    # Actually, the standard matrix multiplication tensor is T[i,j,k] = 1 if the indices match in the right way.
    # For C[i,k] = sum_j A[i,j] B[j,k], the tensor has T[i,j,k] = 1 for all valid i,j,k.
    # So it's a 2x2x15 tensor with all entries 1.
    
    # Rank of an all-ones tensor of size 2x2x15:
    # This can be decomposed as sum of rank-1 tensors.
    # A rank-1 tensor is u[i]*v[j]*w[k].
    # We need sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k.
    
    # This is equivalent to finding the minimum r such that there exist vectors u_r, v_r, w_r
    # with sum_r u_r[i] v_r[j] w_r[k] = 1 for all i,j,k.
    
    # One simple decomposition: use r = 4 terms.
    # Let's try:
    # Term 1: u=[1,0], v=[1,0], w=[1,1,...,1] (all ones, length 15)
    # Term 2: u=[0,1], v=[1,0], w=[1,1,...,1]
    # Term 3: u=[1,0], v=[0,1], w=[1,1,...,1]
    # Term 4: u=[0,1], v=[0,1], w=[1,1,...,1]
    # Sum = (u1[i]*v1[j] + u2[i]*v2[j] + u3[i]*v3[j] + u4[i]*v4[j]) * w[k]
    # For i=0,j=0: 1*1 + 0*1 + 1*0 + 0*0 = 1, times w[k]=1 => 1. Good.
    # For i=0,j=1: 1*0 + 0*0 + 1*1 + 0*0 = 1, times w[k]=1 => 1. Good.
    # For i=1,j=0: 0*1 + 1*1 + 0*0 + 0*0 = 1, times w[k]=1 => 1. Good.
    # For i=1,j=1: 0*0 + 1*0 + 0*1 + 1*1 = 1, times w[k]=1 => 1. Good.
    
    # So rank 4 works! Let's verify the interface.
    
    # U has shape (n*m, rank) = (4, 4)
    # V has shape (m*p, rank) = (30, 4)
    # W has shape (p*n, rank) = (30, 4)
    
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # For our decomposition:
    # r=0: u=[1,0] for A indices, v=[1,0] for B row indices, w=[1]*15 for k
    #   U[0*2+0, 0] = 1 (i=0,j=0)
    #   U[1*2+0, 0] = 0 (i=1,j=0)
    #   U[0*2+1, 0] = 0 (i=0,j=1)
    #   U[1*2+1, 0] = 0 (i=1,j=1)
    #   V[0*15+0, 0] = 1 (j=0,k=0)
    #   V[0*15+1, 0] = 1 (j=0,k=1) ... all k for j=0
    #   V[1*15+k, 0] = 0 for all k
    #   W[0*2+0, 0] = 1 (k=0,i=0)
    #   W[0*2+1, 0] = 1 (k=0,i=1) ... all k for i=0
    #   W[k*2+1, 0] = 1 for all k (i=1)
    
    # Let's build it properly.
    
    rank = 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 0: i=0, j=0, all k
    U[0 * m + 0][0] = 1  # i=0, j=0
    for k in range(p):
        V[0 * p + k][0] = 1  # j=0, all k
        W[k * n + 0][0] = 1  # all k, i=0
        W[k * n + 1][0] = 1  # all k, i=1
    
    # Term 1: i=1, j=0, all k
    U[1 * m + 0][1] = 1  # i=1, j=0
    for k in range(p):
        V[0 * p + k][1] = 1  # j=0, all k
        W[k * n + 0][1] = 1  # all k, i=0
        W[k * n + 1][1] = 1  # all k, i=1
    
    # Term 2: i=0, j=1, all k
    U[0 * m + 1][2] = 1  # i=0, j=1
    for k in range(p):
        V[1 * p + k][2] = 1  # j=1, all k
        W[k * n + 0][2] = 1  # all k, i=0
        W[k * n + 1][2] = 1  # all k, i=1
    
    # Term 3: i=1, j=1, all k
    U[1 * m + 1][3] = 1  # i=1, j=1
    for k in range(p):
        V[1 * p + k][3] = 1  # j=1, all k
        W[k * n + 0][3] = 1  # all k, i=0
        W[k * n + 1][3] = 1  # all k, i=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
