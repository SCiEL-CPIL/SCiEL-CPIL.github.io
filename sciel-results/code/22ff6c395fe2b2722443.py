"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(decomp, tol=1e-9):
    """Verify that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == delta_{i,k} for all i,j,k."""
    n, m, p = DIMENSIONS
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                expected = 1.0 if i == k else 0.0
                if abs(s - expected) > tol:
                    return False
    return True


def _rank11_strassen_block():
    """
    Rank 11 decomposition: Strassen (rank 7) for the 2x2 block (i=0, j=0, k=0,1)
    plus schoolbook (rank 4) for k=2.
    
    The tensor identity is:
      For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k}
    
    For the Strassen part (columns 0-6), we need:
      sum_{r=0}^{6} U[i, r] * V[j, r] * W[k, r] = delta_{i,k} for i in {0,1}, j in {0,1}, k in {0,1}
      where U is 4x7 (i*2+j), V is 4x7 (j*2+k), W is 4x7 (k*2+i)
    
    For the schoolbook part (columns 7-10), we need:
      sum_{r=7}^{10} U[i*2+j, r] * V[j*2+2, r] * W[2*2+i, r] = 1 for all i,j
      And for the cross terms (k != 2 in Strassen part, or k != 0,1 in schoolbook part), contributions should be 0.
    
    Standard Strassen for 2x2*2x2:
      C = A*B where A, B, C are 2x2.
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B10-B11)
      M4 = A11*(B01-B10)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B10)
      M7 = (A01-A11)*(B01+B11)
      C00 = M1 - M4 + M5 + M7
      C01 = M3 + M4
      C10 = M2 + M3
      C11 = M1 - M2 + M6 + M7
    
    Let me define the matrices carefully.
    U has rows indexed by (i,j) where i=0,1 (row of A) and j=0,1 (col of A).
    Row 0: (i=0,j=0) = A00
    Row 1: (i=0,j=1) = A01
    Row 2: (i=1,j=0) = A10
    Row 3: (i=1,j=1) = A11
    
    V has rows indexed by (j,k) where j=0,1 (col of B) and k=0,1 (row of B).
    Row 0: (j=0,k=0) = B00
    Row 1: (j=0,k=1) = B01
    Row 2: (j=1,k=0) = B10
    Row 3: (j=1,k=1) = B11
    
    W has rows indexed by (k,i) where k=0,1 (row of C) and i=0,1 (col of C).
    Row 0: (k=0,i=0) = C00
    Row 1: (k=0,i=1) = C01
    Row 2: (k=1,i=0) = C10
    Row 3: (k=1,i=1) = C11
    
    The tensor entries T[i,j,k] = delta_{i,k} for the multiplication of 2x2 by 2x2.
    
    We want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] = delta_{i,k}.
    
    For Strassen:
    r=0: M1 = (A00+A11)(B00+B11)
      U: A00=1, A01=0, A10=0, A11=1
      V: B00=1, B01=0, B10=0, B11=1
      W: C00=1, C01=0, C10=0, C11=-1
      Check: For (i=0,j=0,k=0): U[0][0]*V[0][0]*W[0][0] = 1*1*1 = 1. T[0,0,0]=1. Good.
      For (i=0,j=0,k=1): U[0][0]*V[1][0]*W[1][0] = 1*0*0 = 0. T[0,0,1]=0. Good.
      For (i=0,j=1,k=0): U[1][0]*V[0][0]*W[0][0] = 0*1*1 = 0. T[0,1,0]=0. Good.
      For (i=1,j=0,k=0): U[2][0]*V[0][0]*W[2][0] = 0*1*0 = 0. T[1,0,0]=0. Good.
      For (i=1,j=0,k=1): U[2][0]*V[1][0]*W[2][0] = 0*0*0 = 0. T[1,0,1]=0. Good.
      For (i=1,j=1,k=0): U[3][0]*V[0][0]*W[0][0] = 1*1*1 = 1. T[1,1,0]=0. BAD!
      
      Hmm, that's not right. Let me reconsider.
      
      The issue is that W[k,i,r] gives the coefficient for C_{k,i}. For M1, the contribution is to C00 and C11.
      C00 gets +M1, C11 gets -M1.
      So W[0][0] = 1 (for C00), W[3][0] = -1 (for C11).
      
      But the tensor entry T[i,j,k] = delta_{i,k} is the coefficient of A_{i,j} * B_{j,k} in C_{k,i}... wait, no.
      
      Actually, the multiplication tensor for (n,m,p) = (2,2,2) is:
      T[i,j,k] where the tensor is sum_{i,j,k} T[i,j,k] a_i b_j c_k = (sum_{i,j} a_i A_{i,j} b_j)(sum_{j,k} B_{j,k} c_k)
      
      Let me re-derive. We have A (2x2) and B (2x2), C = A*B.
      C_{p,q} = sum_r A_{p,r} B_{r,q}
      
      The bilinear form is: C_{p,q} = sum_{r} A_{p,r} B_{r,q}
      
      In the tensor decomposition, we write:
      C_{p,q} = sum_s (sum_{i,j} U_{i,j,s} A_{i,j})(sum_{r,k} V_{r,k,s} B_{r,k})(sum_{p',q'} W_{p',q',s} C_{p',q'})
      
      Wait, that's not quite right either. The standard formulation is:
      
      The multiplication tensor T is a 3-tensor T[i,j,k] such that for matrices A (n x m), B (m x p):
      C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
      
      The tensor decomposition of rank r means there exist matrices U (nm x r), V (mp x r), W (pn x r) such that:
      T[i,j,k] = sum_{s=0}^{r-1} U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
      
      And the identity is: for all A, B, we have C = A*B iff
      C[k*n+i] = sum_{j} A[i*m+j] B[j*p+k]
      
      which is equivalent to:
      sum_{j} A[i*m+j] B[j*p+k] - C[k*n+i] = 0
      
      This is the tensor identity. The tensor T has entries T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[k,i], which it always does, so T[i,j,k] = 1 for all i,j,k? No, that's not right either.
      
      Let me re-read the task description: "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 3 matrix. The output factor follows the official trace convention W[k*n+i,r]."
      
      And the verification in the seed code: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
      
      So the tensor is T[i,j,k] = 1 for all i,j,k? That doesn't make sense for matrix multiplication.
      
      Actually, I think the convention is:
      T[i,j,k] = delta_{i,k} where i is the row of A (and thus the column of C), k is the column of B (and thus the row of C), and j is the shared index.
      
      No wait. Let's look at it differently. The bilinear map is:
      (A, B) -> C = A*B
      
      C[i,k] = sum_j A[i,j] B[j,k]
      
      So the "tensor" that we're decomposing has one entry for each triple (i,j,k) representing the coefficient of A[i,j] * B[j,k] in C[i,k]. This coefficient is 1 for all i,j,k. But that's the identity tensor, not the multiplication tensor.
      
      I think I'm confusing myself. Let me look at the seed code more carefully:
      
      U[i*m+j][column] = 1
      V[j*p+k][column] = 1
      W[k*n+i][column] = 1
      
      So for each (i,j,k), we have a rank-1 term where U picks out A[i,j], V picks out B[j,k], and W picks out C[k*n+i] = C[k,i].
      
      So the tensor T[i,j,k] = 1 for all i,j,k, and the decomposition is:
      sum_s U[i*m+j, s] V[j*p+k, s] W[k*n+i, s] = T[i,j,k] = 1 for all i,j,k
      
      But that's just saying the all-ones tensor has a rank decomposition. That can't be right for matrix multiplication.
      
      Actually, I think the correct interpretation is:
      The multiplication tensor T is defined such that T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[k,i] (note the index order in W: k*n+i means row k, column i of C).
      
      In standard matrix multiplication C = A*B, C[row, col] = sum_mid A[row, mid] * B[mid, col].
      So C[k,i] = sum_j A[k,j] * B[j,i].
      
      Wait, but the seed code has U[i*m+j] (row i, col j of A) and V[j*p+k] (row j, col k of B) and W[k*n+i] (row k, col i of C).
      
      So the tensor entry T[i,j,k] corresponds to the contribution of A[i,j] * B[j,k] to C[k,i].
      
      In C[k,i] = sum_j' A[k,j'] * B[j',i], the term A[i,j] * B[j,k] would contribute if i=k and j=j' and k=i, which is only when i=k and j=i=k. That's not right.
      
      I think the indexing is: A is n x m, B is m x p, C = A*B is n x p.
      C[i,k] = sum_j A[i,j] * B[j,k].
      
      So the contribution of A[i,j] * B[j,k] is to C[i,k].
      
      But W[k*n+i] means row k, column i of C. So W[k*n+i] is C[k,i], not C[i,k].
      
      So the tensor is: T[i,j,k] = coefficient of A[i,j]*B[j,k] in C[k,i].
      This is 1 if i=k (since C[i,i] contains A[i,j]*B[j,i] when k=i)... no, that's not right either.
      
      Let me just trust the seed code's verification logic:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k? Or some other pattern?
      
      Looking at the seed code, it just sets all to 1. The verification I wrote checks T[i,j,k] = delta_{i,k}. Let me reconsider.
      
      Actually, I think the correct tensor for matrix multiplication (n,m,p) is:
      T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
      
      No, that's the all-ones tensor.
      
      OK, I think the issue is that the "tensor decomposition" here is for the bilinear map, and the condition is:
      For all A, B: C = A*B iff for all i,k: C[i,k] = sum_j A[i,j] B[j,k]
      
      This can be written as: sum_j A[i,j] B[j,k] - C[i,k] = 0
      
      The tensor T has entries T[i,j,k] = 1 for the A[i,j]*B[j,k] terms, and the C[i,k] term is separate.
      
      I think the standard formulation for tensor rank of matrix multiplication is:
      T = sum_s u_s v_s w_s where u_s is in (R^n)^* x (R^m)^* (or just a vector of length nm), v_s is a vector of length mp, w_s is a vector of length pn, and the condition is that T[i,j,k] = sum_s u_s[i*m+j] v_s[j*p+k] w_s[k*n+i].
      
      And T[i,j,k] = 1 for all i,j,k if we're decomposing the all-ones tensor, but that's not matrix multiplication.
      
      For matrix multiplication, the tensor is actually the "multiplication tensor" which has T[i,j,k] = 1 if the indices are consistent, i.e., if we're multiplying A (n x m) by B (m x p), then the tensor entry (i,j,k) is 1, and the constraint is that the decomposition gives the correct matrix product.
      
      I'm going in circles. Let me just verify the rank 11 solution numerically in the code and fall back to rank 12 if it fails.
    """
    n, m, p = DIMENSIONS
    # U: 4 x 11, V: 6 x 11, W: 6 x 11
    U = [[0.0] * 11 for _ in range(4)]
    V = [[0.0] * 11 for _ in range(6)]
    W = [[0.0] * 11 for _ in range(6)]
    
    # Strassen part: columns 0-6
    # U rows: (i,j) -> i*2+j
    # V rows: (j,k) -> j*2+k (for k=0,1 only, so rows 0-3)
    # W rows: (k,i) -> k*2+i (for k=0,1 only, so rows 0-3)
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=-1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += -1
    
    # M2 = (A10+A11)B00
    # U: A10=1, A11=1
    # V: B00=1
    # W: C10=1, C11=-1
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] += -1
    
    # M3 = A00(B10-B11)
    # U: A00=1
    # V: B10=1, B11=-1
    # W: C01=1, C10=1
    U[0][2] += 1
    V[2][2] += 1; V[3][2] += -1
    W[1][2] += 1; W[2][2] += 1
    
    # M4 = A11(B01-B10)
    # U: A11=1
    # V: B01=1, B10=-1
    # W: C00=-1, C01=1
    U[3][3] += 1
    V[1][3] += 1; V[2][3] += -1
    W[0][3] += -1; W[1][3] += 1
    
    # M5 = (A00+A01)B11
    # U: A00=1, A01=1
    # V: B11=1
    # W: C00=1, C11=1
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += 1; W[3][4] += 1
    
    # M6 = (A10-A00)(B00+B10)
    # U: A10=1, A00=-1
    # V: B00=1, B10=1
    # W: C11=1
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[2][5] += 1
    W[3][5] += 1
    
    # M7 = (A01-A11)(B01+B11)
    # U: A01=1, A11=-1
    # V: B01=1, B11=1
    # W: C00=1, C11=1
    U[1][6] += 1; U[3][6] += -1
    V[1][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[3][6] += 1
    
    # Schoolbook part for k=2: columns 7-10
    # For each (i,j) in 0..1, 0..1:
    # U[i*2+j, 7+i*2+j] = 1
    # V[j*2+2, 7+i*2+j] = 1  (row j*2+2 since k=2)
    # W[2*2+i, 7+i*2+j] = 1  (row 2*2+i since k=2)
    col = 7
    for i in range(2):
        for j in range(2):
            U[i*2+j][col] = 1
            V[j*2+2][col] = 1
            W[2*2+i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    # Try rank 11 first
    decomp11 = _rank11_strassen_block()
    if _verify(decomp11):
        return decomp11
    
    # Fallback to rank 12
    return _schoolbook()
# EVOLVE-BLOCK-END
