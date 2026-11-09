import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def _check(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each rank-1 term r: U[:,r] * V[:,r] * W[:,r]
    We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    where i in [0,n), j in [0,m), k in [0,p)
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i
    """
    rank = U.shape[1]
    # Build the tensor via contraction
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need rank-minimal decomposition with integer/half-integer entries
    
    # Known: schoolbook rank = n*m*p = 30
    # Strassen-based: 7 + 2*3*3 + 2*1*5 = 7 + 18 + 10 = 35 (worse)
    # Better: Use Strassen for the 2x2x2 sub-blocks and schoolbook for the rest
    # 2x2x2 Strassen: rank 7
    # For 2x3x5: split B into 3 blocks of 2x2 and 2x1
    # Actually let's think more carefully.
    
    # The tensor is T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}
    # This is the outer product of all-ones vectors: T = 1_i * 1_j * 1_k
    # Wait, that's not right. The matrix multiplication tensor has T[i,j,k]=1.
    # This means C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is indeed the all-ones tensor of shape (2,3,5).
    # The rank of the all-ones tensor of shape (n,m,p) is... 
    # Actually for matrix multiplication, the tensor is T[i,j,k] = delta_{i,j,k} in the standard formulation.
    # Let me re-read: "multiplying a 2 by 3 matrix by a 3 by 5 matrix"
    # C = A * B where A is 2x3, B is 3x5, C is 2x5
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k
    # This is the all-ones tensor!
    # The rank of the all-ones tensor of shape (n,m,p) is 1 (it's rank 1 as a tensor)
    # Wait no. The tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor T is defined such that C = T(A,B) where C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # For matrix multiplication, T[i,j,k] = 1 if the indices are consistent, which they always are.
    # So T[i,j,k] = 1 for all i,j,k. This IS the all-ones tensor.
    # But the rank of the all-ones tensor is 1! Just use one term: U=all ones, V=all ones, W=all ones.
    # Wait, that gives C[i,k] = (sum_j A[i,j]) * (sum_k B[j,k])... no.
    # Let me reconsider. The tensor decomposition for matrix multiplication:
    # C[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]... 
    # Actually the standard formulation is:
    # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k V[j,k,r] B[j,k]) ... no that's not right either.
    # 
    # The correct formulation: C[i,k] = sum_{j,r} T[i,j,k,r] A[i,j] B[j,k]
    # where T is the multiplication tensor. For standard matrix mult, T[i,j,k,r] = 1 if r corresponds to j.
    # 
    # In the rank-1 decomposition form:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k]) ... 
    # Hmm, I'm confusing myself. Let me use the standard tensor rank definition.
    #
    # The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 for all i,j,k.
    # A rank-r decomposition is: M[i,j,k] = sum_{r=1}^r u_r[i] v_r[j] w_r[k]
    # where u_r is a vector of length n, v_r of length m, w_r of length p.
    #
    # But wait, in the interface, U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # This is a different parameterization. Let me think about what this means.
    #
    # Given the shapes:
    # U: (n*m, rank) - indexed by (i,j) pairs
    # V: (m*p, rank) - indexed by (j,k) pairs  
    # W: (p*n, rank) - indexed by (k,i) pairs
    #
    # The decomposition must satisfy: for all i,j,k:
    # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # This is exactly the tensor rank decomposition of the all-ones tensor!
    # The all-ones tensor of shape (2,3,5) has tensor rank 1.
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then sum_r U*V*W = 1*1*1 = 1. 
    #
    # But wait, that seems too easy. Let me re-read the problem.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix"
    # 
    # I think the issue is that the tensor is NOT the all-ones tensor. The matrix multiplication
    # tensor has a specific structure. Let me reconsider.
    #
    # Actually, I believe the tensor for matrix multiplication A(n,m) * B(m,p) -> C(n,p) is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This IS the all-ones tensor, and its rank is 1.
    #
    # But the known result is that the rank of 2x3x5 matrix multiplication is 29 (or some specific value).
    # So I must be misunderstanding the tensor structure.
    #
    # Let me look at this differently. Perhaps the tensor is:
    # T[i,j,k] = 1 if the indices form a valid "path" i->j->k, which they always do.
    # So it is indeed the all-ones tensor.
    #
    # Unless... the problem is using a different convention where the tensor encodes the 
    # multiplication structure more explicitly. 
    #
    # Given that the current program has rank 29 and scores 1.0345, and the schoolbook is 30,
    # the rank must be 29. The all-ones tensor rank 1 interpretation must be wrong for this formulation.
    #
    # Let me just go with the known rank 29 decomposition and verify it works.
    
    # Rank 29 decomposition (known optimal or near-optimal for 2x3x5)
    rank = 29
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Part 1: Strassen for the 2x2x2 sub-problem (i in {0,1}, j in {0,1}, k in {0,1})
    # A indices: i*m+j -> 3i+j for i in {0,1}, j in {0,1}
    # B indices: j*p+k -> 5j+k for j in {0,1}, k in {0,1}
    # C indices: k*n+i -> 2k+i for k in {0,1}, i in {0,1}
    
    # Strassen 7 terms for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M4 - M7 - M1  ... wait, let me use the standard Strassen
    # Actually standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00-B10)
    # M3 = (A00-A10)(B10+B11)
    # M4 = (A01-A11)(B00+B10)
    # M5 = (A00+A01)(B11-B01)
    # M6 = (A10+A11)(B01-B11)
    # M7 = (A00-A10)(B00-B10)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3 - M4 + M5
    # C10 = -M2 + M3 + M6 - M7
    # C11 = M1 - M3 + M2 + M6
    #
    # Let me use a simpler verification approach. I'll use the known rank-29 construction.
    # For 2x3x5, one known approach:
    # - Use Strassen (rank 7) for the 2x2x2 block (j in {0,1}, k in {0,1})
    # - Use schoolbook for remaining blocks
    # - Total: 7 + 2*3*3 + 2*1*5 - overlaps... 
    #
    # Actually, let me just use a direct construction and verify.
    
    # Direct approach: We need sum_r U[3i+j,r] * V[5j+k,r] * W[2k+i,r] = 1 for all i,j,k
    
    # Let me try a construction based on the observation that this is related to 
    # the tensor rank of 2x3x5 matrix multiplication.
    
    # Known result: rank(2x3x5) <= 29
    # Construction:
    # Split j into {0,1,2} and k into {0,1,2,3,4}
    # For j=0,1: use Strassen-style for the 2x2x5 part
    # For j=2: use schoolbook for the 2x1x5 part
    
    # 2x2x5: Can we do better than schoolbook (2*2*5=20)?
    # 2x2x2 has rank 7, 2x2x3 has rank ?, etc.
    # 2x2xp: known results... 2x2x2=7, 2x2x3=10 or 11?
    # 
    # Let me just use the schoolbook for simplicity and get rank 30, then try to optimize.
    # Actually, the current program has rank 29, so let me reproduce that.
    
    # From the current program's structure:
    # Part 1: 7 terms (Strassen for 2x2x2, j in {0,1}, k in {0,1})
    # Part 2: 2*2*3 = 12 terms (schoolbook for j in {0,1}, k in {2,3,4})
    # Part 3: 2*1*5 = 10 terms (schoolbook for j=2, k in {0..4})
    # Total: 7 + 12 + 10 = 29
    
    col = 0
    # Part 1: Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    # Using the standard Strassen decomposition
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1  # A[0,0] and A[1,1]
    V[0][col] = 1; V[6][col] = 1  # B[0,0] and B[1,1]
    W[0][col] = 1; W[3][col] = 1  # C[0,0] and C[1,1]
    col += 1
    
    # M2 = (A10+A11)(B00-B10)
    U[3][col] = 1; U[4][col] = 1  # A[1,0] and A[1,1]
    V[0][col] = 1; V[5][col] = -1  # B[0,0] and B[1,0]
    W[2][col] = 1; W[0][col] = -1  # C[1,0] and C[0,0]
    col += 1
    
    # M3 = (A00-A10)(B10+B11)
    U[0][col] = 1; U[3][col] = -1  # A[0,0] and A[1,0]
    V[5][col] = 1; V[6][col] = 1  # B[1,0] and B[1,1]
    W[2][col] = 1; W[3][col] = 1  # C[1,0] and C[1,1]
    col += 1
    
    # M4 = (A01-A11)(B00+B10)
    U[1][col] = 1; U[4][col] = -1  # A[0,1] and A[1,1]
    V[0][col] = 1; V[5][col] = 1  # B[0,0] and B[1,0]
    W[0][col] = 1; W[2][col] = 1  # C[0,0] and C[1,0]
    col += 1
    
    # M5 = (A00+A01)(B11-B01)
    U[0][col] = 1; U[1][col] = 1  # A[0,0] and A[0,1]
    V[6][col] = 1; V[1][col] = -1  # B[1,1] and B[0,1]
    W[0][col] = 1; W[3][col] = -1  # C[0,0] and C[1,1]
    col += 1
    
    # M6 = (A10+A11)(B01-B11)
    U[3][col] = 1; U[4][col] = 1  # A[1,0] and A[1,1]
    V[1][col] = 1; V[6][col] = -1  # B[0,1] and B[1,1]
    W[2][col] = 1; W[3][col] = -1  # C[1,0] and C[1,1]
    col += 1
    
    # M7 = (A00-A10)(B00-B10)
    U[0][col] = 1; U[3][col] = -1  # A[0,0] and A[1,0]
    V[0][col] = 1; V[5][col] = -1  # B[0,0] and B[1,0]
    W[0][col] = 1; W[2][col] = -1  # C[0,0] and C[1,0]
    col += 1
    
    # Verify Strassen part: for i in {0,1}, j in {0,1}, k in {0,1}
    # sum_{r=0}^{6} U[3i+j,r] * V[5j+k,r] * W[2k+i,r] should = 1
    
    # Part 2: Schoolbook for i in {0,1}, j in {0,1}, k in {2,3,4}
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[3*i + j][col] = 1
                V[5*j + k][col] = 1
                W[2*k + i][col] = 1
                col += 1
    
    # Part 3: Schoolbook for i in {0,1}, j=2, k in {0,1,2,3,4}
    for i in range(2):
        for k in range(5):
            U[3*i + 2][col] = 1
            V[5*2 + k][col] = 1
            W[2*k + i][col] = 1
            col += 1
    
    # Verify the full decomposition
    if not _check(U, V, W, n, m, p):
        # Fallback to schoolbook rank 30
        rank = 30
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
