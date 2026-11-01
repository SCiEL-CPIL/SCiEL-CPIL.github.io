"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W):
    """Verify exact decomposition. Returns True if valid."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check shape
    if U.shape != (n*m, U.shape[1]) or V.shape != (m*p, V.shape[1]) or W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    # The tensor T[i*m+j, j'*p+k, k'*n+i'] should be decomposable as
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # For matrix multiplication 2x2 * 2x3, the tensor is:
    # T[i*m+j, j2*p+k, k2*n+i2] = 1 if i==i2, j==j2, k==k2, else 0
    # Wait, let me think about this more carefully.
    # C = A @ B, where A is 2x2, B is 2x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form: C[i,k] = sum_{i',j',j'',k'} A[i',j'] * B[j'',k'] * T[i',j',j'',k',i,k]
    # For this to equal sum_j A[i,j]*B[j,k], we need T[i',j',j'',k',i,k] = 1 iff i'=i, j'=j, j''=j, k'=k, else 0
    # In the decomposition: T[a,b,c] where a=i'*m+j', b=j''*p+k', c=k'*n+i (trace convention)
    # Actually the trace convention is W[k*n+i, r]
    # So C[i,k] = sum_r (sum_{i',j'} A[i',j'] * U[i'*m+j', r]) * (sum_{j'',k'} V[j''*p+k', r] * B[j'',k']) * W[k*n+i, r]
    # = sum_r W[k*n+i, r] * (sum_{i',j'} A[i',j'] U[i'*m+j',r]) * (sum_{j'',k'} V[j''*p+k',r] B[j'',k'])
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r U[i'*m+j', r] * V[j''*p+k', r] * W[k*n+i, r] = delta_{i',i} * delta_{j',j''} * delta_{k',k}
    
    # Let's build the full tensor and check
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    # Check entry (i*m+j, j2*p+k, k*n+i) in the decomposed tensor
                    # Actually, the tensor index is (a, b, c) where:
                    # a = i'*m+j' (input A index)
                    # b = j''*p+k' (input B index)  
                    # c = k*n+i (output C index, trace convention)
                    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                    # Should equal 1 if a's (i',j') = (i,j) AND b's (j'',k') = (j,k) ... 
                    # Wait, I need to be more careful.
                    pass
    
    # Simpler check: for each r, the outer product structure
    # Actually, let's just verify by computing C from A,B using the decomposition
    # and comparing to direct multiplication for random inputs.
    np.random.seed(42)
    for trial in range(20):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_direct = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            A_vec = U[:, r]  # shape (n*m,)
            B_vec = V[:, r]  # shape (m*p,)
            W_vec = W[:, r]  # shape (p*n,)
            A_mat = A_vec.reshape(n, m)
            B_mat = B_vec.reshape(m, p)
            scalar_A = np.sum(A * A_mat)  # <A, A_r>
            scalar_B = np.sum(B * B_mat)  # <B, B_r>
            W_mat = W_vec.reshape(p, n)  # trace convention: W[k*n+i] -> W_mat[k,i]
            C_decomp += scalar_A * scalar_B * W_mat.T  # W_mat.T[i,k]
        if not np.allclose(C_direct, C_decomp, atol=1e-9):
            return False
    return True

def _try_rank_11():
    """Try to find a rank-11 decomposition using the structure of 2x2x2."""
    n, m, p = DIMENSIONS
    # Start with known rank-7 for 2x2x2 (Strassen-like)
    # Actually, the optimal rank for 2x2x2 is 7 (Strassen)
    # For 2x2x3, we can extend by handling the third column of B separately
    
    # Strategy: Use rank-7 for the first two columns of B (treating B as 2x2 x 2),
    # then add 4 more terms for the third column? That would give rank 11.
    # But we can do better by sharing some terms.
    
    # Actually, let's think about it differently.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For k=0,1: this is like multiplying 2x2 by 2x2, rank 7 suffices
    # For k=2: C[i,2] = sum_j A[i,j] * B[j,2], which is a 2x1 output
    # The terms for k=2 involve A[i,0]*B[0,2] and A[i,1]*B[1,2]
    
    # Let me try a different approach: construct a rank-11 decomposition directly.
    # For 2x2x3, the tensor has 2*2*2*3 = 24 nonzero entries (one for each i,j,k)
    # Rank-11 means 11 rank-1 tensors.
    
    # Let's use the following structure based on Strassen's idea:
    # For 2x2x2, Strassen uses 7 products:
    # M1 = (A11)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B21+B22)
    # M4 = (A22)(B11+B12)
    # M5 = (A11+A12)(B22)
    # M6 = (A12-A22)(B21+B22)
    # M7 = (A11-A21)(B11+B12)
    
    # For the third column, we need:
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    
    # We can add terms:
    # M8 = A[0,0]*B[0,2]
    # M9 = A[1,0]*B[0,2]
    # M10 = A[0,1]*B[1,2]
    # M11 = A[1,1]*B[1,2]
    
    # But wait, some of these might be combinable with existing Strassen terms.
    # Let me just construct it explicitly and verify.
    
    rank = 11
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen terms for columns 0,1 of B (indices in B: j*p+k, so j=0,k=0 -> 0; j=0,k=1->1; j=1,k=0->2; j=1,k=1->3)
    # A indices: i*m+j: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # W indices: k*n+i: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # M1 = A[0,0] * (B[0,0]+B[1,1])
    U[0, 0] = 1; V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1  # C[0,0] += 1, C[1,0] -= 1? No wait.
    # C[0,0] gets +1, C[1,1] gets +1? Let me be careful.
    # M1 contributes to C[0,0] and C[1,1]? No.
    # In Strassen for C = A*B (2x2):
    # C[0,0] = M1 - M3 + M7
    # C[0,1] = M3 - M5 - M7
    # C[1,0] = M2 + M4 - M6
    # C[1,1] = -M1 + M4 + M5 + M6
    
    # So for M1 = A[0,0]*(B[0,0]+B[1,1]):
    # It contributes +1 to C[0,0] and -1 to C[1,1]
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = -1 # C[1,1]
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1
    W[2, 1] = 1  # C[1,0]
    
    # M3 = A[0,0] * (B[1,0]+B[1,1])
    U[0, 2] = 1; V[2, 2] = 1; V[3, 2] = 1
    W[0, 2] = -1 # C[0,0]
    W[1, 2] = 1  # C[0,1]
    
    # M4 = A[1,1] * (B[0,0]+B[0,1])
    U[3, 3] = 1; V[0, 3] = 1; V[1, 3] = 1
    W[2, 3] = 1  # C[1,0]
    W[3, 3] = 1  # C[1,1]
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1
    W[1, 4] = -1 # C[0,1]
    W[3, 4] = 1  # C[1,1]
    
    # M6 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[1, 5] = 1; U[3, 5] = -1; V[2, 5] = 1; V[3, 5] = 1
    W[2, 5] = -1 # C[1,0]
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (A[0,0]-A[1,0]) * (B[0,0]+B[0,1])
    U[0, 6] = 1; U[2, 6] = -1; V[0, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1  # C[0,0]
    W[1, 6] = -1 # C[0,1]
    
    # Now for the third column (k=2): B indices are j*p+2 = j*3+2: j=0->2, j=1->5
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # M8 = A[0,0]*B[0,2]
    U[0, 7] = 1; V[2, 7] = 1  # B[0,2] index is 0*3+2=2
    W[5, 7] = 1  # C[0,2] index is 2*2+0=4? Wait, W[k*n+i]: k=2,i=0 -> 2*2+0=4
    
    # Let me recheck: W[k*n+i], so for C[i,k], the index is k*n+i.
    # C[0,2]: k=2, i=0 -> 2*2+0 = 4
    # C[1,2]: k=2, i=1 -> 2*2+1 = 5
    W[4, 7] = 1  # C[0,2]
    
    # M9 = A[0,1]*B[1,2]
    U[1, 8] = 1; V[5, 8] = 1  # B[1,2] index is 1*3+2=5
    W[4, 8] = 1  # C[0,2]
    
    # M10 = A[1,0]*B[0,2]
    U[2, 9] = 1; V[2, 9] = 1
    W[5, 9] = 1  # C[1,2]
    
    # M11 = A[1,1]*B[1,2]
    U[3, 10] = 1; V[5, 10] = 1
    W[5, 10] = 1  # C[1,2]
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if _verify(U, V, W):
        return result
    
    # If verification fails, try adjusting signs
    # Let me re-derive more carefully by checking each term
    
    # Actually, let me just return None and fall back
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try rank 11 first
    r11 = _try_rank_11()
    if r11 is not None:
        return r11
    
    # If that fails, use schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
