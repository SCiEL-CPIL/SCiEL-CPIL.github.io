import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin for 120s limit

    n, m, p = 2, 2, 10

    # Matrix multiplication tensor for 2x2 * 2x10
    # U shape (n*m, rank) = (4, rank)
    # V shape (m*p, rank) = (20, rank)
    # W shape (p*n, rank) = (20, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # A is 2x2, B is 2x10, C = A*B is 2x10

    # For 2x2 * 2x2, the rank is 7 (Strassen)
    # For 2x2 * 2x10, we can decompose B into five 2x2 blocks
    # Each 2x2 block needs rank 7, so total rank = 5 * 7 = 35
    # But we can do better by exploiting the shared structure!
    
    # Actually, let me think about this more carefully.
    # C = A * B where A is 2x2 and B is 2x10
    # Let A = [[a11, a12], [a21, a22]]
    # Let B = [[b11, b12, ..., b1,10], [b21, b22, ..., b2,10]]
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # C[0,j] = a11*b1j + a12*b2j
    # C[1,j] = a21*b1j + a22*b2j
    
    # The naive rank is n*m*p = 2*2*10 = 40
    # Using Strassen on 2x2 blocks of B: 5 blocks * 7 = 35
    # But can we do better?
    
    # Actually, for 2x2 * 2xp, the tensor rank is known.
    # For p=2, rank is 7 (Strassen)
    # For p=10, we can think of it as multiplying 2x2 by 10 columns of 2x1
    # Each column is a 2x1 vector, and A * column_j gives a 2x1 result
    
    # The multiplication of 2x2 by 2xp can be seen as:
    # For each column j of B, compute C[:,j] = A * B[:,j]
    # This is 2x2 * 2x1 multiplication, which has rank 3 (since it's just 2 scalar products per row)
    # Wait, 2x2 * 2x1: C[0] = a11*b1 + a12*b2, C[1] = a21*b1 + a22*b2
    # Rank 3: (a11+a21)*(b1), (a12+a22)*(b2), (a11-a21)*(b2-b1) or something similar?
    
    # Actually for 2x2 * 2x1, the rank is 3:
    # C[0] = a11*b1 + a12*b2
    # C[1] = a21*b1 + a22*b2
    # We can write:
    # M1 = (a11+a21)*b1 -> contributes a11*b1 to C[0], a21*b1 to C[1]
    # M2 = (a12+a22)*b2 -> contributes a12*b2 to C[0], a22*b2 to C[1]
    # M3 = (a11-a21-a12+a22)*(b1-b2) -> need to work this out
    
    # Let me just use the block Strassen approach with rank 35, which is known to be optimal for 2x2*2x2 blocks.
    
    # Strassen's 7-term decomposition for 2x2 x 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 2x2 * 2x10, decompose B into 5 blocks of 2x2
    # Each block uses 7 terms, total rank = 35
    
    # Let me construct this properly.
    # A indices: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,2,3
    # B indices: B[0,j] for j=0..9, B[1,j] for j=0..9 -> V indices 0..9 for B[0,:], 10..19 for B[1,:]
    # C indices: C[0,j] for j=0..9, C[1,j] for j=0..9 -> W indices
    #   W[k*n+i] where k is column of C, i is row of C
    #   C[0,0]->W[0], C[0,1]->W[1], ..., C[0,9]->W[9]
    #   C[1,0]->W[10], C[1,1]->W[11], ..., C[1,9]->W[19]
    
    # Wait, the trace convention says W[k*n+i, r]
    # k is the column index of C (0..p-1=0..9), i is the row index of C (0..n-1=0..1)
    # So W[0*2+0]=W[0] is C[0,0], W[0*2+1]=W[1] is C[1,0], W[1*2+0]=W[2] is C[0,1], etc.
    # Actually W[k*n+i]: k is column, i is row
    # C[0,0] -> k=0, i=0 -> W[0]
    # C[1,0] -> k=0, i=1 -> W[1]  
    # C[0,1] -> k=1, i=0 -> W[2]
    # C[1,1] -> k=1, i=1 -> W[3]
    # ...
    # C[0,j] -> k=j, i=0 -> W[2*j]
    # C[1,j] -> k=j, i=1 -> W[2*j+1]
    
    # U[i*m+j, r]: i is row of A (0..1), j is column of A (0..1)
    # U[0] = A[0,0], U[1] = A[0,1], U[2] = A[1,0], U[3] = A[1,1]
    
    # V[j*p+k, r]: j is row of B (0..1), k is column of B (0..9)
    # V[0..9] = B[0,0..9], V[10..19] = B[1,0..9]
    
    # Strassen for block q (columns 2q, 2q+1 of B):
    # B_block = [[B[0,2q], B[0,2q+1]], [B[1,2q], B[1,2q+1]]]
    # C_block = A * B_block, so:
    # C[0,2q] = A[0,0]*B[0,2q] + A[0,1]*B[1,2q]
    # C[0,2q+1] = A[0,0]*B[0,2q+1] + A[0,1]*B[1,2q+1]
    # C[1,2q] = A[1,0]*B[0,2q] + A[1,1]*B[1,2q]
    # C[1,2q+1] = A[1,0]*B[0,2q+1] + A[1,1]*B[1,2q+1]
    
    # Strassen terms for this block:
    # M1 = (A[0,0]+A[1,1])*(B[0,2q]+B[1,2q+1])
    # M2 = (A[1,0]+A[1,1])*B[0,2q]
    # M3 = A[0,0]*(B[1,2q]-B[1,2q+1])
    # M4 = A[1,1]*(B[0,2q+1]-B[1,2q])
    # M5 = (A[0,0]+A[0,1])*B[1,2q+1]
    # M6 = (A[1,0]-A[0,0])*(B[0,2q]+B[0,2q+1])
    # M7 = (A[0,1]-A[1,1])*(B[1,2q]+B[1,2q+1])
    # C[0,2q] = M1 + M4 - M5 + M7
    # C[0,2q+1] = M3 + M5
    # C[1,2q] = M2 + M4
    # C[1,2q+1] = M1 - M2 + M3 + M6
    
    # For each term, U, V, W vectors:
    # Term 1 (M1):
    #   U: A[0,0]=1, A[1,1]=1 -> U = [1, 0, 0, 1]
    #   V: B[0,2q]=1, B[1,2q+1]=1 -> V has 1 at position 2q and 10+2q+1
    #   W: C[0,2q]=1, C[1,2q+1]=1 -> W has 1 at 2*(2q) and 2*(2q+1)+1
    #       W[4q] += 1, W[4q+3] += 1
    
    # Let me build the full rank-35 decomposition.
    
    R = 35
    U = np.zeros((n * m, R), dtype=float)  # (4, 35)
    V = np.zeros((m * p, R), dtype=float)  # (20, 35)
    W = np.zeros((p * n, R), dtype=float)  # (20, 35)
    
    # Strassen coefficients: (u_vec, v_vec, w_vec)
    # u_vec = [A[0,0], A[0,1], A[1,0], A[1,1]]
    # v_vec = [B[0,2q], B[0,2q+1], B[1,2q], B[1,2q+1]] (for the block)
    # w_vec = [C[0,2q], C[0,2q+1], C[1,2q], C[1,2q+1]] (for the block)
    
    strassen = [
        # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A10+A11)B00 -> C10 += 1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3: A00(B10-B11) -> C01 += 1, C11 += 1
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        # M4: A11(B01-B10) -> C00 += 1, C10 += 1
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
        # M5: (A00+A01)B11 -> C00 -= 1, C01 += 1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: (A10-A00)(B00+B01) -> C11 += 1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (A01-A11)(B10+B11) -> C00 += 1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for q in range(5):  # 5 blocks of 2 columns
        for u_vec, v_vec, w_vec in strassen:
            # U: 4 elements
            U[0, col] = u_vec[0]
            U[1, col] = u_vec[1]
            U[2, col] = u_vec[2]
            U[3, col] = u_vec[3]
            
            # V: 20 elements, only 4 non-zero for this block
            V[2*q, col] = v_vec[0]       # B[0, 2q]
            V[2*q+1, col] = v_vec[1]     # B[0, 2q+1]
            V[10+2*q, col] = v_vec[2]    # B[1, 2q]
            V[10+2*q+1, col] = v_vec[3]  # B[1, 2q+1]
            
            # W: 20 elements
            # C[0, 2q] -> W[2*(2q)+0] = W[4q]
            # C[0, 2q+1] -> W[2*(2q+1)+0] = W[4q+2]
            # C[1, 2q] -> W[2*(2q)+1] = W[4q+1]
            # C[1, 2q+1] -> W[2*(2q+1)+1] = W[4q+3]
            W[4*q, col] = w_vec[0]
            W[4*q+2, col] = w_vec[1]
            W[4*q+1, col] = w_vec[2]
            W[4*q+3, col] = w_vec[3]
            
            col += 1
    
    # Verify
    def verify(U, V, W, n=2, m=2, p=10, n_trials=50):
        R = U.shape[1]
        T = np.zeros((n, m, p))
        for r in range(R):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
        
        np.random.seed(42)
        for _ in range(n_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.einsum('ijk,ij,jk->ik', T, A, B)
            if not np.allclose(C_computed, C_expected, atol=1e-10, rtol=1e-10):
                return False
        return True
    
    is_valid = verify(U, V, W)
    
    if not is_valid:
        # Fallback: naive rank-40
        R_naive = n * m * p
        U_naive = np.zeros((n * m, R_naive), dtype=float)
        V_naive = np.zeros((m * p, R_naive), dtype=float)
        W_naive = np.zeros((p * n, R_naive), dtype=float)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i * m + j, idx] = 1.0
                    V_naive[j * p + k, idx] = 1.0
                    W_naive[k * n + i, idx] = 1.0
                    idx += 1
        
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
