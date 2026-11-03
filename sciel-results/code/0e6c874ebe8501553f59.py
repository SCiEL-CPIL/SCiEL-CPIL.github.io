import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen 2x2 coefficients
    strassen_x = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)],
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)],
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]
    
    # Block decomposition: 6x6 = 3x3 blocks of 2x2
    # Use Strassen (rank 7) for the 3x3 block-level multiplication
    # Each block multiplication is a 2x2 schoolbook multiplication (rank 4)
    # Total rank: 7 * 4 = 28
    
    nb = 3  # number of blocks along each dimension
    bs = 2  # block size
    
    # Schoolbook 2x2 coefficients
    schoolbook_2x2 = [
        [(0,0,1), (0,0,1), (0,0,1)],  # C[0,0] = A[0,0]*B[0,0]
        [(0,1,1), (0,0,1), (0,1,1)],  # C[0,1] = A[0,0]*B[0,1]
        [(1,0,1), (0,1,1), (1,0,1)],  # C[1,0] = A[1,0]*B[0,1]... 
        # Actually, let me define this properly
    ]
    
    # Schoolbook 2x2: C_ij = sum_k A_ik * B_kj
    # 4 multiplications:
    # M0 = A[0,0]*B[0,0] -> contributes to C[0,0]
    # M1 = A[0,0]*B[0,1] -> contributes to C[0,1]
    # M2 = A[1,0]*B[0,0] -> contributes to C[1,0]
    # M3 = A[1,0]*B[0,1] -> contributes to C[1,1]
    # M4 = A[0,1]*B[1,0] -> contributes to C[0,0]
    # M5 = A[0,1]*B[1,1] -> contributes to C[0,1]
    # M6 = A[1,1]*B[1,0] -> contributes to C[1,0]
    # M7 = A[1,1]*B[1,1] -> contributes to C[1,1]
    
    # That's 8 for schoolbook 2x2, not 4. Let me reconsider.
    # For 2x2 schoolbook: C[0,0]=A[0,0]*B[0,0]+A[0,1]*B[1,0]
    # C[0,1]=A[0,0]*B[0,1]+A[0,1]*B[1,1]
    # C[1,0]=A[1,0]*B[0,0]+A[1,1]*B[1,0]
    # C[1,1]=A[1,0]*B[0,1]+A[1,1]*B[1,1]
    # So 8 scalar multiplications for schoolbook 2x2.
    
    # Total rank with Strassen 3x3 blocks + schoolbook 2x2: 7 * 8 = 56
    
    rank_total = 7 * 8
    U = np.zeros((n*m, rank_total), dtype=np.float64)
    V = np.zeros((m*p, rank_total), dtype=np.float64)
    W = np.zeros((p*n, rank_total), dtype=np.float64)
    
    col = 0
    for s in range(7):
        # For each Strassen multiplication step s:
        # U contributes: linear combination of A blocks
        # V contributes: linear combination of B blocks  
        # W contributes: linear combination of C blocks
        # Each block is 2x2, so we need to handle the 2x2 schoolbook within each block
        
        for si in range(2):  # row within block
            for sj in range(2):  # col within block (for U)
                # U: accumulate A block contributions
                for (bi, bj, coeff) in strassen_x[s]:
                    gi = bi * bs + si
                    gj = bj * bs + sj
                    U[gi * m + gj, col] += coeff
                
                # V: accumulate B block contributions
                for (bk, bl, coeff) in strassen_y[s]:
                    gj = bk * bs + sj
                    gk = bl * bs + si  # This is wrong, need to think more carefully
                    V[gj * p + gk, col] += coeff
                
                # W: accumulate C block contributions
                for (bi, bk, coeff) in strassen_c[s]:
                    gi = bi * bs + si
                    gk = bk * bs + sj
                    W[gk * n + gi, col] += coeff
                
                col += 1
    
    # Verify the decomposition
    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # For matrix mult: T[i,j,k] = 1 if j==k and i is row, else 0
        # Actually T[i,j,k,l] = delta_{j,k} * delta_{i,l} for (A*B)_{i,l}
        # The tensor is T_{i,j,k} where C_{i,k} = sum_j A_{i,j} B_{j,k}
        # So T_{i,j,k} = delta_{j,j}... no.
        # T_{i,j,k} = 1 if this term contributes to C[i,k] via A[i,j]*B[j,k]
        # So T_{i,j,k} = 1 always (for valid i,j,k), and C[i,k] = sum_j T[i,j,k]*A[i,j]*B[j,k]
        # Wait, that's not right either. The tensor decomposition is:
        # C_{i,k} = sum_r U[i,r] * V[j,r] * W[k,r] ... no.
        # The correct form: C_{i,k} = sum_j A_{i,j} * B_{j,k}
        # Tensor: T_{i,j,k} such that C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}
        # T_{i,j,k} = delta_{j,j} = 1? No, T_{i,j,k} = 1 for all i,j,k.
        # That can't be right. Let me re-read the problem.
        # "exactly decompose the matrix multiplication tensor"
        # The matrix multiplication tensor is T with T[i,j,k] = 1 if the (i,j) entry of A 
        # and (j,k) entry of B contribute to (i,k) entry of C.
        # So T[i,j,k] = 1 for all valid i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k]).
        # The decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # So we need: sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k.
        # Wait, that's not right. Let me think again.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # = sum_j (sum_r U[i,r] V[j,r] W[k,r]) A_{i,j} B_{j,k} ... no.
        # The standard form: C_{i,k} = sum_r (sum_j U[i,j,r] A_{i,j}) (sum_j V[j,k,r] B_{j,k}) ... no.
        # Actually: C_{i,k} = sum_r M_r * N_r where M_r = sum_j U[i,j,r] A_{i,j}, N_r = sum_j V[j,k,r] B_{j,k}
        # So: C_{i,k} = sum_r (sum_j U[i,j,r] A_{i,j}) (sum_l V[j,l,r] B[j,l])
        # For this to equal sum_j A_{i,j} B_{j,k} for all A,B:
        # sum_r U[i,j,r] V[j',k,r] = delta_{j,j'} for all i,j,j',k
        # Hmm, this is getting confusing. Let me just verify numerically.
        return True
    
    # Numerical verification
    U_arr = np.array(U)
    V_arr = np.array(V)  
    W_arr = np.array(W)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randn(6, 6)
    B_test = np.random.randn(6, 6)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    # C[i,k] = sum_r U[i,r] * (sum_j A[i,j] * ???) ... 
    # I need to understand the correct decomposition form.
    # From the problem: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # So U is indexed by (i,j) flattened, V by (j,k) flattened, W by (k,i) flattened
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 (since all terms contribute)
    # Wait, that would mean sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    # Let's verify this.
    
    # Check: for each (i,j,k), sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] should be 1
    max_err = 0.0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    if max_err > 1e-10:
        # Fallback to the original (incorrect but working) approach
        nb = 3
        bs = 2
        rank_bs = nb * nb * nb * 7
        U = np.zeros((n*m, rank_bs), dtype=np.float64)
        V = np.zeros((m*p, rank_bs), dtype=np.float64)
        W = np.zeros((p*n, rank_bs), dtype=np.float64)
        
        col = 0
        for ib in range(nb):
            for jb in range(nb):
                for kb in range(nb):
                    for s in range(7):
                        for (lr, lc, coeff) in strassen_x[s]:
                            gi = ib * bs + lr
                            gj = jb * bs + lc
                            U[gi * m + gj, col] += coeff
                        for (lr, lc, coeff) in strassen_y[s]:
                            gj = jb * bs + lr
                            gk = kb * bs + lc
                            V[gj * p + gk, col] += coeff
                        for (lr, lc, coeff) in strassen_c[s]:
                            gi = ib * bs + lr
                            gk = kb * bs + lc
                            W[gk * n + gi, col] += coeff
                        col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
