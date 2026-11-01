import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def strassen_block():
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for u in range(2):
            for v in range(4):
                base_col = 7 * (4 * u + v)
                
                a11 = 0 * m + 2 * u
                a12 = 0 * m + 2 * u + 1
                a21 = 1 * m + 2 * u
                a22 = 1 * m + 2 * u + 1
                
                b11 = (2 * u) * p + 2 * v
                b12 = (2 * u) * p + 2 * v + 1
                b21 = (2 * u + 1) * p + 2 * v
                b22 = (2 * u + 1) * p + 2 * v + 1
                
                c11 = (2 * v) * n + 0
                c12 = (2 * v + 1) * n + 0
                c21 = (2 * v) * n + 1
                c22 = (2 * v + 1) * n + 1
                
                U[a11, base_col] += 1; U[a22, base_col] += 1
                V[b11, base_col] += 1; V[b22, base_col] += 1
                W[c11, base_col] += 1; W[c22, base_col] += 1
                
                U[a21, base_col+1] += 1; U[a22, base_col+1] += 1
                V[b11, base_col+1] += 1
                W[c21, base_col+1] += 1; W[c22, base_col+1] -= 1
                
                U[a11, base_col+2] += 1
                V[b12, base_col+2] += 1; V[b22, base_col+2] -= 1
                W[c12, base_col+2] += 1; W[c22, base_col+2] += 1
                
                U[a22, base_col+3] += 1
                V[b21, base_col+3] += 1; V[b11, base_col+3] -= 1
                W[c11, base_col+3] += 1; W[c21, base_col+3] += 1
                
                U[a11, base_col+4] += 1; U[a12, base_col+4] += 1
                V[b22, base_col+4] += 1
                W[c11, base_col+4] -= 1; W[c12, base_col+4] += 1
                
                U[a21, base_col+5] += 1; U[a11, base_col+5] -= 1
                V[b11, base_col+5] += 1; V[b12, base_col+5] += 1
                W[c22, base_col+5] += 1
                
                U[a12, base_col+6] += 1; U[a22, base_col+6] -= 1
                V[b21, base_col+6] += 1; V[b22, base_col+6] += 1
                W[c11, base_col+6] += 1
        
        return U, V, W, rank
    
    def verify_fast(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m+j, :rank]
                    v_row = V[j*p+k, :rank]
                    w_row = W[k*n+i, :rank]
                    val = np.dot(u_row, v_row * w_row)
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True
    
    # Start with Strassen (rank 56)
    U_best, V_best, W_best, rank_best = strassen_block()
    if not verify_fast(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()
    
    # Try to find lower rank by exploiting structure
    # The 2x4 x 4x8 problem can be viewed as 4 independent 2x4 x 4x2 blocks
    # Each 2x4 x 4x2 has schoolbook rank 16. Strassen gives 7*2=14 per pair of columns.
    # But maybe we can do better by sharing across blocks.
    
    # Try rank 48: 8 blocks * 6 each (if we can save 1 per block)
    # Try rank 42: 6 blocks * 7 each (merge 2 blocks into 1 with 7 ops)
    
    # For now, try to verify if any truncation works
    for target in [48, 42, 36]:
        if time.time() > deadline - 15:
            break
        
        if target < rank_best:
            U_t = U_best[:, :target]
            V_t = V_best[:, :target]
            W_t = W_best[:, :target]
            if verify_fast(U_t, V_t, W_t, target):
                U_best, V_best, W_best, rank_best = U_t, V_t, W_t, target
                break
    
    # Final verification
    if not verify_fast(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
