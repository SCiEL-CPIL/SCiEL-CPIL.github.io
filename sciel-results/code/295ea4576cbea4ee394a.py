import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 135.0

    n, m, p = DIMENSIONS
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    
    # Helper to get flat indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Iterate over A blocks (L, R) and B blocks (0, 1, 2)
    for s in range(2):
        for t in range(3):
            # Determine A indices for this block
            if s == 0:
                # A_L: cols 0, 1
                a11 = u_idx(0, 0)
                a12 = u_idx(0, 1)
                a21 = u_idx(1, 0)
                a22 = u_idx(1, 1)
                # B part: Top (rows 0, 1)
                b11 = v_idx(0, 2*t)
                b12 = v_idx(0, 2*t+1)
                b21 = v_idx(1, 2*t)
                b22 = v_idx(1, 2*t+1)
            else:
                # A_R: cols 2, 3
                a11 = u_idx(0, 2)
                a12 = u_idx(0, 3)
                a21 = u_idx(1, 2)
                a22 = u_idx(1, 3)
                # B part: Bot (rows 2, 3)
                b11 = v_idx(2, 2*t)
                b12 = v_idx(2, 2*t+1)
                b21 = v_idx(3, 2*t)
                b22 = v_idx(3, 2*t+1)
            
            # C indices for this block t
            c11 = w_idx(2*t, 0)
            c12 = w_idx(2*t+1, 0)
            c21 = w_idx(2*t, 1)
            c22 = w_idx(2*t+1, 1)
            
            # Strassen Multipliers
            # P1 = (A11+A22)(B11+B22)
            U[a11, r] = 1.0
            U[a22, r] = 1.0
            V[b11, r] = 1.0
            V[b22, r] = 1.0
            W[c11, r] = 1.0
            W[c12, r] = 1.0
            W[c21, r] = 1.0
            W[c22, r] = 1.0
            r += 1
            
            # P2 = (A21+A22)B11
            U[a21, r] = 1.0
            U[a22, r] = 1.0
            V[b11, r] = 1.0
            W[c21, r] = 1.0
            W[c22, r] = -1.0
            r += 1
            
            # P3 = A11(B21-B11)
            U[a11, r] = 1.0
            V[b21, r] = 1.0
            V[b11, r] = -1.0
            W[c12, r] = 1.0
            W[c22, r] = 1.0
            r += 1
            
            # P4 = A22(B12-B22)
            U[a22, r] = 1.0
            V[b12, r] = 1.0
            V[b22, r] = -1.0
            W[c11, r] = 1.0
            W[c21, r] = 1.0
            r += 1
            
            # P5 = (A11+A12)B22
            U[a11, r] = 1.0
            U[a12, r] = 1.0
            V[b22, r] = 1.0
            W[c11, r] = -1.0
            W[c12, r] = 1.0
            r += 1
            
            # P6 = (A21-A11)(B11+B12)
            U[a21, r] = 1.0
            U[a11, r] = -1.0
            V[b11, r] = 1.0
            V[b12, r] = 1.0
            W[c22, r] = 1.0
            r += 1
            
            # P7 = (A12-A22)(B21+B22)
            U[a12, r] = 1.0
            U[a22, r] = -1.0
            V[b21, r] = 1.0
            V[b22, r] = 1.0
            W[c11, r] = 1.0
            r += 1

    # Verification
    if time.time() < DEADLINE - 10:
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        u_vals = U @ A_flat
        v_vals = V @ B_flat
        prod_vals = u_vals * v_vals
        
        W_reshaped = W.reshape(p, n, rank).transpose(1, 0, 2)
        C_decomp = np.sum(W_reshaped * prod_vals[None, None, :], axis=2)
        
        if not np.allclose(C_true, C_decomp):
            raise ValueError("Decomposition verification failed")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
