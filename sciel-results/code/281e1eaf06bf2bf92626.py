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
    # Blocks: (s, t) where s is A-col-block (0: L, 1: R), t is B-col-block (0, 1, 2)
    # But B row-block depends on s.
    # If s=0, B rows are 0, 1.
    # If s=1, B rows are 2, 3.
    
    for s in range(2):
        a_j0 = 2 * s
        b_j0 = 2 * s  # B row start
        
        for t in range(3):
            b_k0 = 2 * t
            
            # Define the 7 Strassen multipliers
            # Each multiplier is a pair (U_coeff, V_coeff)
            # U_coeff is a vector of length n*m (4)
            # V_coeff is a vector of length m*p (24)
            
            # M1: (A00 + A11) * (B00 + B11)
            # A00 is at (0, a_j0), A11 is at (1, a_j0+1)
            # B00 is at (b_j0, b_k0), B11 is at (b_j0+1, b_k0+1)
            
            def add_multiplier(u_idx, u_val, v_idx, v_val, w_updates):
                nonlocal r
                U[u_idx, r] += u_val
                V[v_idx, r] += v_val
                for (i, k), val in w_updates:
                    W[k * n + i, r] += val
                r += 1

            # Helper to get flat indices
            def u_idx(i, j): return i * m + j
            def v_idx(j, k): return j * p + k
            def w_idx(k, i): return k * n + i

            # M1
            u1 = {u_idx(0, a_j0): 1.0, u_idx(1, a_j0+1): 1.0}
            v1 = {v_idx(b_j0, b_k0): 1.0, v_idx(b_j0+1, b_k0+1): 1.0}
            w1 = { (0, b_k0): 1.0, (0, b_k0+1): 1.0, (1, b_k0): 1.0, (1, b_k0+1): -1.0 }
            # Wait, W updates are for C[i, k].
            # C00 += M1, C01 += M1, C10 += M1, C11 -= M1?
            # Formulas:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So for M1:
            # C00 (i=0, k=b_k0) += 1
            # C01 (i=0, k=b_k0+1) += 1  <-- Wait, C01 is (0, b_k0+1).
            # C10 (i=1, k=b_k0) += 1
            # C11 (i=1, k=b_k0+1) += 1
            
            # Let's re-verify Strassen output mapping.
            # Standard Strassen:
            # P1 = (A11+A22)(B11+B22)
            # P2 = (A21+A22)B11
            # P3 = A11(B21-B11)
            # P4 = A22(B12-B22)
            # P5 = (A11+A12)B22
            # P6 = (A21-A11)(B11+B12)
            # P7 = (A12-A22)(B21+B22)
            # C11 = P1 + P4 - P5 + P7
            # C12 = P3 + P5
            # C21 = P2 + P4
            # C22 = P1 - P2 + P3 + P6
            
            # My indices:
            # A11 -> A[0, a_j0]
            # A12 -> A[0, a_j0+1]
            # A21 -> A[1, a_j0]
            # A22 -> A[1, a_j0+1]
            
            # B11 -> B[b_j0, b_k0]
            # B12 -> B[b_j0, b_k0+1]
            # B21 -> B[b_j0+1, b_k0]
            # B22 -> B[b_j0+1, b_k0+1]
            
            # C11 -> C[0, b_k0]
            # C12 -> C[0, b_k0+1]
            # C21 -> C[1, b_k0]
            # C22 -> C[1, b_k0+1]

            # M1 (P1):
            # U: A11(1), A22(1)
            # V: B11(1), B22(1)
            # W: C11(1), C12(1), C21(1), C22(1)
            
            # M2 (P2):
            # U: A21(1), A22(1)
            # V: B11(1)
            # W: C21(1), C22(-1)
            
            # M3 (P3):
            # U: A11(1)
            # V: B21(1), B11(-1)
            # W: C12(1), C22(1)
            
            # M4 (P4):
            # U: A22(1)
            # V: B12(1), B22(-1)
            # W: C11(1), C21(1)
            
            # M5 (P5):
            # U: A11(1), A12(1)
            # V: B22(1)
            # W: C11(-1), C12(1)
            
            # M6 (P6):
            # U: A21(1), A11(-1)
            # V: B11(1), B12(1)
            # W: C22(1)
            
            # M7 (P7):
            # U: A12(1), A22(-1)
            # V: B21(1), B22(1)
            # W: C11(1)

            # Implementing these 7 steps
            
            # 1
            U[u_idx(0, a_j0), r] = 1.0
            U[u_idx(1, a_j0+1), r] = 1.0
            V[v_idx(b_j0, b_k0), r] = 1.0
            V[v_idx(b_j0+1, b_k0+1), r] = 1.0
            W[w_idx(b_k0, 0), r] = 1.0
            W[w_idx(b_k0+1, 0), r] = 1.0
            W[w_idx(b_k0, 1), r] = 1.0
            W[w_idx(b_k0+1, 1), r] = 1.0
            r += 1
            
            # 2
            U[u_idx(1, a_j0), r] = 1.0
            U[u_idx(1, a_j0+1), r] = 1.0
            V[v_idx(b_j0, b_k0), r] = 1.0
            W[w_idx(b_k0, 1), r] = 1.0
            W[w_idx(b_k0+1, 1), r] = -1.0
            r += 1
            
            # 3
            U[u_idx(0, a_j0), r] = 1.0
            V[v_idx(b_j0+1, b_k0), r] = 1.0
            V[v_idx(b_j0, b_k0), r] = -1.0
            W[w_idx(b_k0+1, 0), r] = 1.0
            W[w_idx(b_k0+1, 1), r] = 1.0
            r += 1
            
            # 4
            U[u_idx(1, a_j0+1), r] = 1.0
            V[v_idx(b_j0, b_k0+1), r] = 1.0
            V[v_idx(b_j0+1, b_k0+1), r] = -1.0
            W[w_idx(b_k0, 0), r] = 1.0
            W[w_idx(b_k0, 1), r] = 1.0
            r += 1
            
            # 5
            U[u_idx(0, a_j0), r] = 1.0
            U[u_idx(0, a_j0+1), r] = 1.0
            V[v_idx(b_j0+1, b_k0+1), r] = 1.0
            W[w_idx(b_k0, 0), r] = -1.0
            W[w_idx(b_k0+1, 0), r] = 1.0
            r += 1
            
            # 6
            U[u_idx(1, a_j0), r] = 1.0
            U[u_idx(0, a_j0), r] = -1.0
            V[v_idx(b_j0, b_k0), r] = 1.0
            V[v_idx(b_j0, b_k0+1), r] = 1.0
            W[w_idx(b_k0+1, 1), r] = 1.0
            r += 1
            
            # 7
            U[u_idx(0, a_j0+1), r] = 1.0
            U[u_idx(1, a_j0+1), r] = -1.0
            V[v_idx(b_j0+1, b_k0), r] = 1.0
            V[v_idx(b_j0+1, b_k0+1), r] = 1.0
            W[w_idx(b_k0, 0), r] = 1.0
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
