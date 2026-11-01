import time
import numpy as np

def _verify(U, V, W, n=2, m=2, p=4):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.ones((n, m, p))
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.allclose(T, T_recon, atol=1e-9)

def _strassen_rank14():
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2x2 factors
    # m1 = (A00+A11)(B00+B11) -> C00 += m1, C11 += m1
    # m2 = (A10+A11)B00       -> C10 += m2, C11 -= m2
    # m3 = A00(B10-B11)       -> C01 += m3, C11 += m3
    # m4 = A11(B01-B00)       -> C00 += m4, C10 += m4
    # m5 = (A00+A01)B11       -> C01 += m5, C00 -= m5
    # m6 = (A10-A00)(B00+B01) -> C11 += m6
    # m7 = (A01-A11)(B10+B11) -> C00 += m7
    
    # Local indices for 2x2 block:
    # A: 0=A00, 1=A01, 2=A10, 3=A11
    # B: 0=B00, 1=B01, 2=B10, 3=B11
    # C: 0=C00, 1=C01, 2=C10, 3=C11
    
    # Global indices:
    # U: i*m+j. A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # V: j*p+k. B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # W: k*n+i. C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    def add_block(rank_offset, b_map, c_map):
        # b_map: local B idx -> global V idx
        # c_map: local C idx -> global W idx
        r0 = rank_offset
        
        # m1
        r = r0
        U[0, r] = 1; U[3, r] = 1
        V[b_map[0], r] = 1; V[b_map[3], r] = 1
        W[c_map[0], r] = 1; W[c_map[3], r] = 1
        
        # m2
        r = r0 + 1
        U[2, r] = 1; U[3, r] = 1
        V[b_map[0], r] = 1
        W[c_map[2], r] = 1; W[c_map[3], r] = -1
        
        # m3
        r = r0 + 2
        U[0, r] = 1
        V[b_map[2], r] = 1; V[b_map[3], r] = -1
        W[c_map[1], r] = 1; W[c_map[3], r] = 1
        
        # m4
        r = r0 + 3
        U[3, r] = 1
        V[b_map[1], r] = 1; V[b_map[0], r] = -1
        W[c_map[0], r] = 1; W[c_map[2], r] = 1
        
        # m5
        r = r0 + 4
        U[0, r] = 1; U[1, r] = 1
        V[b_map[3], r] = 1
        W[c_map[1], r] = 1; W[c_map[0], r] = -1
        
        # m6
        r = r0 + 5
        U[2, r] = 1; U[0, r] = -1
        V[b_map[0], r] = 1; V[b_map[1], r] = 1
        W[c_map[3], r] = 1
        
        # m7
        r = r0 + 6
        U[1, r] = 1; U[3, r] = -1
        V[b_map[2], r] = 1; V[b_map[3], r] = 1
        W[c_map[0], r] = 1

    # Block 1: Left (k=0,1)
    # B local 0 (B00) -> global 0 (B[0,0])
    # B local 1 (B01) -> global 1 (B[0,1])
    # B local 2 (B10) -> global 2 (B[1,0])
    # B local 3 (B11) -> global 3 (B[1,1])
    # Wait, V index is j*p+k.
    # B[0,0] -> j=0,k=0 -> 0*4+0=0
    # B[0,1] -> j=0,k=1 -> 0*4+1=1
    # B[1,0] -> j=1,k=0 -> 1*4+0=4
    # B[1,1] -> j=1,k=1 -> 1*4+1=5
    # My previous mapping in the "current program" had:
    # b_left = {0: 0, 1: 1, 2: 4, 3: 5}
    # This is correct.
    
    # C local 0 (C00) -> i=0,k=0 -> W idx k*n+i = 0*2+0=0
    # C local 1 (C01) -> i=0,k=1 -> W idx 1*2+0=2
    # C local 2 (C10) -> i=1,k=0 -> W idx 0*2+1=1
    # C local 3 (C11) -> i=1,k=1 -> W idx 1*2+1=3
    
    # Wait, let's re-read the W convention: W[k*n+i].
    # Output C is 2x4. C[i,k].
    # i in 0..1, k in 0..3.
    # For block 1 (left half, k=0,1):
    # C[0,0] -> i=0, k=0 -> W[0*2+0] = W[0]
    # C[0,1] -> i=0, k=1 -> W[1*2+0] = W[2]
    # C[1,0] -> i=1, k=0 -> W[0*2+1] = W[1]
    # C[1,1] -> i=1, k=1 -> W[1*2+1] = W[3]
    
    # So for block 1:
    # c_map local 0 (C00) -> 0
    # c_map local 1 (C01) -> 2
    # c_map local 2 (C10) -> 1
    # c_map local 3 (C11) -> 3
    
    b_left = {0: 0, 1: 1, 2: 4, 3: 5}
    c_left = {0: 0, 1: 2, 2: 1, 3: 3}
    
    add_block(0, b_left, c_left)
    
    # Block 2: Right (k=2,3)
    # B[0,2] -> j=0,k=2 -> 2
    # B[0,3] -> j=0,k=3 -> 3
    # B[1,2] -> j=1,k=2 -> 6
    # B[1,3] -> j=1,k=3 -> 7
    b_right = {0: 2, 1: 3, 2: 6, 3: 7}
    
    # C[0,2] -> i=0, k=2 -> W[2*2+0] = W[4]
    # C[0,3] -> i=0, k=3 -> W[3*2+0] = W[6]
    # C[1,2] -> i=1, k=2 -> W[2*2+1] = W[5]
    # C[1,3] -> i=1, k=3 -> W[3*2+1] = W[7]
    
    c_right = {0: 4, 1: 6, 2: 5, 3: 7}
    
    add_block(7, b_right, c_right)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    
    # Try rank 14
    res = _strassen_rank14()
    
    if _verify(res["U"], res["V"], res["W"]):
        return res
    
    # Fallback to rank 16 (schoolbook)
    n, m, p = 2, 2, 4
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1
                V[j*p+k, idx] = 1
                W[k*n+i, idx] = 1
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Quick self-test if run as main
if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    print(f"Rank: {rank}")
    print(f"Valid: {_verify(U, V, W)}")
