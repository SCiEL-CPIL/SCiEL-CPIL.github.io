import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    
    # We will determine the total rank first.
    # Block 1: 2x4x2 -> 14
    # Block 2: 2x4x3 -> 22
    total_rank = 36
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i
        
    current_rank = 0
    
    # Helper to add Strassen block
    # A is 2x2, B is 2x2, C is 2x2
    # A rows: a_r0, a_r1. A cols: a_c0, a_c1
    # B rows: b_r0, b_r1. B cols: b_c0, b_c1
    # C rows: c_r0, c_r1. C cols: c_c0, c_c1
    def add_strassen(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
        nonlocal current_rank
        
        # M1 = (A00+A11)(B00+B11)
        U[u_idx(a_r0, a_c0)][current_rank] += 1
        U[u_idx(a_r1, a_c1)][current_rank] += 1
        V[v_idx(b_r0, b_c0)][current_rank] += 1
        V[v_idx(b_r1, b_c1)][current_rank] += 1
        W[w_idx(c_c0, c_r0)][current_rank] += 1 # C00
        W[w_idx(c_c1, c_r1)][current_rank] += 1 # C11
        current_rank += 1
        
        # M2 = (A10+A11)B00
        U[u_idx(a_r1, a_c0)][current_rank] += 1
        U[u_idx(a_r1, a_c1)][current_rank] += 1
        V[v_idx(b_r0, b_c0)][current_rank] += 1
        W[w_idx(c_c0, c_r1)][current_rank] += 1 # C10
        W[w_idx(c_c1, c_r1)][current_rank] += -1 # C11
        current_rank += 1
        
        # M3 = A00(B10-B11)
        U[u_idx(a_r0, a_c0)][current_rank] += 1
        V[v_idx(b_r1, b_c0)][current_rank] += 1
        V[v_idx(b_r1, b_c1)][current_rank] += -1
        W[w_idx(c_c1, c_r0)][current_rank] += 1 # C01
        W[w_idx(c_c1, c_r1)][current_rank] += 1 # C11
        current_rank += 1
        
        # M4 = A11(B01-B00)
        U[u_idx(a_r1, a_c1)][current_rank] += 1
        V[v_idx(b_r0, b_c1)][current_rank] += 1
        V[v_idx(b_r0, b_c0)][current_rank] += -1
        W[w_idx(c_c0, c_r0)][current_rank] += 1 # C00
        W[w_idx(c_c0, c_r1)][current_rank] += 1 # C10
        current_rank += 1
        
        # M5 = (A00+A01)B11
        U[u_idx(a_r0, a_c0)][current_rank] += 1
        U[u_idx(a_r0, a_c1)][current_rank] += 1
        V[v_idx(b_r1, b_c1)][current_rank] += 1
        W[w_idx(c_c0, c_r0)][current_rank] += -1 # C00
        W[w_idx(c_c1, c_r0)][current_rank] += 1 # C01
        current_rank += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u_idx(a_r1, a_c0)][current_rank] += 1
        U[u_idx(a_r0, a_c0)][current_rank] += -1
        V[v_idx(b_r0, b_c0)][current_rank] += 1
        V[v_idx(b_r0, b_c1)][current_rank] += 1
        W[w_idx(c_c1, c_r1)][current_rank] += 1 # C11
        current_rank += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u_idx(a_r0, a_c1)][current_rank] += 1
        U[u_idx(a_r1, a_c1)][current_rank] += -1
        V[v_idx(b_r1, b_c0)][current_rank] += 1
        V[v_idx(b_r1, b_c1)][current_rank] += 1
        W[w_idx(c_c0, c_r0)][current_rank] += 1 # C00
        current_rank += 1
        
    # Helper to add 2x2x3 Rank 11 block
    # A is 2x2, B is 2x3, C is 2x3
    # A rows: a_r0, a_r1. A cols: a_c0, a_c1
    # B rows: b_r0, b_r1. B cols: b_c0, b_c1, b_c2
    # C rows: c_r0, c_r1. C cols: c_c0, c_c1, c_c2
    def add_2x2x3_rank11(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, b_c2, c_r0, c_r1, c_c0, c_c1, c_c2):
        nonlocal current_rank
        
        # First 7 are Strassen on B[:, 0:2]
        add_strassen(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1)
        
        # Last 4 are Schoolbook for B[:, 2]
        # K1 = A00 * B02
        U[u_idx(a_r0, a_c0)][current_rank] += 1
        V[v_idx(b_r0, b_c2)][current_rank] += 1
        W[w_idx(c_c2, c_r0)][current_rank] += 1
        current_rank += 1
        
        # K2 = A01 * B12
        U[u_idx(a_r0, a_c1)][current_rank] += 1
        V[v_idx(b_r1, b_c2)][current_rank] += 1
        W[w_idx(c_c2, c_r0)][current_rank] += 1
        current_rank += 1
        
        # K3 = A10 * B02
        U[u_idx(a_r1, a_c0)][current_rank] += 1
        V[v_idx(b_r0, b_c2)][current_rank] += 1
        W[w_idx(c_c2, c_r1)][current_rank] += 1
        current_rank += 1
        
        # K4 = A11 * B12
        U[u_idx(a_r1, a_c1)][current_rank] += 1
        V[v_idx(b_r1, b_c2)][current_rank] += 1
        W[w_idx(c_c2, c_r1)][current_rank] += 1
        current_rank += 1

    # Block 1: C[:, 0:2] = A * B[:, 0:2]
    # A is rows 0,1 cols 0..3
    # B is rows 0..3 cols 0,1
    # Split A into AL (cols 0,1) and AR (cols 2,3)
    # Split B into BT (rows 0,1) and BB (rows 2,3)
    
    # AL * BT
    add_strassen(
        a_r0=0, a_r1=1, a_c0=0, a_c1=1, # AL
        b_r0=0, b_r1=1, b_c0=0, b_c1=1, # BT
        c_r0=0, c_r1=1, c_c0=0, c_c1=1  # C[:, 0:2]
    )
    
    # AR * BB
    add_strassen(
        a_r0=0, a_r1=1, a_c0=2, a_c1=3, # AR
        b_r0=2, b_r1=3, b_c0=0, b_c1=1, # BB
        c_r0=0, c_r1=1, c_c0=0, c_c1=1  # C[:, 0:2]
    )
    
    # Block 2: C[:, 2:5] = A * B[:, 2:5]
    # Split A into AL (cols 0,1) and AR (cols 2,3)
    # Split B into BT2 (rows 0,1) and BB2 (rows 2,3)
    
    # AL * BT2
    add_2x2x3_rank11(
        a_r0=0, a_r1=1, a_c0=0, a_c1=1, # AL
        b_r0=0, b_r1=1, b_c0=2, b_c1=3, b_c2=4, # BT2
        c_r0=0, c_r1=1, c_c0=2, c_c1=3, c_c2=4  # C[:, 2:5]
    )
    
    # AR * BB2
    add_2x2x3_rank11(
        a_r0=0, a_r1=1, a_c0=2, a_c1=3, # AR
        b_r0=2, b_r1=3, b_c0=2, b_c1=3, b_c2=4, # BB2
        c_r0=0, c_r1=1, c_c0=2, c_c1=3, c_c2=4  # C[:, 2:5]
    )
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple validation
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    n, m, p = DIMENSIONS
    # Check shape
    assert U.shape == (n*m, U.shape[1])
    assert V.shape == (m*p, V.shape[1])
    assert W.shape == (p*n, W.shape[1])
    assert U.shape[1] == V.shape[1] == W.shape[1]
    
    rank = U.shape[1]
    print(f"Rank: {rank}")
    
    # Test with random matrices
    import random
    for _ in range(10):
        A = np.random.randint(0, 5, (n, m))
        B = np.random.randint(0, 5, (m, p))
        C_expected = A @ B
        
        # Reconstruct C
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is not how it works.
        # The tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Here i is row of A, j is col of A / row of B, k is col of B.
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        C_recon = np.zeros((n, p), dtype=int)
        for r in range(rank):
            # Extract vectors
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = u_vec[i*m + j]
                        v_val = v_vec[j*p + k]
                        w_val = w_vec[k*n + i]
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            C_recon[i, k] += u_val * v_val * w_val
        
        if not np.array_equal(C_expected, C_recon):
            print("Mismatch!")
            print(A)
            print(B)
            print(C_expected)
            print(C_recon)
            break
    else:
        print("All tests passed.")
