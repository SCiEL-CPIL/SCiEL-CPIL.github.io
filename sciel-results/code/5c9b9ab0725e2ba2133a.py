"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = 2, 3, 5
    
    # We'll build U, V, W as dense matrices
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    max_rank = 40
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))
    col = 0
    
    def add_term(u_vals, v_vals, w_vals):
        """u_vals: list of (i,j,val), v_vals: list of (j,k,val), w_vals: list of (k,i,val)"""
        nonlocal col
        if col >= max_rank:
            return
        for (i, j, val) in u_vals:
            U[i*m+j, col] += val
        for (j, k, val) in v_vals:
            V[j*p+k, col] += val
        for (k, i, val) in w_vals:
            W[k*n+i, col] += val
        col += 1
    
    # ===== PART 1: j in {0,1}, k in {0,1} (2x2x2 Strassen, rank 7) =====
    i0,i1 = 0,1; j0,j1 = 0,1; k0,k1 = 0,1
    
    # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    add_term([(i0,j0,1),(i1,j1,1)], [(j0,k0,1),(j1,k1,1)], [(k0,i0,1),(k1,i1,1)])
    # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
    add_term([(i1,j0,1),(i1,j1,1)], [(j0,k0,1)], [(k0,i1,1)])
    # M3 = A[i0,j0] * (B[j1,k0]-B[j1,k1])
    add_term([(i0,j0,1)], [(j1,k0,1),(j1,k1,-1)], [(k0,i0,1),(k1,i1,1)])
    # M4 = A[i1,j1] * (B[j0,k1]-B[j0,k0])
    add_term([(i1,j1,1)], [(j0,k1,1),(j0,k0,-1)], [(k0,i1,1),(k1,i0,1)])
    # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    add_term([(i0,j0,1),(i0,j1,1)], [(j1,k1,1)], [(k1,i0,1),(k1,i1,1)])
    # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
    add_term([(i1,j0,1),(i0,j0,-1)], [(j0,k0,1),(j0,k1,1)], [(k1,i0,1),(k1,i1,1)])
    # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
    add_term([(i0,j1,1),(i1,j1,-1)], [(j1,k0,1),(j1,k1,1)], [(k0,i0,1),(k1,i1,1)])
    
    # ===== PART 2: j in {0,1}, k in {2,3} (2x2x2 Strassen, rank 7) =====
    k0,k1 = 2,3
    
    # M1
    add_term([(i0,j0,1),(i1,j1,1)], [(j0,k0,1),(j1,k1,1)], [(k0,i0,1),(k1,i1,1)])
    # M2
    add_term([(i1,j0,1),(i1,j1,1)], [(j0,k0,1)], [(k0,i1,1)])
    # M3
    add_term([(i0,j0,1)], [(j1,k0,1),(j1,k1,-1)], [(k0,i0,1),(k1,i1,1)])
    # M4
    add_term([(i1,j1,1)], [(j0,k1,1),(j0,k0,-1)], [(k0,i1,1),(k1,i0,1)])
    # M5
    add_term([(i0,j0,1),(i0,j1,1)], [(j1,k1,1)], [(k1,i0,1),(k1,i1,1)])
    # M6
    add_term([(i1,j0,1),(i0,j0,-1)], [(j0,k0,1),(j0,k1,1)], [(k1,i0,1),(k1,i1,1)])
    # M7
    add_term([(i0,j1,1),(i1,j1,-1)], [(j1,k0,1),(j1,k1,1)], [(k0,i0,1),(k1,i1,1)])
    
    # ===== PART 3: j in {0,1}, k = 4 (2x2x1, rank 4) =====
    for i in range(n):
        for j in range(2):
            add_term([(i,j,1)], [(j,4,1)], [(4,i,1)])
    
    # ===== PART 4: j = 2, all k (2x1x5, rank 2) =====
    # Term for i=0: A[0,2]*B[2,k] for all k
    add_term([(0,2,1)], [(2,k,1) for k in range(p)], [(k,0,1) for k in range(p)])
    # Term for i=1: A[1,2]*B[2,k] for all k
    add_term([(1,2,1)], [(2,k,1) for k in range(p)], [(k,1,1) for k in range(p)])
    
    rank = col
    U_out = U[:, :rank]
    V_out = V[:, :rank]
    W_out = W[:, :rank]
    
    # Convert to lists of lists with float entries
    def to_list(arr):
        result = []
        for row in arr:
            r = []
            for val in row:
                val_f = float(val)
                if abs(val_f - round(val_f*2)) < 1e-10:
                    val_half = round(val_f*2)/2
                    if val_half == int(val_half):
                        r.append(int(val_half))
                    else:
                        r.append(val_half)
                else:
                    r.append(val_f)
            result.append(r)
        return result
    
    # Verify numerically
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_verify = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                total = 0
                for j in range(m):
                    total += U_out[i*m+j, r] * V_out[j*p+k, r]
                C_verify[i,k] += W_out[k*n+i, r] * total
    
    error = np.max(np.abs(C_verify - C_expected))
    
    if error > 1e-8:
        # Fallback to schoolbook
        U_school = np.zeros((n*m, n*m*p))
        V_school = np.zeros((m*p, n*m*p))
        W_school = np.zeros((p*n, n*m*p))
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col_s] = 1
                    V_school[j*p+k, col_s] = 1
                    W_school[k*n+i, col_s] = 1
                    col_s += 1
        return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
    
    return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
# EVOLVE-BLOCK-END
