"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_BUDGET = 120.0

def _time_left():
    return max(0.0, TIME_BUDGET - (time.time() - _start_time))

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    r = n*m*p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _build_strassen_3x3x8():
    """
    Build a Strassen-based decomposition for 3x3 * 3x8.
    
    Strategy: Split the 3x3 matrix A into a 2x2 block (top-left) and remaining parts.
    Use Strassen's 7 products on the 2x2 sub-block, plus schoolbook for cross terms.
    
    Block decomposition:
    A = [[A00, A01, A02],   B = [[B00, B01, ..., B07],
         [A10, A11, A12],        [B10, B11, ..., B17],
         [A20, A21, A22]]        [B20, B21, ..., B27]]
    
    Split into:
    A_tl = [[A00, A01],     B_tl = [[B00..B07],
             [A10, A11]]           [B10..B17]]   (2x2 * 2x8)
    
    A_tr = [A02, A12]       (2x1)
    A_bl = [A20, A21]       (1x2)
    A_br = [A22]            (1x1)
    
    B_bl = [B20..B27]       (1x8)
    
    C = A @ B
    C[0:2, :] = A_tl @ B_tl + A_tr @ B_bl
    C[2, :]   = A_bl @ B_tl + A_br @ B_bl
    
    For A_tl @ B_tl (2x2 * 2x8), use Strassen with 7 products:
    M1 = (A00+A11) * (B0 + B1)     where B0 = B_tl[0,:], B1 = B_tl[1,:]
    M2 = (A10+A11) * B0
    M3 = A00 * (B1 - B0)
    M4 = A11 * (B0 - B1)
    M5 = (A00+A01) * B1
    M6 = (A10-A00) * (B0+B1)
    M7 = (A01-A11) * (B1+B0)
    
    C_tl[0,:] = M1 + M4 - M5 + M7
    C_tl[1,:] = M3 + M1 - M2 - M6
    
    Then add cross terms:
    C[0,:] += A02 * B2
    C[1,:] += A12 * B2
    C[2,:] += A20 * B0 + A21 * B1 + A22 * B2
    
    Total rank: 7 (Strassen) + 2 (A_tr @ B_bl) + 3 (A_bl @ B_tl + A_br @ B_bl) = 12
    Wait, let me recount:
    - 7 Strassen products
    - A02 * B2: 1 product
    - A12 * B2: 1 product  
    - A20 * B0: 1 product
    - A21 * B1: 1 product
    - A22 * B2: 1 product
    Total: 7 + 5 = 12
    
    But wait, we can share some. Let me think more carefully.
    
    Actually, let's just build it explicitly and count.
    """
    n, m, p = 3, 3, 8
    
    # We'll build the factor matrices column by column
    U_cols = []  # Each is a dict {i*m+j: coeff}
    V_cols = []  # Each is a dict {j*p+k: coeff}
    W_cols = []  # Each is a dict {k*n+i: coeff}
    
    def add_term(U_dict, V_dict, W_dict):
        U_cols.append(U_dict)
        V_cols.append(V_dict)
        W_cols.append(W_dict)
    
    # Strassen for top-left 2x2 block (rows 0,1 of A; rows 0,1 of B)
    # M1 = (A00+A11) * (B0+B1)
    add_term({0: 1, 3: 1}, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1,
                              8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1},
             {0: 1, 1: 1, 2: 1})  # W[k*n+i] for k in 0..7, i in 0,1
    # Actually W[k*n+i]: for output C[i,k], i in {0,1}, k in 0..7
    # W index = k*n + i = k*3 + i
    # For M1 contributing to C[0,:] and C[1,:]:
    # W should have entries for i=0 and i=1, all k
    # Let me redo this more carefully
    
    # Reset and be very explicit
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Index helpers
    def u_idx(i, j): return i * m + j  # A[i,j]
    def v_idx(j, k): return j * p + k  # B[j,k]
    def w_idx(k, i): return k * n + i  # C[i,k] in trace convention W[k*n+i]
    
    # Strassen 2x2 * 2x8
    # A_sub = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B_sub = [[B[0,0..7]], [B[1,0..7]]]
    
    # M1 = (A[0,0] + A[1,1]) * (B[0,:] + B[1,:])
    # Contributes to C[0,:] and C[1,:]
    add_term({u_idx(0,0): 1, u_idx(1,1): 1},
             {v_idx(0,k): 1 for k in range(p)} | {v_idx(1,k): 1 for k in range(p)},
             {w_idx(k,0): 1 for k in range(p)} | {w_idx(k,1): 1 for k in range(p)})
    
    # M2 = (A[1,0] + A[1,1]) * B[0,:]
    # Contributes to C[1,:]
    add_term({u_idx(1,0): 1, u_idx(1,1): 1},
             {v_idx(0,k): 1 for k in range(p)},
             {w_idx(k,1): 1 for k in range(p)})
    
    # M3 = A[0,0] * (B[1,:] - B[0,:])
    # Contributes to C[1,:]
    add_term({u_idx(0,0): 1},
             {v_idx(1,k): 1 for k in range(p)} | {v_idx(0,k): -1 for k in range(p)},
             {w_idx(k,1): 1 for k in range(p)})
    
    # M4 = A[1,1] * (B[0,:] - B[1,:])
    # Contributes to C[0,:]
    add_term({u_idx(1,1): 1},
             {v_idx(0,k): 1 for k in range(p)} | {v_idx(1,k): -1 for k in range(p)},
             {w_idx(k,0): 1 for k in range(p)})
    
    # M5 = (A[0,0] + A[0,1]) * B[1,:]
    # Contributes to C[0,:]
    add_term({u_idx(0,0): 1, u_idx(0,1): 1},
             {v_idx(1,k): 1 for k in range(p)},
             {w_idx(k,0): 1 for k in range(p)})
    
    # M6 = (A[1,0] - A[0,0]) * (B[0,:] + B[1,:])
    # Contributes to C[1,:]
    add_term({u_idx(1,0): 1, u_idx(0,0): -1},
             {v_idx(0,k): 1 for k in range(p)} | {v_idx(1,k): 1 for k in range(p)},
             {w_idx(k,1): 1 for k in range(p)})
    
    # M7 = (A[0,1] - A[1,1]) * (B[1,:] + B[0,:])
    # Contributes to C[0,:]
    add_term({u_idx(0,1): 1, u_idx(1,1): -1},
             {v_idx(1,k): 1 for k in range(p)} | {v_idx(0,k): 1 for k in range(p)},
             {w_idx(k,0): 1 for k in range(p)})
    
    # Now add cross terms
    # C[0,:] += A[0,2] * B[2,:]
    add_term({u_idx(0,2): 1},
             {v_idx(2,k): 1 for k in range(p)},
             {w_idx(k,0): 1 for k in range(p)})
    
    # C[1,:] += A[1,2] * B[2,:]
    add_term({u_idx(1,2): 1},
             {v_idx(2,k): 1 for k in range(p)},
             {w_idx(k,1): 1 for k in range(p)})
    
    # C[2,:] += A[2,0] * B[0,:]
    add_term({u_idx(2,0): 1},
             {v_idx(0,k): 1 for k in range(p)},
             {w_idx(k,2): 1 for k in range(p)})
    
    # C[2,:] += A[2,1] * B[1,:]
    add_term({u_idx(2,1): 1},
             {v_idx(1,k): 1 for k in range(p)},
             {w_idx(k,2): 1 for k in range(p)})
    
    # C[2,:] += A[2,2] * B[2,:]
    add_term({u_idx(2,2): 1},
             {v_idx(2,k): 1 for k in range(p)},
             {w_idx(k,2): 1 for k in range(p)})
    
    # Build the final matrices
    r = len(U_cols)
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c in range(r):
        for idx, val in U_cols[c].items():
            U[idx][c] = val
        for idx, val in V_cols[c].items():
            V[idx][c] = val
        for idx, val in W_cols[c].items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try Strassen-based decomposition
    if _time_left() > 5.0:
        cand = _build_strassen_3x3x8()
        if cand is not None and _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            r = len(cand["U"][0])
            if r < best_rank:
                best = cand
                best_rank = r
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
