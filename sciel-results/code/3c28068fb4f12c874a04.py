import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _build_2x2x2_strassen():
    """
    Build the rank-7 Strassen decomposition for 2x2x2.
    Index conventions:
    U: i*2 + j
    V: j*2 + k
    W: k*2 + i
    """
    # Terms are (u, v, w) vectors of length 4
    # u indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # v indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # w indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # u: [1, 0, 0, 1]
    # v: [1, 0, 0, 1]
    # w: [1, 0, 0, 1]
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # u: [0, 0, 1, 1]
    # v: [1, 0, 0, 0]
    # w: [0, 0, 1, -1]
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # u: [1, 0, 0, 0]
    # v: [0, 1, 0, -1]
    # w: [0, 1, 0, 1]
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # u: [0, 0, 0, 1]
    # v: [-1, 0, 1, 0]
    # w: [1, 0, 1, 0]
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # u: [1, 1, 0, 0]
    # v: [0, 0, 0, 1]
    # w: [-1, 1, 0, 0]
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # u: [-1, 0, 1, 0]
    # v: [1, 1, 0, 0]
    # w: [0, 0, 0, 1]
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    # u: [0, 1, 0, -1]
    # v: [0, 0, 1, 1]
    # w: [1, 0, 0, 0]
    
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build a rank-49 decomposition for 4x4x4 using Strassen.
    This is the tensor product of the 2x2x2 Strassen decomposition.
    """
    U2, V2, W2 = _build_2x2x2_strassen()
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for r_out in range(7):
        for r_in in range(7):
            col = r_out * 7 + r_in
            
            # U: (i, j) where i, j in {0,1,2,3}
            # i = 2*i_out + i_in, j = 2*j_out + j_in
            # U[idx, col] = U2[i_out*2+j_out, r_out] * U2[i_in*2+j_in, r_in]
            for i_out in range(2):
                for j_out in range(2):
                    u_out_val = U2[i_out * 2 + j_out, r_out]
                    if u_out_val == 0:
                        continue
                    for i_in in range(2):
                        for j_in in range(2):
                            u_in_val = U2[i_in * 2 + j_in, r_in]
                            if u_in_val == 0:
                                continue
                            i = 2 * i_out + i_in
                            j = 2 * j_out + j_in
                            idx = i * 4 + j
                            U[idx, col] += u_out_val * u_in_val
            
            # V: (j, k) where j, k in {0,1,2,3}
            # j = 2*j_out + j_in, k = 2*k_out + k_in
            # V[idx, col] = V2[j_out*2+k_out, r_out] * V2[j_in*2+k_in, r_in]
            for j_out in range(2):
                for k_out in range(2):
                    v_out_val = V2[j_out * 2 + k_out, r_out]
                    if v_out_val == 0:
                        continue
                    for j_in in range(2):
                        for k_in in range(2):
                            v_in_val = V2[j_in * 2 + k_in, r_in]
                            if v_in_val == 0:
                                continue
                            j = 2 * j_out + j_in
                            k = 2 * k_out + k_in
                            idx = j * 4 + k
                            V[idx, col] += v_out_val * v_in_val
            
            # W: (k, i) where k, i in {0,1,2,3}
            # k = 2*k_out + k_in, i = 2*i_out + i_in
            # W[idx, col] = W2[k_out*2+i_out, r_out] * W2[k_in*2+i_in, r_in]
            for k_out in range(2):
                for i_out in range(2):
                    w_out_val = W2[k_out * 2 + i_out, r_out]
                    if w_out_val == 0:
                        continue
                    for k_in in range(2):
                        for i_in in range(2):
                            w_in_val = W2[k_in * 2 + i_in, r_in]
                            if w_in_val == 0:
                                continue
                            k = 2 * k_out + k_in
                            i = 2 * i_out + i_in
                            idx = k * 4 + i
                            W[idx, col] += w_out_val * w_in_val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Vectorized check for speed
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We can check a subset or all. With rank 98, 4*4*8=128 checks.
    # Each check is a dot product of length 98.
    # Total ops ~ 128 * 98 * 3 ~ 37k. Very fast.
    
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                total = np.dot(u_row, v_row * w_row)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _build_4x4x8_rank98():
    """
    Rank-98 decomposition: Strassen for both halves.
    """
    U44, V44, W44 = _build_4x4x4_strassen()
    
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # C1 = A*B1: columns 0..48
    # U is same for both blocks
    U[:, 0:49] = U44
    # V: B1 is top-left 4x4 of B (4x8).
    # B1[j, k] corresponds to V44[j, k]
    # V index is j*8 + k
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V44[j * 4 + k, :]
    # W: C1[i, k] corresponds to W44[i, k]
    # W index is k*n + i = k*4 + i
    W[0:16, 0:49] = W44
    
    # C2 = A*B2: columns 49..97
    # U is same
    U[:, 49:98] = U44
    # V: B2 is bottom-right 4x4 of B.
    # B2[j, k] corresponds to V44[j, k]
    # V index is j*8 + (k+4)
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 49:98] = V44[j * 4 + k, :]
    # W: C2[i, k] corresponds to W44[i, k]
    # W index is (k+4)*n + i = (k+4)*4 + i
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:98] = W44[k * 4 + i, :]
            
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x8 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Try rank-98 first (two rank-49 halves)
    try:
        U, V, W = _build_4x4x8_rank98()
        if _verify_decomposition(U, V, W, n, m, p):
            if time.time() < deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition (rank 128)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Optional: Test locally
if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"Valid: {_verify_decomposition(U, V, W, 4, 4, 8)}")
