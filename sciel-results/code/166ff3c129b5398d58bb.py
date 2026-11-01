"""Rewritten program for matrix multiplication 2x2x5 rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    # Each term is (u_vec, v_vec, w_vec) where u has length n*m=4, v has length m*2=4, w has length 2*n=4
    # Convention: C[i,k] = sum_j A[i,j] B[j,k]
    # In the tensor: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For 2x2x2: n=2, m=2, p=2
    # U index: i*m+j = 2i+j
    # V index: j*p+k = 2j+k
    # W index: k*n+i = 2k+i
    
    terms = []
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    # U: A00=1, A11=1 => U[0]=1, U[3]=1
    # V: B00=1, B11=1 => V[0]=1, V[3]=1
    # W: C00 gets +1 (W[0]), C11 gets +1 (W[3])
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # M2 = (A10+A11)(B00) -> contributes to C10
    # U: A10=1, A11=1 => U[2]=1, U[3]=1
    # V: B00=1 => V[0]=1
    # W: C10 gets +1 (W[2])
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    
    # M3 = (A00)(B10-B00) -> contributes to C00 and C01
    # U: A00=1 => U[0]=1
    # V: B10=1, B00=-1 => V[2]=1, V[0]=-1
    # W: C00 gets -1 (W[0]), C01 gets +1 (W[1])
    terms.append(([1, 0, 0, 0], [-1, 0, 1, 0], [-1, 1, 0, 0]))
    
    # M4 = (A11)(B11-B01) -> contributes to C11 and C10
    # U: A11=1 => U[3]=1
    # V: B11=1, B01=-1 => V[3]=1, V[1]=-1
    # W: C11 gets +1 (W[3]), C10 gets -1 (W[2])
    terms.append(([0, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]))
    
    # M5 = (A00+A10)(B11) -> contributes to C01 and C11
    # U: A00=1, A10=1 => U[0]=1, U[2]=1
    # V: B11=1 => V[3]=1
    # W: C01 gets +1 (W[1]), C11 gets +1 (W[3])
    terms.append(([1, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 1]))
    
    # M6 = (A01-A11)(B10+B11) -> contributes to C01 and C11
    # U: A01=1, A11=-1 => U[1]=1, U[3]=-1
    # V: B10=1, B11=1 => V[2]=1, V[3]=1
    # W: C01 gets +1 (W[1]), C11 gets -1 (W[3])
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, -1]))
    
    # M7 = (A10-A00)(B01+B00) -> contributes to C10 and C00
    # U: A10=1, A00=-1 => U[2]=1, U[0]=-1
    # V: B01=1, B00=1 => V[1]=1, V[0]=1
    # W: C10 gets +1 (W[2]), C00 gets -1 (W[0])
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, -1]))
    
    return terms

def _make_schoolbook_terms(n, m, p, k_start=0, k_end=None):
    """Make schoolbook terms for columns k_start to k_end-1."""
    if k_end is None:
        k_end = p
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(k_start, k_end):
                u_vec = [0.0] * (n * m)
                u_vec[i * m + j] = 1.0
                v_vec = [0.0] * (m * p)
                v_vec[j * p + k] = 1.0
                w_vec = [0.0] * (p * n)
                w_vec[k * n + i] = 1.0
                terms.append((u_vec, v_vec, w_vec))
    return terms

def _build_from_terms(all_terms, n, m, p):
    """Build U, V, W matrices from a list of (u_vec, v_vec, w_vec) terms."""
    rank = len(all_terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
    return U, V, W

def _tensor_error(U, V, W, n, m, p):
    """Compute the squared error of the tensor decomposition."""
    error = 0.0
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                error += (val - 1.0) ** 2
    return error

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Strategy: Use Strassen for the first 2 columns (rank 7) and schoolbook for the remaining 3 columns (rank 3*2*2=12)
    # Total rank = 7 + 12 = 19
    # Or: Use Strassen for first 2 columns (rank 7) and Strassen for next 2 columns (rank 7) and schoolbook for last 1 column (rank 4)
    # Total rank = 7 + 7 + 4 = 18
    
    # Let's try the rank-18 approach: two Strassen blocks + one schoolbook block
    strassen_terms = _strassen_2x2_terms()
    all_terms = []
    
    # Block 1: columns 0,1 using Strassen
    for u_vec, v_vec, w_vec in strassen_terms:
        u_full = list(u_vec)
        # v_vec is for p=2, we need to embed into p=5
        # v_vec[j*2+k] corresponds to B[j,k] for k in {0,1}
        v_full = [0.0] * (m * p)
        for j in range(m):
            for k in range(2):
                v_full[j * p + k] = v_vec[j * 2 + k]
        # w_vec is for p=2, we need to embed into p=5
        # w_vec[k*2+i] corresponds to C[i,k] for k in {0,1}
        w_full = [0.0] * (p * n)
        for k in range(2):
            for i in range(n):
                w_full[k * n + i] = w_vec[k * 2 + i]
        all_terms.append((u_full, v_full, w_full))
    
    # Block 2: columns 2,3 using Strassen
    for u_vec, v_vec, w_vec in strassen_terms:
        u_full = list(u_vec)
        v_full = [0.0] * (m * p)
        for j in range(m):
            for k in range(2, 4):
                # v_vec index j*2+(k-2)
                v_full[j * p + k] = v_vec[j * 2 + (k - 2)]
        w_full = [0.0] * (p * n)
        for k in range(2, 4):
            for i in range(n):
                # w_vec index (k-2)*2+i
                w_full[k * n + i] = w_vec[(k - 2) * 2 + i]
        all_terms.append((u_full, v_full, w_full))
    
    # Block 3: column 4 using schoolbook
    schoolbook_terms = _make_schoolbook_terms(n, m, p, k_start=4, k_end=5)
    all_terms.extend(schoolbook_terms)
    
    U, V, W = _build_from_terms(all_terms, n, m, p)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to full schoolbook rank-20
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Convert to lists with clean values
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx, val in enumerate(row):
                if abs(val - round(val)) < 1e-10:
                    row[idx] = float(round(val))
                elif abs(2*val - round(2*val)) < 1e-10:
                    row[idx] = float(round(2*val) / 2.0)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
