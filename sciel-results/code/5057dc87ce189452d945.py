import numpy as np
import time

DIMENSIONS = [3, 3, 4]

_DEADLINE = 150.0
_START = None

def _remaining():
    global _START
    if _START is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _is_valid(n, m, p, U, V, W, tol=1e-9):
    """Check exact decomposition of the MM tensor with the official W trace convention."""
    if not U or not V or not W:
        return False
    R = len(U[0])
    if len(V[0]) != R or len(W[0]) != R:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    
    # Reshape for efficient tensor contraction
    # U: (n*m, R) -> (n, m, R)
    # V: (m*p, R) -> (m, p, R)
    # W: (p*n, R) -> (p, n, R) -> transpose to (n, p, R) to match indices?
    # Convention: W[k*n+i, r]. So W_row = k*n + i.
    # We want to compute sum_r U[i,j,r] * V[j,k,r] * W[k,n+i,r]
    
    U_reshaped = Ua.reshape(n, m, R)
    V_reshaped = Va.reshape(m, p, R)
    W_reshaped = Wa.reshape(p, n, R)
    
    # Target tensor T[i, j, k] = 1 if j==j (wait, MM tensor is T[i, j, k] = delta_{j, j'}? No.)
    # Matrix Multiplication Tensor: C[i, k] = sum_j A[i, j] B[j, k].
    # Tensor T is such that C[i, k] = sum_{j, r} A[i, j] B[j, k] W_{i, k, r} U_{i, j, r} V_{j, k, r}?
    # The decomposition is: T_{i, k, j} = sum_r U_{i, j, r} V_{j, k, r} W_{i, k, r}
    # Here indices are:
    # U: row i*m+j, col r. So U[i, j, r]
    # V: row j*p+k, col r. So V[j, k, r]
    # W: row k*n+i, col r. So W[k, i, r]
    
    # We need sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] == 1 if j==j (always 1 for the diagonal? No.)
    # The MM tensor is T[i, k, j] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Actually, the tensor is defined by T_{i, k, j} = \delta_{j, j} ? No.
    # The tensor T is the structure tensor of MM.
    # T_{i, k, j} = 1 if the output C[i,k] depends on A[i,j] and B[j,k].
    # Since C[i,k] = sum_j A[i,j] B[j,k], the coefficient of A[i,j]B[j,k] in C[i,k] is 1.
    # So T[i, k, j] = 1 for all i, k, j?
    # Wait, the tensor is a 3-tensor. The indices are (i, k, j).
    # The value is 1 if the pair (A[i,j], B[j,k]) contributes to C[i,k].
    # Yes, for every i, j, k, the term A[i,j]B[j,k] contributes to C[i,k].
    # So the target tensor is all ones?
    # No, the "tensor" of matrix multiplication is usually defined such that
    # C_{ik} = \sum_{j} A_{ij} B_{jk}.
    # The decomposition is C_{ik} = \sum_{r} U_{i, j, r} V_{j, k, r} W_{i, k, r} ???
    # No, the standard decomposition is:
    # C_{ik} = \sum_{j} A_{ij} B_{jk}
    # Rank R decomposition:
    # C_{ik} = \sum_{r=1}^R (\sum_j U_{i, j, r} A_{ij}) (\sum_k V_{j, k, r} B_{jk}) W_{i, k, r} ? No.
    
    # Let's look at the seed code's `_is_valid`:
    # val = sum_r Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
    # It checks if val == 1.0.
    # This implies that for each fixed i, j, k, the sum must be 1.
    # This corresponds to the tensor T[i, j, k] = 1.
    # This is correct for the MM tensor where the "indices" of the tensor are (i, j, k) and the value is 1.
    # Wait, if T[i,j,k]=1 for all i,j,k, then the tensor is just the all-ones tensor.
    # But MM is not the all-ones tensor.
    # Let's re-read the convention.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # The decomposition is:
    # \sum_r U_{i, j, r} V_{j, k, r} W_{i, k, r} = \delta_{j, j} ?
    # No, the MM tensor is T_{i, k, j} = 1.
    # But the code sums over r for a fixed i, j, k.
    # If the result is 1 for all i, j, k, then the tensor is all ones.
    # Is the MM tensor all ones?
    # C_{ik} = \sum_j A_{ij} B_{jk}.
    # If we set A_{ij} = \delta_{i, i'} \delta_{j, j'} and B_{jk} = \delta_{j, j''} \delta_{k, k'},
    # Then C_{ik} is non-zero only if j'=j'' and i=i', k=k'.
    # The tensor T is defined by the bilinear map.
    # The rank is the minimum R such that T = \sum_r u_r \otimes v_r \otimes w_r.
    # The entries of T are T_{i, j, k}.
    # For MM, T_{i, j, k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # Since C_{ik} = \sum_j A_{ij} B_{jk}, every term A_{ij} B_{jk} contributes to C_{ik}.
    # So T_{i, j, k} = 1 for all valid i, j, k.
    # So yes, the target tensor is all ones.
    
    # So the check is:
    # For all i, j, k: sum_r U[i, j, r] V[j, k, r] W[k, i, r] == 1.
    
    # My vectorized check:
    # U_reshaped: (n, m, R)
    # V_reshaped: (m, p, R)
    # W_reshaped: (p, n, R)
    
    # We need to compute S[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # This is a batched tensor contraction.
    
    # Loop over j (m=3)
    # For fixed j:
    # U_j = U_reshaped[:, j, :]  # (n, R)
    # V_j = V_reshaped[j, :, :]  # (p, R)
    # We need S[i, k] = sum_r U_j[i, r] V_j[k, r] W[k, i, r]
    # Note W is (p, n, R). W[k, i, r].
    # So for fixed j, we have matrices U_j (n x R), V_j (p x R).
    # And W is a set of n x p matrices W_{r} where W_{r}[k, i] = W_reshaped[k, i, r].
    # S_j[i, k] = sum_r U_j[i, r] V_j[k, r] W_r[k, i]
    
    # This can be computed as:
    # Let M_r = U_j[:, r] * V_j[:, r]  # outer product (n x p)
    # Then S_j = sum_r M_r * W_r  (element-wise product and sum)
    
    # Vectorized:
    # U_j: (n, R)
    # V_j: (p, R)
    # Outer = U_j[:, :, None] * V_j[None, :, :]  # (n, p, R)
    # W_reshaped: (p, n, R) -> transpose to (n, p, R)
    # W_trans = W_reshaped.transpose(1, 0, 2) # (n, p, R)
    # S_j = np.sum(Outer * W_trans, axis=2) # (n, p)
    
    # Check if S_j is all ones.
    
    for j in range(m):
        U_j = U_reshaped[:, j, :]  # (n, R)
        V_j = V_reshaped[j, :, :]  # (p, R)
        W_trans = W_reshaped.transpose(1, 0, 2) # (n, p, R)
        
        # Outer product of U_j and V_j along R
        # U_j: (n, R) -> (n, 1, R)
        # V_j: (p, R) -> (1, p, R)
        outer = U_j[:, np.newaxis, :] * V_j[np.newaxis, :, :] # (n, p, R)
        
        # Element-wise multiply with W_trans and sum over R
        S_j = np.sum(outer * W_trans, axis=2) # (n, p)
        
        if not np.allclose(S_j, 1.0, atol=tol):
            return False
            
    return True

def _construct_rank34():
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    c = 0

    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal c
        if c >= rank:
            return
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val
        c += 1

    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal c
        # M1 = (a00+a11)(b00+b11)
        U[a00][c] += 1; U[a11][c] += 1
        V[b00][c] += 1; V[b11][c] += 1
        W[c00][c] += 1; W[c11][c] += 1
        c += 1
        # M2 = (a10+a11)b00
        U[a10][c] += 1; U[a11][c] += 1
        V[b00][c] += 1
        W[c10][c] += 1; W[c11][c] += -1
        c += 1
        # M3 = a00(b01-b11)
        U[a00][c] += 1
        V[b01][c] += 1; V[b11][c] += -1
        W[c01][c] += 1; W[c11][c] += 1
        c += 1
        # M4 = a11(b10-b00)
        U[a11][c] += 1
        V[b10][c] += 1; V[b00][c] += -1
        W[c00][c] += 1; W[c10][c] += 1
        c += 1
        # M5 = (a00+a01)b11
        U[a00][c] += 1; U[a01][c] += 1
        V[b11][c] += 1
        W[c00][c] += -1; W[c01][c] += 1
        c += 1
        # M6 = (a10-a00)(b00+b01)
        U[a10][c] += 1; U[a00][c] += -1
        V[b00][c] += 1; V[b01][c] += 1
        W[c11][c] += 1
        c += 1
        # M7 = (a01-a11)(b10+b11)
        U[a01][c] += 1; U[a11][c] += -1
        V[b10][c] += 1; V[b11][c] += 1
        W[c00][c] += 1
        c += 1

    def c_idx(i, k): return k * n + i
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k

    # Strassen for A11 (i=0,1; j=0,1) and B11 (j=0,1; k=0,1)
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
                 v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1),
                 c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1))

    # Strassen for A11 (i=0,1; j=0,1) and B12 (j=0,1; k=2,3)
    add_strassen(u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
                 v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3),
                 c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3))

    # Schoolbook for A12 (i=0,1; j=2) and B21 (j=2; k=0,1)
    add_term(u_idx(0,2), 1, v_idx(2,0), 1, c_idx(0,0), 1)
    add_term(u_idx(0,2), 1, v_idx(2,1), 1, c_idx(0,1), 1)
    add_term(u_idx(1,2), 1, v_idx(2,0), 1, c_idx(1,0), 1)
    add_term(u_idx(1,2), 1, v_idx(2,1), 1, c_idx(1,1), 1)

    # Schoolbook for A12 (i=0,1; j=2) and B22 (j=2; k=2,3)
    add_term(u_idx(0,2), 1, v_idx(2,2), 1, c_idx(0,2), 1)
    add_term(u_idx(0,2), 1, v_idx(2,3), 1, c_idx(0,3), 1)
    add_term(u_idx(1,2), 1, v_idx(2,2), 1, c_idx(1,2), 1)
    add_term(u_idx(1,2), 1, v_idx(2,3), 1, c_idx(1,3), 1)

    # Schoolbook for A21 (i=2; j=0,1) and B11 (j=0,1; k=0,1)
    add_term(u_idx(2,0), 1, v_idx(0,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,0), 1, v_idx(0,1), 1, c_idx(2,1), 1)
    add_term(u_idx(2,1), 1, v_idx(1,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,1), 1, v_idx(1,1), 1, c_idx(2,1), 1)

    # Schoolbook for A21 (i=2; j=0,1) and B12 (j=0,1; k=2,3)
    add_term(u_idx(2,0), 1, v_idx(0,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,0), 1, v_idx(0,3), 1, c_idx(2,3), 1)
    add_term(u_idx(2,1), 1, v_idx(1,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,1), 1, v_idx(1,3), 1, c_idx(2,3), 1)

    # Schoolbook for A22 (i=2; j=2) and B21 (j=2; k=0,1)
    add_term(u_idx(2,2), 1, v_idx(2,0), 1, c_idx(2,0), 1)
    add_term(u_idx(2,2), 1, v_idx(2,1), 1, c_idx(2,1), 1)

    # Schoolbook for A22 (i=2; j=2) and B22 (j=2; k=2,3)
    add_term(u_idx(2,2), 1, v_idx(2,2), 1, c_idx(2,2), 1)
    add_term(u_idx(2,2), 1, v_idx(2,3), 1, c_idx(2,3), 1)

    return {"U": U, "V": V, "W": W}

def _try_lower_rank(R, n, m, p, deadline):
    """Attempt to find an exact rank-R decomposition via iterative least-squares with
    rational rounding. Returns dict or None."""
    def _solve_rank(R):
        np.random.seed(42)
        U = np.random.randn(n, m, R) * 0.1
        V = np.random.randn(m, p, R) * 0.1
        W = np.random.randn(p, n, R) * 0.1

        max_iters = 100
        tol = 1e-10

        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j1 in range(m):
                    # A is (p*m, R), b is (p*m,)
                    # A[idx, :] = W[:, i, k] * V[:, j2, k]
                    # b[idx] = 1.0 if j1 == j2 else 0.0
                    # We can construct A more efficiently.
                    # For each k, j2:
                    # Row corresponds to (k, j2).
                    # A[k*m + j2, :] = W[k, i, :] * V[j2, k, :]
                    
                    A = np.zeros((p * m, R))
                    b = np.zeros(p * m)
                    idx = 0
                    for k in range(p):
                        for j2 in range(m):
                            A[idx, :] = W[k, i, :] * V[j2, k, :]
                            b[idx] = 1.0 if j1 == j2 else 0.0
                            idx += 1
                    try:
                        U[i, j1, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        return None

            # Update V
            for j in range(m):
                for k in range(p):
                    # A is (n*m*p, R) ? No, we fix j, k.
                    # We want to find V[j, k, :]
                    # Equation: sum_r U[i, j1, r] V[j, k, r] W[k, i, r] = delta_{j1, j}
                    # This is linear in V[j, k, :].
                    # Let v = V[j, k, :].
                    # sum_r v_r * (U[i, j1, r] * W[k, i, r]) = delta_{j1, j}
                    # A[row, r] = U[i, j1, r] * W[k, i, r]
                    # b[row] = delta_{j1, j}
                    # row index: (i, j1)
                    
                    A = np.zeros((n * m, R))
                    b = np.zeros(n * m)
                    idx = 0
                    for i in range(n):
                        for j1 in range(m):
                            A[idx, :] = U[i, j1, :] * W[k, i, :]
                            b[idx] = 1.0 if j1 == j else 0.0
                            idx += 1
                    try:
                        V[j, k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        return None

            # Update W
            for i in range(n):
                for k in range(p):
                    # Equation: sum_r U[i, j1, r] V[j2, k, r] W[k, i, r] = delta_{j1, j2}
                    # Linear in W[k, i, :].
                    # A[row, r] = U[i, j1, r] * V[j2, k, r]
                    # b[row] = delta_{j1, j2}
                    # row index: (j1, j2)
                    
                    A = np.zeros((m * m, R))
                    b = np.zeros(m * m)
                    idx = 0
                    for j1 in range(m):
                        for j2 in range(m):
                            A[idx, :] = U[i, j1, :] * V[j2, k, :]
                            b[idx] = 1.0 if j1 == j2 else 0.0
                            idx += 1
                    try:
                        W[k, i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        return None

            # Convergence check
            err = 0.0
            for j in range(m):
                U_j = U[:, j, :]
                V_j = V[j, :, :]
                W_trans = W.transpose(1, 0, 2) # (n, p, R)
                outer = U_j[:, np.newaxis, :] * V_j[np.newaxis, :, :]
                S_j = np.sum(outer * W_trans, axis=2)
                err += np.sum((S_j - 1.0) ** 2)
            
            if err < tol:
                break

        # Rational rounding to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2

        # Verify exactness
        err = 0.0
        for j in range(m):
            U_j = U_r[:, j, :]
            V_j = V_r[j, :, :]
            W_trans = W_r.transpose(1, 0, 2)
            outer = U_j[:, np.newaxis, :] * V_j[np.newaxis, :, :]
            S_j = np.sum(outer * W_trans, axis=2)
            err += np.sum((S_j - 1.0) ** 2)
        
        if err > 1e-9:
            return None

        U_out = U_r.reshape(n * m, R)
        V_out = V_r.reshape(m * p, R)
        W_out = np.zeros((p * n, R))
        for r in range(R):
            for i in range(n):
                for k in range(p):
                    W_out[k * n + i, r] = W_r[k, i, r]
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

    return _solve_rank(R)

def construct():
    global _START
    _START = time.time()
    n, m, p = DIMENSIONS

    # Guaranteed rank-34 construction
    fallback = _construct_rank34()
    if not _is_valid(n, m, p, fallback["U"], fallback["V"], fallback["W"]):
        return _schoolbook()

    # Try to find a lower rank with a tight deadline guard
    deadline = _START + 140.0  # 10s safety margin
    best = fallback
    for R in range(33, 19, -1):
        if time.time() > deadline:
            break
        res = _try_lower_rank(R, n, m, p, deadline)
        if res is not None and _is_valid(n, m, p, res["U"], res["V"], res["W"]):
            return res

    return best
