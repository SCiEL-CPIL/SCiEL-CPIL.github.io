import numpy as np
import time

DIMENSIONS = [2, 3, 10]
_TIME_LIMIT = 130  # safety margin before 150s kill

def _strassen_2x2x2():
    """Return the 7 rank-1 tensors for M_{2,2,2} using Strassen's algorithm.
    
    U indices: i*m+j, V indices: j*p+k, W indices: k*n+i
    """
    products = [
        # M0 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M1 = (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        # M2 = A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        # M3 = A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # M4 = (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # M5 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M6 = (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    return products

def _build_rank55():
    """Build the known rank-55 Strassen-based decomposition.
    
    Split m=3 into m1=2, m2=1.
    For M_{2,2,10}: tile p into 5 blocks of 2, each uses Strassen (7 mults) = 35
    For M_{2,1,10}: schoolbook = 2*1*10 = 20
    Total: 55
    """
    n, m, p = DIMENSIONS
    total_rank = 55
    U = [[0.0] * total_rank for _ in range(n * m)]  # 6 x 55
    V = [[0.0] * total_rank for _ in range(m * p)]  # 30 x 55
    W = [[0.0] * total_rank for _ in range(p * n)]  # 20 x 55
    
    strassen_products = _strassen_2x2x2()
    
    # M_{2,2,10}: for each block of 2 columns in p
    for b in range(5):
        base_rank = b * 7
        for t, (u_dict, v_dict, w_dict) in enumerate(strassen_products):
            r = base_rank + t
            # U mapping: local index li -> global index i*m+j where li=i*2+j
            for li, val in u_dict.items():
                i = li // 2
                j = li % 2
                gi = i * m + j
                U[gi][r] += val
            # V mapping: local index lv -> global index j*p+(2b+k) where lv=j*2+k
            for lv, val in v_dict.items():
                j = lv // 2
                k = lv % 2
                gv = j * p + (2 * b + k)
                V[gv][r] += val
            # W mapping: local index lw -> global index (2b+k)*n+i where lw=k*2+i
            for lw, val in w_dict.items():
                k = lw // 2
                i = lw % 2
                gw = (2 * b + k) * n + i
                W[gw][r] += val
    
    # M_{2,1,10}: schoolbook
    base_rank = 35
    col = base_rank
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1.0
            V[2 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    T = np.einsum('at,bt,ct->abc', U_np, V_np, W_np)
    
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m + j, j*p + k, k*n + i] = 1.0
    
    diff = np.max(np.abs(T - expected))
    return diff < 1e-9

def _try_find_rank54(U, V, W, n, m, p, deadline):
    """
    Try to find a rank-54 solution by looking for linear dependencies
    among the 55 rank-1 tensors that allow dropping one.
    """
    rank = 55
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    def flatten_term(u_col, v_col, w_col):
        vec = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*m*p + j*p + k] = u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        return vec
    
    all_terms = []
    for r in range(rank):
        all_terms.append(flatten_term(U_np[:, r], V_np[:, r], W_np[:, r]))
    
    for r0 in range(rank):
        if time.time() > deadline:
            break
        
        term_r0 = all_terms[r0]
        others = np.array([all_terms[r] for r in range(rank) if r != r0])
        
        try:
            x, residuals, rank_overs, sv = np.linalg.lstsq(others.T, term_r0, rcond=None)
            residual = np.linalg.norm(others.T @ x - term_r0)
            if residual < 1e-10:
                new_U = [U[r] for r in range(rank) if r != r0]
                new_V = [V[r] for r in range(rank) if r != r0]
                new_W = [W[r] for r in range(rank) if r != r0]
                if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                    return new_U, new_V, new_W
        except:
            pass
    
    return None

def _try_border_rank_lifting(U, V, W, n, m, p, deadline):
    """
    Try border rank lifting: parameterize a rank-54 decomposition with symbolic variable epsilon
    such that tensor equations hold in Q(epsilon), then clear denominators.
    
    Strategy: For each pair of terms, try to merge them into one term by finding
    a common factor structure. This is equivalent to checking if two rank-1 tensors
    can be combined into one rank-1 tensor in the border rank sense.
    """
    rank = 55
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Try to find if any two terms can be merged
    # Two rank-1 tensors u1⊗v1⊗w1 and u2⊗v2⊗w2 can be merged into one
    # if they share a common structure.
    
    # For now, try a simpler approach: check if the sum of two terms is rank-1
    # in the tensor space.
    
    for i in range(rank):
        if time.time() > deadline:
            break
        for j in range(i+1, rank):
            if time.time() > deadline:
                break
            
            # Check if u_i⊗v_i⊗w_i + u_j⊗v_j⊗w_j is rank-1
            # This is equivalent to checking if the 6x6 matrix formed by
            # the two terms has rank 1.
            
            # Flatten the two terms
            def flatten_term_idx(idx):
                vec = np.zeros(n*m*p)
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            vec[a*(m*p)*(p*n) + b*(p*n) + c] = U_np[a, idx] * V_np[b, idx] * W_np[c, idx]
                return vec
            
            term_i = flatten_term_idx(i)
            term_j = flatten_term_idx(j)
            combined = term_i + term_j
            
            # Check if combined is rank-1 by reshaping to a matrix
            # and checking if all 2x2 minors are zero
            # Reshape to (n*m, m*p*p*n)
            mat = combined.reshape(n*m, m*p*p*n)
            
            # Check rank
            try:
                r = np.linalg.matrix_rank(mat, tol=1e-10)
                if r <= 1:
                    # The combined term is rank-1!
                    # Extract the rank-1 factorization
                    # Find the dominant singular vector
                    U_svd, S, Vt = np.linalg.svd(mat, full_matrices=False)
                    if S[0] > 1e-10:
                        u_vec = U_svd[:, 0] * S[0]
                        v_vec = Vt[0]
                        
                        # Now we need to factor v_vec into v⊗w
                        # v_vec has shape (m*p*p*n,)
                        # We need v_vec[b,c] = V[b]*W[c] for some V, W
                        # This is a rank-1 matrix factorization problem
                        v_mat = v_vec.reshape(m*p, p*n)
                        U_svd2, S2, Vt2 = np.linalg.svd(v_mat, full_matrices=False)
                        if S2[0] > 1e-10 and (len(S2) == 1 or S2[1] < 1e-10 * S2[0]):
                            v_vec2 = U_svd2[:, 0] * S2[0]
                            w_vec2 = Vt2[0]
                            
                            # Now we have u_vec, v_vec2, w_vec2
                            # Remove terms i and j, add the new term
                            new_terms = [r for r in range(rank) if r != i and r != j]
                            new_U = [U_np[:, r] for r in new_terms]
                            new_V = [V_np[:, r] for r in new_terms]
                            new_W = [W_np[:, r] for r in new_terms]
                            
                            new_U.append(u_vec)
                            new_V.append(v_vec2)
                            new_W.append(w_vec2)
                            
                            new_U_np = np.array(new_U)
                            new_V_np = np.array(new_V)
                            new_W_np = np.array(new_W)
                            
                            # Verify
                            T = np.einsum('at,bt,ct->abc', new_U_np, new_V_np, new_W_np)
                            expected = np.zeros((n*m, m*p, p*n))
                            for ii in range(n):
                                for jj in range(m):
                                    for kk in range(p):
                                        expected[ii*m + jj, jj*p + kk, kk*n + ii] = 1.0
                            
                            if np.max(np.abs(T - expected)) < 1e-9:
                                # Convert back to list format
                                new_U_list = new_U_np.tolist()
                                new_V_list = new_V_np.tolist()
                                new_W_list = new_W_np.tolist()
                                return new_U_list, new_V_list, new_W_list
            except:
                pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Build the rank-55 baseline
    U, V, W = _build_rank55()
    
    # Verify the baseline
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a rank-54 solution via linear dependency
    result = _try_find_rank54(U, V, W, n, m, p, deadline)
    if result is not None:
        new_U, new_V, new_W = result
        return {"U": new_U, "V": new_V, "W": new_W}
    
    # Try border rank lifting
    result = _try_border_rank_lifting(U, V, W, n, m, p, deadline)
    if result is not None:
        new_U, new_V, new_W = result
        return {"U": new_U, "V": new_V, "W": new_W}
    
    # Fall back to rank 55
    return {"U": U, "V": V, "W": W}
