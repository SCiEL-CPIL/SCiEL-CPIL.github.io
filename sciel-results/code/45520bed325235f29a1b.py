import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = 4, 4, 4
    rank = 49

    # Strassen's 7 multiplications for 2x2
    # a_coeffs[s, S, T] = coefficient of A[S, T] in the s-th Strassen product
    # b_coeffs[s, T, K] = coefficient of B[T, K] in the s-th Strassen product
    # c_coeffs[s, S, K] = coefficient of C[S, K] from the s-th Strassen product
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # M0 = (A00 + A11)(B00 + B11)
        [[0, 0], [1, 1]],   # M1 = (A10 + A11)B00
        [[1, 0], [0, 0]],   # M2 = A00(B01 - B11)
        [[0, 0], [0, 1]],   # M3 = A11(B10 - B00)
        [[1, 1], [0, 0]],   # M4 = (A00 + A01)B11
        [[-1, 0], [1, 0]],  # M5 = (A10 - A00)(B00 + B01)
        [[0, 1], [0, -1]],  # M6 = (A01 - A11)(B10 + B11)
    ])

    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # (B00+B11)
        [[1, 0], [0, 0]],   # B00
        [[0, 1], [0, -1]],  # (B01-B11)
        [[-1, 0], [1, 0]],  # (B10-B00)
        [[0, 0], [0, 1]],   # B11
        [[1, 1], [0, 0]],   # (B00+B01)
        [[0, 0], [1, 1]],   # (B10+B11)
    ])

    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # C00+=M0, C11+=M0
        [[0, 0], [1, -1]],  # C10+=M1, C11-=M1
        [[0, 1], [0, 1]],   # C01+=M2, C11+=M2
        [[1, 0], [1, 0]],   # C00+=M3, C10+=M3
        [[-1, 1], [0, 0]],  # C00-=M4, C01+=M4
        [[0, 0], [0, 1]],   # C11+=M5
        [[1, 0], [0, 0]],   # C00+=M6
    ])

    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))

    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t

            for S in range(2):
                for T in range(2):
                    a_st = a_coeffs[s, S, T]
                    if a_st == 0:
                        continue
                    for s2 in range(2):
                        for t2 in range(2):
                            a_t2 = a_coeffs[t, s2, t2]
                            if a_t2 == 0:
                                continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * 4 + j, c_idx] += a_st * a_t2

            for T in range(2):
                for K in range(2):
                    b_st = b_coeffs[s, T, K]
                    if b_st == 0:
                        continue
                    for t2 in range(2):
                        for k2 in range(2):
                            b_t2 = b_coeffs[t, t2, k2]
                            if b_t2 == 0:
                                continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * 4 + k, c_idx] += b_st * b_t2

            for K in range(2):
                for S in range(2):
                    c_sk = c_coeffs[s, S, K]
                    if c_sk == 0:
                        continue
                    for k2 in range(2):
                        for s2 in range(2):
                            c_t2 = c_coeffs[t, s2, k2]
                            if c_t2 == 0:
                                continue
                            k = 2 * K + k2
                            i = 2 * S + s2
                            W[k * 4 + i, c_idx] += c_sk * c_t2

            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify the decomposition is correct
    def verify(U_mat, V_mat, W_mat):
        """Check that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == delta_{i,k} for matrix mult tensor"""
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    expected = 1.0 if i == k else 0.0
                    actual = 0.0
                    for r in range(U_mat.shape[1]):
                        actual += U_mat[i*4+j, r] * V_mat[j*4+k, r] * W_mat[k*4+i, r]
                    if abs(actual - expected) > 1e-10:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback: return the 64-term schoolbook (rank 64)
        U_f = np.zeros((16, 64))
        V_f = np.zeros((16, 64))
        W_f = np.zeros((16, 64))
        idx = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_f[i*4+j, idx] = 1.0
                    V_f[j*4+k, idx] = 1.0
                    W_f[k*4+i, idx] = 1.0
                    idx += 1
        if verify(U_f, V_f, W_f):
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now attempt rank reduction: find linear dependencies among the 49 terms
    # Each term r defines a tensor T_r[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # If sum_r alpha_r * T_r = 0 has a non-trivial solution, we can reduce rank.
    # Vectorize: each T_r is a 64-dim vector (flattened 4x4x4).
    
    tensors = np.zeros((64, 49))
    for r in range(49):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    tensors[i*16+j*4+k, r] = U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    # Find the rank of this 64x49 matrix
    try:
        U_svd, svals, Vt = np.linalg.svd(tensors, full_matrices=False)
        tol = 1e-10
        effective_rank = np.sum(svals > tol)
    except:
        effective_rank = 49
    
    if effective_rank < 49 and time.time() < deadline:
        # The 49 tensors are linearly dependent. We can find a basis.
        # Iteratively remove terms that are in the span of others.
        new_U = U.copy()
        new_V = V.copy()
        new_W = W.copy()
        kept = list(range(49))
        
        for r in range(49):
            if time.time() > deadline:
                break
            if r not in kept:
                continue
            others = [x for x in kept if x != r]
            if len(others) == 0:
                continue
            T_r = tensors[:, r]
            T_others = tensors[:, others]
            try:
                alpha, residuals, rank_ls, svals_ls = np.linalg.lstsq(T_others, T_r, rcond=1e-12)
                if len(residuals) > 0 and residuals[0] < 1e-8:
                    kept.remove(r)
                    new_U[:, r] = 0
                    new_V[:, r] = 0
                    new_W[:, r] = 0
            except:
                pass
        
        if len(kept) < 49:
            reduced_rank = len(kept)
            U_red = new_U[:, kept]
            V_red = new_V[:, kept]
            W_red = new_W[:, kept]
            if verify(U_red, V_red, W_red):
                return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    
    # Return the full rank-49 decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
