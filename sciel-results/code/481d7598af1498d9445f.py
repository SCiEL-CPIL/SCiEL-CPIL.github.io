# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t0 = time.time()
    n, m, p = 2, 3, 10

    def deadline_ok():
        return (time.time() - t0) < 135.0

    # Build rank-55 Strassen + schoolbook decomposition
    rank0 = 55
    U_mat = [[0] * rank0 for _ in range(n * m)]
    V_mat = [[0] * rank0 for _ in range(m * p)]
    W_mat = [[0] * rank0 for _ in range(p * n)]

    terms = []

    # Strassen for 5 pairs of columns in B (rows 0,1 of A)
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        a00, a01, a10, a11 = 0, 1, 3, 4
        b0k0, b0k1 = 0 * 10 + k0, 0 * 10 + k1
        b1k0, b1k1 = 1 * 10 + k0, 1 * 10 + k1
        c0k0, c0k1 = k0 * 2 + 0, k1 * 2 + 0
        c1k0, c1k1 = k0 * 2 + 1, k1 * 2 + 1

        terms.append(([(a00, 1), (a11, 1)], [(b0k0, 1), (b1k1, 1)], [(c0k0, 1), (c1k1, 1)]))
        terms.append(([(a10, 1), (a11, 1)], [(b0k0, 1)], [(c1k0, 1), (c1k1, -1)]))
        terms.append(([(a00, 1)], [(b0k1, 1), (b1k1, -1)], [(c0k1, 1), (c1k1, 1)]))
        terms.append(([(a11, 1)], [(b1k0, 1), (b0k0, -1)], [(c0k0, 1), (c1k0, 1)]))
        terms.append(([(a00, 1), (a01, 1)], [(b1k1, 1)], [(c0k0, -1), (c0k1, 1)]))
        terms.append(([(a10, 1), (a00, -1)], [(b0k0, 1), (b0k1, 1)], [(c1k1, 1)]))
        terms.append(([(a01, 1), (a11, -1)], [(b1k0, 1), (b1k1, 1)], [(c0k0, 1)]))

    # Schoolbook for third row of A
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            terms.append(([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]))

    for t_idx, (u_terms, v_terms, w_terms) in enumerate(terms):
        for idx, coeff in u_terms:
            U_mat[idx][t_idx] = coeff
        for idx, coeff in v_terms:
            V_mat[idx][t_idx] = coeff
        for idx, coeff in w_terms:
            W_mat[idx][t_idx] = coeff

    U_arr = np.array(U_mat, dtype=float)
    V_arr = np.array(V_mat, dtype=float)
    W_arr = np.array(W_mat, dtype=float)
    best_rank = rank0

    # Verify the decomposition is correct
    def verify(U, V, W):
        T = np.einsum('it,jt,kt->ijk', U, V, W)
        # Check against the actual matrix multiplication tensor
        # T[i,j,k] should be sum_l A[i,l]*B[l,j] where i indexes (n*m), j indexes (m*p), k indexes (p*n)
        # Actually the tensor for matrix mult C = A*B where A is n x m, B is m x p:
        # C[i,k] = sum_l A[i,l] * B[l,k]
        # In flattened form: A has index (i,l) -> i*n+m... wait
        # U has shape (n*m, rank) -> index i = row*n + col (of A)
        # V has shape (m*p, rank) -> index j = row*p + col (of B)  
        # W has shape (p*n, rank) -> index k = col*n + row (of C)
        # T[i,j,k] = sum_t U[i,t]*V[j,t]*W[k,t]
        # This should equal: if i = a*n+la, j = lb*p+b, k = b*n+a then T = A[a,la]*B[lb,b] when la==lb
        # So T[i,j,k] = delta(la,lb) * (A[a,la] * B[lb,b])
        # For the identity tensor: T[i,j,k] = 1 if la==lb and the "matching" condition holds
        # Actually for matrix multiplication, the tensor is:
        # T[a,la, lb,b, b,a] = delta(la,lb)
        # In flattened indices: i=a*n+la, j=lb*p+b, k=b*n+a
        # T[i,j,k] = 1 if la == lb, else 0
        # Let's verify
        for a in range(n):
            for la in range(m):
                for lb in range(m):
                    for b in range(p):
                        i = a * m + la
                        j = lb * p + b
                        k = b * n + a
                        expected = 1.0 if la == lb else 0.0
                        if abs(T[i, j, k] - expected) > 1e-9:
                            return False
        return True

    if not verify(U_arr, V_arr, W_arr):
        # Fallback: just return the initial construction
        return {'U': U_mat, 'V': V_mat, 'W': W_mat}

    # Try to reduce rank by finding linearly dependent rank-1 tensors
    def try_remove_term(U, V, W, t_remove):
        """Try to express term t_remove as a linear combination of other terms."""
        r = U.shape[1]
        if r <= 1:
            return None
        
        term_vec = (U[:, t_remove] * V[:, t_remove] * W[:, t_remove]).flatten()
        
        other_idx = [i for i in range(r) if i != t_remove]
        # Build matrix of other terms
        A_mat = np.zeros((U.shape[0] * V.shape[0] * W.shape[0], len(other_idx)))
        for idx, t in enumerate(other_idx):
            A_mat[:, idx] = (U[:, t] * V[:, t] * W[:, t]).flatten()
        
        try:
            coeff, _, rank_, _ = np.linalg.lstsq(A_mat, term_vec, rcond=1e-10)
            residual = np.linalg.norm(A_mat @ coeff - term_vec)
        except:
            return None
        
        if residual < 1e-8:
            new_U = np.delete(U, t_remove, axis=1)
            new_V = np.delete(V, t_remove, axis=1)
            new_W = np.delete(W, t_remove, axis=1)
            return (new_U, new_V, new_W)
        return None

    # Iteratively try to remove terms
    for iteration in range(30):
        if not deadline_ok():
            break
        
        removed = False
        r = U_arr.shape[1]
        
        for t_remove in range(r):
            if not deadline_ok():
                break
            
            result = try_remove_term(U_arr, V_arr, W_arr, t_remove)
            if result is not None:
                U_arr, V_arr, W_arr = result
                best_rank = U_arr.shape[1]
                removed = True
                break
        
        if not removed:
            break

    # Final verification
    if not verify(U_arr, V_arr, W_arr):
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        best_rank = rank0

    return {'U': U_arr.tolist(), 'V': V_arr.tolist(), 'W': W_arr.tolist()}
# EVOLVE-BLOCK-END
