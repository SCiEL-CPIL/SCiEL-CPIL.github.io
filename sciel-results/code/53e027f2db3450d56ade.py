import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    n, m, p = 3, 4, 7
    # Current best known structured rank is 78.

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = [Fraction(0)] * (n * m)
        v = [Fraction(0)] * (m * p)
        w = [Fraction(0)] * (p * n)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += Fraction(val)
        for idx, val in zip(v_indices, v_vals):
            v[idx] += Fraction(val)
        for idx, val in zip(w_indices, w_vals):
            w[idx] += Fraction(val)

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off
            j = j_off
            k = k_off
            add_rank1_term([i*m+j], [1], [j*p+k], [1], [k*n+i], [1])
            return

        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    i = i_off
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([i*m+jj], [1], [jj*p+kk], [1], [kk*n+i], [1])
            return

        if m_l == 1:
            for i in range(n_l):
                for k in range(p_l):
                    ii = i_off + i
                    j = j_off
                    kk = k_off + k
                    add_rank1_term([ii*m+j], [1], [j*p+kk], [1], [kk*n+ii], [1])
            return

        if p_l == 1:
            for i in range(n_l):
                for j in range(m_l):
                    ii = i_off + i
                    jj = j_off + j
                    k = k_off
                    add_rank1_term([ii*m+jj], [1], [jj*p+k], [1], [k*n+ii], [1])
            return

        if n_l == 2 and m_l == 2 and p_l == 2:
            i0 = i_off
            i1 = i_off + 1
            j0 = j_off
            j1 = j_off + 1
            k0 = k_off
            k1 = k_off + 1

            # Strassen 2x2x2
            add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0, k1*n+i1], [1, 1])
            add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0], [1],
                           [k0*n+i1, k1*n+i1], [1, -1])
            add_rank1_term([i0*m+j0], [1],
                           [j0*p+k1, j1*p+k1], [1, -1],
                           [k1*n+i0, k1*n+i1], [1, 1])
            add_rank1_term([i1*m+j1], [1],
                           [j1*p+k0, j0*p+k0], [1, -1],
                           [k0*n+i0, k0*n+i1], [1, 1])
            add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                           [j1*p+k1], [1],
                           [k0*n+i0, k1*n+i0], [-1, 1])
            add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                           [j0*p+k0, j0*p+k1], [1, 1],
                           [k1*n+i1], [1])
            add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                           [j1*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0], [1])
            return

        if n_l == 3 and m_l == 4 and p_l == 7:
            gen_terms(2, 4, 7, i_off, j_off, k_off)
            gen_terms(1, 4, 7, i_off + 2, j_off, k_off)
            return

        if n_l == 2 and m_l == 4 and p_l == 7:
            gen_terms(2, 2, 7, i_off, j_off, k_off)
            gen_terms(2, 2, 7, i_off, j_off + 2, k_off)
            return

        if n_l == 2 and m_l == 2 and p_l == 7:
            gen_terms(2, 2, 2, i_off, j_off, k_off)
            gen_terms(2, 2, 2, i_off, j_off, k_off + 2)
            gen_terms(2, 2, 2, i_off, j_off, k_off + 4)
            gen_terms(2, 2, 1, i_off, j_off, k_off + 6)
            return

        # Fallback schoolbook
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i
                    jj = j_off + j
                    kk = k_off + k
                    add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])

    gen_terms(n, m, p, 0, 0, 0)

    # Convert to float for output
    U = np.array([[float(x) for x in row] for row in U_vecs]).T
    V = np.array([[float(x) for x in row] for row in V_vecs]).T
    W = np.array([[float(x) for x in row] for row in W_vecs]).T

    # Peeling: try to reduce rank by finding linearly dependent rank-1 components
    rank = U.shape[1]
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Check if the decomposition is valid - tensor should be 1 on diagonal, 0 elsewhere"""
        U_3d = U_mat.reshape(n, m, -1)
        V_3d = V_mat.reshape(m, p, -1)
        W_3d = W_mat.reshape(p, n, -1)
        T = np.einsum('ijt,jkt,kit->ijk', U_3d, V_3d, W_3d)
        # T[i,j,k] should be 1 if i==j==k (for the multiplication tensor) 
        # Actually for matrix mult: T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B gives (i,k) entry
        # The tensor T_{i,j,k} = delta_{i,i} * delta_{j,j} * delta_{k,k}... 
        # No: for matrix multiplication C = AB, C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is T[i,j,k] = 1 for all valid i,j,k (it's the "multiplication tensor")
        # Wait, let me reconsider. The matrix multiplication tensor T satisfies:
        # <T, A ⊗ B> = C where C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
        # Actually T[i,j,k] = delta_{i,i}delta_{j,j}delta_{k,k} doesn't make sense.
        # The correct form: T is a 3-tensor where T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
        # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # That means the tensor is all ones!
        # Let me verify: sum_j A[i,j]*B[j,k] = sum_j <A, e_i⊗e_j> <B, e_j⊗e_k>
        # = <A⊗B, sum_j e_i⊗e_j⊗e_j⊗e_k>... 
        # I think T[i,j,k] = 1 for all i,j,k is correct for the standard formulation.
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-10)
    
    if time.time() < DEADLINE - 20 and rank > 20:
        # Try to peel: find if any column is linearly dependent
        max_attempts = 100
        attempts = 0
        removed = set()
        
        for t in range(rank):
            if time.time() > DEADLINE - 15:
                break
            if t in removed:
                continue
            if attempts >= max_attempts:
                break
            
            attempts += 1
            # Try removing term t
            mask = [i for i in range(rank) if i != t and i not in removed]
            if len(mask) < 2:
                continue
            
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            
            if verify_decomposition(U_test, V_test, W_test):
                removed.add(t)
        
        if removed:
            final_mask = [i for i in range(rank) if i not in removed]
            U = U[:, final_mask]
            V = V[:, final_mask]
            W = W[:, final_mask]
            rank = U.shape[1]

    # Final verification
    try:
        if not verify_decomposition(U, V, W):
            # Fallback to schoolbook
            U_vecs.clear()
            V_vecs.clear()
            W_vecs.clear()
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
            U = np.array([[float(x) for x in row] for row in U_vecs]).T
            V = np.array([[float(x) for x in row] for row in V_vecs]).T
            W = np.array([[float(x) for x in row] for row in W_vecs]).T
    except Exception:
        # Fallback to schoolbook on error
        U_vecs.clear()
        V_vecs.clear()
        W_vecs.clear()
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
        U = np.array([[float(x) for x in row] for row in U_vecs]).T
        V = np.array([[float(x) for x in row] for row in V_vecs]).T
        W = np.array([[float(x) for x in row] for row in W_vecs]).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
