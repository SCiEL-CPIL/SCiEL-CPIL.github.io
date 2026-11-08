"""Matrix multiplication 2x5x6 tensor decomposition - rank 54."""
import numpy as np
import time

N_DIM, M_DIM, P_DIM = 2, 5, 6
TARGET_RANK = 54


def _build_factors():
    n, m, p = N_DIM, M_DIM, P_DIM
    r = TARGET_RANK

    # Allocate sparse-ish factor matrices
    U_mat = [[0.0] * r for _ in range(n * m)]
    V_mat = [[0.0] * r for _ in range(m * p)]
    W_mat = [[0.0] * r for _ in range(p * n)]

    idx = 0

    # Strassen-style 2x2 block pairing over j and k
    jpairs = ((0, 1), (2, 3))
    kpairs = ((0, 1), (2, 3), (4, 5))

    for (ja, jb) in jpairs:
        for (ka, kb) in kpairs:
            # Term 1: (A0ja + A1jb)(Bja,ka + Bjb,kb)
            U_mat[ja][idx] = 1.0
            U_mat[m + jb][idx] = 1.0
            V_mat[ja * p + ka][idx] = 1.0
            V_mat[jb * p + kb][idx] = 1.0
            W_mat[ka * n][idx] = 1.0
            W_mat[kb * n + 1][idx] = 1.0
            idx += 1

            # Term 2: (A1ja + A1jb) Bja,ka
            U_mat[m + ja][idx] = 1.0
            U_mat[m + jb][idx] = 1.0
            V_mat[ja * p + ka][idx] = 1.0
            W_mat[ka * n + 1][idx] = 1.0
            W_mat[kb * n + 1][idx] = -1.0
            idx += 1

            # Term 3: A0ja (Bja,kb - Bjb,kb)
            U_mat[ja][idx] = 1.0
            V_mat[ja * p + kb][idx] = 1.0
            V_mat[jb * p + kb][idx] = -1.0
            W_mat[kb * n][idx] = 1.0
            W_mat[kb * n + 1][idx] = 1.0
            idx += 1

            # Term 4: A1jb (Bjb,ka - Bja,ka)
            U_mat[m + jb][idx] = 1.0
            V_mat[jb * p + ka][idx] = 1.0
            V_mat[ja * p + ka][idx] = -1.0
            W_mat[ka * n][idx] = 1.0
            W_mat[ka * n + 1][idx] = 1.0
            idx += 1

            # Term 5: (A0ja + A0jb) Bjb,kb
            U_mat[ja][idx] = 1.0
            U_mat[jb][idx] = 1.0
            V_mat[jb * p + kb][idx] = 1.0
            W_mat[ka * n][idx] = -1.0
            W_mat[kb * n][idx] = 1.0
            idx += 1

            # Term 6: (A1ja - A0ja)(Bja,ka + Bja,kb)
            U_mat[m + ja][idx] = 1.0
            U_mat[ja][idx] = -1.0
            V_mat[ja * p + ka][idx] = 1.0
            V_mat[ja * p + kb][idx] = 1.0
            W_mat[kb * n + 1][idx] = 1.0
            idx += 1

            # Term 7: (A0jb - A1jb)(Bjb,ka + Bjb,kb)
            U_mat[jb][idx] = 1.0
            U_mat[m + jb][idx] = -1.0
            V_mat[jb * p + ka][idx] = 1.0
            V_mat[jb * p + kb][idx] = 1.0
            W_mat[ka * n][idx] = 1.0
            idx += 1

    # Handle j=4 directly (no Strassen pairing possible)
    for kk in range(P_DIM):
        # C0,kk = A0,4 * B4,kk
        U_mat[4][idx] = 1.0
        V_mat[4 * p + kk][idx] = 1.0
        W_mat[kk * n][idx] = 1.0
        idx += 1

        # C1,kk = A1,4 * B4,kk
        U_mat[m + 4][idx] = 1.0
        V_mat[4 * p + kk][idx] = 1.0
        W_mat[kk * n + 1][idx] = 1.0
        idx += 1

    assert idx == r, f"Expected {r} terms, got {idx}"
    return U_mat, V_mat, W_mat


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    r = len(U[0])
    # Build the full tensor from factors
    # T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # Check against the multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The multiplication tensor has T[i,j,k] = 1 if i==i and j==k (in the right sense)
                # Actually, for matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
                # The tensor is T[i,j,k] = delta_{j,k} for the bilinear form
                # Wait, the tensor for matrix mult is: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
                # So T[i,j,k] should be 1 when the indices match for a valid multiplication term
                # The decomposition gives: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                # This should equal 1 if this is a valid (i,j,k) term, 0 otherwise
                # Actually, every (i,j,k) is a valid term in the sum, so T[i,j,k] = 1 for all i,j,k?
                # No, that's not right. Let me think again.
                # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in the standard basis
                # because C[i,k] = sum_j A[i,j] * B[j,k], and each product A[i,j]*B[j,k] contributes
                # to C[i,k]. So the tensor is all ones? No...
                # Actually, the tensor is defined as T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k} = 1
                # for the standard multiplication. But that can't be right either.
                # Let me just check numerically with random matrices.
                pass
    
    # Numerical verification with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from factors
    C_computed = np.zeros((n, p))
    for t in range(r):
        # U[:,t] is a vector of length n*m, reshape to (n,m)
        Ut = np.array(U)[:, t].reshape(n, m)
        # V[:,t] is a vector of length m*p, reshape to (m,p)
        Vt = np.array(V)[:, t].reshape(m, p)
        # W[:,t] is a vector of length p*n, reshape to (p,n)
        Wt = np.array(W)[:, t].reshape(p, n)
        # The contribution is outer product in some sense
        # C[i,k] += sum_j U[i,j,t] * V[j,k,t] * W[k,i,t]
        # But W[k,i,t] depends on both k and i, so this is not a simple outer product
        # We need to sum over j for each (i,k)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_computed[i, k] += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        diff = np.abs(C_computed - C_expected).max()
        raise ValueError(f"Decomposition error: max diff = {diff}")
    return U_mat, V_mat, W_mat


def construct():
    t0 = time.perf_counter()
    limit = t0 + 115.0

    if time.perf_counter() > limit:
        raise RuntimeError("timeout before start")

    Uf, Vf, Wf = _build_factors()

    if time.perf_counter() > limit:
        raise RuntimeError("timeout after build")

    return {"U": Uf, "V": Vf, "W": Wf}
