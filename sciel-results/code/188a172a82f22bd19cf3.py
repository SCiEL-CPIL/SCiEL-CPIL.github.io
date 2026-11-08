import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the target tensor T[i, j, k] = delta[i,j,k] for matmul
    # T is of shape (n, m, p) where T[i, j, k] = 1 if i,j,k are valid indices?
    # Actually, the tensor for matrix multiplication M_{n,m,p} has entries:
    # T[i, j, k] = 1 if i < n, j < m, k < p ? No.
    # The multiplication (A @ B)[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T is defined by T[i, j, k] = 1 for all i, j, k in the support?
    # No, the tensor decomposition represents the bilinear form.
    # The tensor T has dimensions (nm, mp, pn) in the factorization context?
    # Let's stick to the definition:
    # U (nm, r), V (mp, r), W (pn, r)
    # T is the "matrix multiplication tensor".
    # T[i, j, k] is 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    # Wait, the standard tensor T for matmul has T_{i, j, k} = 1 if i, j, k are valid indices for A, B, C such that C_{i, k} depends on A_{i, j} and B_{j, k}.
    # Specifically, T is a 3-tensor of size n x m x p?
    # No, the indices are:
    # 1. Input A index: (i, j) -> i in [0, n), j in [0, m)
    # 2. Input B index: (j, k) -> j in [0, m), k in [0, p)
    # 3. Output C index: (i, k) -> i in [0, n), k in [0, p)
    # The tensor T has entries T[(i, j), (j, k), (i, k)] = 1.
    # Note that the middle index j must match.
    
    # We want to find U, V, W such that:
    # sum_r U[(i, j), r] * V[(j, k), r] * W[(i, k), r] == 1 if (i,j,k) is a valid triple, else 0?
    # No, the identity is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The decomposition says:
    # C[i, k] = sum_r (sum_j U[i, j, r] A[i, j]) (sum_j' V[j', k, r] B[j', k]) W[i, k, r]?
    # No, the factors are linear forms.
    # U_r is a linear form in A. V_r is a linear form in B. W_r is a linear form in C?
    # No, W_r is just a coefficient for the product of the two linear forms to contribute to C[i,k].
    # Actually, the decomposition is:
    # M_{n,m,p} = sum_r u_r \otimes v_r \otimes w_r
    # where u_r in A^*, v_r in B^*, w_r in C^*?
    # No, w_r is a vector in C.
    # So, T = sum_r u_r \otimes v_r \otimes w_r
    # T[i, j, k] = sum_r u_r[i, j] v_r[j, k] w_r[i, k]
    # We require T[i, j, k] = 1 if i,j,k form a valid "link" (i.e., the middle index j is shared), and 0 otherwise?
    # Let's check the indices.
    # A is n x m. B is m x p.
    # A element A[i, j]. B element B[j, k].
    # Product term A[i, j] B[j, k] contributes to C[i, k].
    # So the tensor T has a 1 at ( (i, j), (j, k), (i, k) )?
    # The indices for U are (i, j). Size n*m.
    # The indices for V are (j, k). Size m*p.
    # The indices for W are (i, k). Size n*p.
    # So T is a tensor of shape (n*m, m*p, n*p).
    # T[a, b, c] = 1 if a=(i,j), b=(j',k), c=(i',k') such that i=i', j=j', k=k'.
    # Otherwise 0.
    
    # Let's construct the target tensor T.
    dim_u = n * m
    dim_v = m * p
    dim_w = n * p
    
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = i * p + k
                T[idx_u, idx_v, idx_w] = 1.0

    def decompose_rank(r):
        # Initialize
        np.random.seed(42 + r)
        U = np.random.randn(dim_u, r)
        V = np.random.randn(dim_v, r)
        W = np.random.randn(dim_w, r)
        
        # Normalize to avoid scale drift
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)

        def compute_error(U, V, W):
            # Compute sum_r U[:, r] \otimes V[:, r] \otimes W[:, r]
            # This is expensive if done naively.
            # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            # return np.linalg.norm(T_recon - T)
            
            # Optimized check:
            # We want sum_r U[i,j,r] V[j,k,r] W[i,k,r] == T[i,j,k]
            # Let's flatten T.
            # Actually, for local search, we can just compute the difference on a subset or full.
            # Full computation is O(r * dim_u * dim_v * dim_w) which is too slow for large r?
            # dim_u=8, dim_v=16, dim_w=8. 8*16*8 = 1024 elements.
            # r=27. 27 * 1024 * 3 ~ 80k ops. Very fast.
            
            T_recon = np.zeros_like(T)
            for r_idx in range(r):
                T_recon += np.einsum('i,j,k->ijk', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            return np.linalg.norm(T_recon - T)

        def round_to_half_integer(M):
            return np.round(M * 2) / 2.0

        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        # Local Search
        # We use a simple coordinate descent or random perturbation.
        # Given the small size, we can try to minimize the error.
        # However, exact decomposition is hard.
        # Strategy:
        # 1. Run a few iterations of minimizing the error with real values.
        # 2. Round to half-integers.
        # 3. Check if error is 0.
        
        # Since we suspect rank 28 is minimal, we might not find 27.
        # But we try.
        
        # Use scipy minimize? It might be slow or get stuck.
        # Let's try a simple random restart and local refinement.
        
        for restart in range(5):
            if time.time() - start_time > 120:
                break
                
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
            
            # Random perturbations
            for step in range(100):
                if time.time() - start_time > 120:
                    break
                    
                # Perturb one matrix
                idx = np.random.randint(3)
                if idx == 0:
                    U_curr[np.random.randint(dim_u), np.random.randint(r)] += np.random.randn() * 0.1
                elif idx == 1:
                    V_curr[np.random.randint(dim_v), np.random.randint(r)] += np.random.randn() * 0.1
                else:
                    W_curr[np.random.randint(dim_w), np.random.randint(r)] += np.random.randn() * 0.1
                
                err = compute_error(U_curr, V_curr, W_curr)
                if err < best_err:
                    best_err = err
                    best_U = U_curr.copy()
                    best_V = V_curr.copy()
                    best_W = W_curr.copy()
                    
                    # Check for exactness after rounding
                    U_rounded = round_to_half_integer(best_U)
                    V_rounded = round_to_half_integer(best_V)
                    W_rounded = round_to_half_integer(best_W)
                    
                    # Check if rounded solution is exact
                    T_recon_rounded = np.zeros_like(T)
                    for r_idx in range(r):
                        T_recon_rounded += np.einsum('i,j,k->ijk', U_rounded[:, r_idx], V_rounded[:, r_idx], W_rounded[:, r_idx])
                    
                    if np.allclose(T_recon_rounded, T, atol=1e-9):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        
            # Reinitialize for next restart
            U = np.random.randn(dim_u, r)
            V = np.random.randn(dim_v, r)
            W = np.random.randn(dim_w, r)
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
        # If we found a good real solution but not exact integer, we can't return it.
        # We must return a valid integer solution.
        # Fallback to known rank 28 solution.
        return None

    # Try rank 27
    res_27 = decompose_rank(27)
    if res_27 is not None:
        return res_27

    # Fallback to Rank 28 (Strassen)
    # This part is the same as the seed, but we include it here as fallback.
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    def apply_strassen(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_off + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_off + r) * p + (b_col_off + c)
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (c_col_off + c) * n + r
                W[g_idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(2, 2, 0, 0, 7)
    apply_strassen(0, 0, 2, 2, 14)
    apply_strassen(2, 2, 2, 2, 21)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)

    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B

    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc

    if not np.allclose(C, Cf):
        print("Mismatch")
        print(C)
        print(Cf)
    else:
        print("OK rank", U.shape[1])
