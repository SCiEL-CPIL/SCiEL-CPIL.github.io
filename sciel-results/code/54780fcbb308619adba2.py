"""
5x5 matrix multiplication - rank minimization via tensor compression.
Uses ALS on the matrix multiplication tensor to search for low-rank decompositions,
with a block-structured fallback (rank 117).
"""
import time
import numpy as np
from numpy.linalg import lstsq, norm

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 110.0  # Safety margin: evaluator kills at 150s

    def build_block_decomposition():
        """Build the rank-117 block decomposition (2+3 split with Strassen)."""
        U_list = []
        V_list = []
        W_list = []

        def add_term(u_vals, v_vals, w_vals):
            U = np.zeros((n*m, 1), dtype=float)
            V = np.zeros((m*p, 1), dtype=float)
            W = np.zeros((p*n, 1), dtype=float)

            for (i, j), val in u_vals.items():
                U[i*m+j, 0] = val
            for (j, k), val in v_vals.items():
                V[j*p+k, 0] = val
            for (k, i), val in w_vals.items():
                W[k*n+i, 0] = val

            U_list.append(U.flatten())
            V_list.append(V.flatten())
            W_list.append(W.flatten())

        def strassen_2x2(i0, i1, j0, j1, k0, k1):
            add_term({(i0,j0):1, (i1,j1):1}, {(j0,k0):1, (j1,k1):1}, {(k0,i0):1, (k1,i1):1})
            add_term({(i1,j0):1, (i1,j1):1}, {(j0,k0):1}, {(k0,i1):1, (k1,i1):-1})
            add_term({(i0,j0):1}, {(j0,k1):1, (j1,k1):-1}, {(k1,i0):1, (k1,i1):1})
            add_term({(i1,j1):1}, {(j1,k0):1, (j0,k0):-1}, {(k0,i0):1, (k0,i1):1})
            add_term({(i0,j0):1, (i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1, (k1,i0):1})
            add_term({(i1,j0):1, (i0,j0):-1}, {(j0,k0):1, (j0,k1):1}, {(k1,i1):1})
            add_term({(i0,j1):1, (i1,j1):-1}, {(j1,k0):1, (j1,k1):1}, {(k0,i0):1})

        # C_22 = A_22 B_22 + A_23 B_32
        strassen_2x2(0, 1, 0, 1, 0, 1)
        strassen_2x2(0, 1, 2, 3, 0, 1)
        for i in [0, 1]:
            for k in [0, 1]:
                add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

        # C_23 = A_22 B_23 + A_23 B_33
        strassen_2x2(0, 1, 0, 1, 2, 3)
        for i in [0, 1]:
            for j in [0, 1]:
                add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
        strassen_2x2(0, 1, 2, 3, 2, 3)
        for i in [0, 1]:
            for j in [2, 3]:
                add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
        for i in [0, 1]:
            for k in [2, 3, 4]:
                add_term({(i,4):1}, {(4,k):1}, {(k,i):1})

        # C_32 = A_32 B_22 + A_33 B_32
        strassen_2x2(2, 3, 0, 1, 0, 1)
        for j in [0, 1]:
            for k in [0, 1]:
                add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
        strassen_2x2(2, 3, 2, 3, 0, 1)
        for i in [2, 3]:
            for k in [0, 1]:
                add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
        for j in [2, 3]:
            for k in [0, 1]:
                add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
        for k in [0, 1]:
            add_term({(4,4):1}, {(4,k):1}, {(k,4):1})

        # C_33 = A_32 B_23 + A_33 B_33
        strassen_2x2(2, 3, 0, 1, 2, 3)
        for i in [2, 3]:
            for j in [0, 1]:
                add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
        for j in [0, 1]:
            for k in [2, 3]:
                add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
        for j in [0, 1]:
            add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
        strassen_2x2(2, 3, 2, 3, 2, 3)
        for i in [2, 3]:
            for k in [2, 3]:
                add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
        for i in [2, 3]:
            for j in [2, 3]:
                add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
        for i in [2, 3]:
            add_term({(i,4):1}, {(4,4):1}, {(4,i):1})
        for j in [2, 3]:
            for k in [2, 3]:
                add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
        for k in [2, 3]:
            add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
        for j in [2, 3]:
            add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
        add_term({(4,4):1}, {(4,4):1}, {(4,4):1})

        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        return U, V, W

    def verify_decomposition(U, V, W, seed=42, num_tests=3):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rng = np.random.RandomState(seed)
        for _ in range(num_tests):
            A_test = rng.randint(-3, 4, size=(n, m)).astype(float)
            B_test = rng.randint(-3, 4, size=(m, p)).astype(float)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n, p), dtype=float)
            for r in range(U.shape[1]):
                U_r = U[:, r].reshape(n, m)
                V_r = V[:, r].reshape(m, p)
                W_r = W[:, r].reshape(p, n)
                C_computed += U_r @ V_r @ W_r
            
            if not np.allclose(C_computed, C_expected, atol=1e-8):
                return False
        return True

    def als_compression(U0, V0, W0, target_rank, max_iter=50, tol=1e-10):
        """
        Alternating Least Squares to compress a rank-r0 decomposition to rank-r.
        Projects the tensor onto the variety of rank-r decompositions.
        Returns (U, V, W) of shape (nm, r), (mp, r), (pn, r) or None if fails.
        """
        r0 = U0.shape[1]
        r = target_rank
        
        # Initialize with random projection of the original factors
        rng = np.random.RandomState(42)
        P = rng.randn(r0, r)
        Q = rng.randn(r0, r)
        R = rng.randn(r0, r)
        
        U = U0 @ P
        V = V0 @ Q
        W = W0 @ R
        
        # Build the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For ALS, we fix two factors and solve for the third.
        # T_{ijk} = sum_r U_{ij}^r V_{jk}^r W_{ki}^r
        # Fix V, W: for each (i,j), solve U_{ij}^r from T_{ijk} = sum_r U_{ij}^r (V_{jk}^r W_{ki}^r)
        # This is: for each (i,j), U_{ij,:} @ diag(V_{jk,:} * W_{ki,:}) ... no
        # Actually: T_{ijk} = sum_r U_{ij}^r * V_{jk}^r * W_{ki}^r
        # For fixed V,W: T_{ijk} = sum_r U_{ij}^r * (V_{jk}^r * W_{ki}^r)
        # So for each (i,j): T_{ij,:} = U_{ij,:} * elementwise(V_{:j,:} * W_{:i,:})... 
        # No, it's: for each (i,j), the vector T_{ij,k} for all k equals sum_r U_{ij}^r * (V_{jk}^r * W_{ki}^r)
        # This is a linear system in U_{ij}^r.
        
        # Better approach: use the tensor structure directly.
        # T is a 3-tensor with indices (i,j,k). We want T_{ijk} = sum_r U_{ij}^r V_{jk}^r W_{ki}^r.
        # This is NOT a standard CP decomposition because the index pairing is cyclic.
        # Let's reformulate: define U'_{ij} = U_{ij}, V'_{jk} = V_{jk}, W'_{ki} = W_{ki}.
        # Then T_{ijk} = sum_r U'_{ij} V'_{jk} W'_{ki} -- this is still cyclic.
        
        # Alternative: use the fact that T_{ijk} = sum_r U_{ij}^r V_{jk}^r W_{ki}^r
        # For ALS on U: fix V, W. For each (i,j), we need U_{ij}^r such that
        # for all k: T_{ijk} = sum_r U_{ij}^r * V_{jk}^r * W_{ki}^r
        # Let M_{jk}^{(i)} = V_{jk}^r * W_{ki}^r (vector over r for each k)
        # Then T_{ij,:} = U_{ij,:} * M (where M is r x p matrix with M[r,k] = V_{jk}^r * W_{ki}^r)
        # So U_{ij,:} = T_{ij,:} * M^{-1} (if M is square and invertible)
        # But M is r x p, and p=5, r might be < 5. Use least squares.
        
        # Actually, let's think differently. The constraint is:
        # For all (i,j,k): T_{ijk} = sum_r U_{ij}^r V_{jk}^r W_{ki}^r
        # This is a system of n*m*p = 125 equations in r*(n*m + m*p + p*n) = r*75 unknowns.
        # For r >= ceil(125/75) = 2, it's potentially solvable.
        
        # ALS approach:
        # 1. Fix V, W. For each (i,j), solve for U_{ij,:} in the linear system:
        #    For each k: sum_r U_{ij}^r * (V_{jk}^r * W_{ki}^r) = T_{ijk}
        #    This is: U_{ij,:} @ A_{(i,j)} = T_{ij,:} where A_{(i,j)}[r,k] = V_{jk}^r * W_{ki}^r
        #    U_{ij,:} = lstsq(A_{(i,j)}, T_{ij,:})
        
        # 2. Fix U, W. For each (j,k), solve for V_{jk,:} similarly.
        #    For each i: sum_r V_{jk}^r * (U_{ij}^r * W_{ki}^r) = T_{ijk}
        #    V_{jk,:} @ B_{(j,k)} = T_{:jk} where B_{(j,k)}[r,i] = U_{ij}^r * W_{ki}^r
        
        # 3. Fix U, V. For each (k,i), solve for W_{ki,:} similarly.
        #    For each j: sum_r W_{ki}^r * (U_{ij}^r * V_{jk}^r) = T_{ijk}
        #    W_{ki,:} @ C_{(k,i)} = T_{:ki} where C_{(k,i)}[r,j] = U_{ij}^r * V_{jk}^r
        
        # First, compute the full tensor T from the current decomposition
        T = np.zeros((n, m, p), dtype=float)
        for r_idx in range(r0):
            U_r = U0[:, r_idx].reshape(n, m)
            V_r = V0[:, r_idx].reshape(m, p)
            W_r = W0[:, r_idx].reshape(p, n)
            # T_{ijk} += U_{ij}^r * V_{jk}^r * W_{ki}^r
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U_r[i,j] * V_r[j,k] * W_r[k,i]
        
        # Now run ALS
        best_residual = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if time.time() - t0 > deadline - 20:
                break
            
            # Update U: fix V, W
            U_new = np.zeros((n*m, r), dtype=float)
            for i in range(n):
                for j in range(m):
                    # A[r,k] = V_{jk}^r * W_{ki}^r
                    A = np.zeros((r, p), dtype=float)
                    for rr in range(r):
                        for k in range(p):
                            A[rr, k] = V[j*p+k, rr] * W[k*n+i, rr]
                    b = T[i, j, :]  # shape (p,)
                    # Solve U_{ij,:} @ A = b, i.e., A^T @ U_{ij,:}^T = b
                    sol, _, _, _ = lstsq(A.T, b, rcond=None)
                    U_new[i*m+j, :] = sol
            
            # Update V: fix U, W
            V_new = np.zeros((m*p, r), dtype=float)
            for j in range(m):
                for k in range(p):
                    B = np.zeros((r, n), dtype=float)
                    for rr in range(r):
                        for i in range(n):
                            B[rr, i] = U[i*m+j, rr] * W[k*n+i, rr]
                    b = T[:, j, k]  # shape (n,)
                    sol, _, _, _ = lstsq(B.T, b, rcond=None)
                    V_new[j*p+k, :] = sol
            
            # Update W: fix U, V
            W_new = np.zeros((p*n, r), dtype=float)
            for k in range(p):
                for i in range(n):
                    C = np.zeros((r, m), dtype=float)
                    for rr in range(r):
                        for j in range(m):
                            C[rr, j] = U[i*m+j, rr] * V[j*p+k, rr]
                    b = T[i, :, k]  # shape (m,)
                    sol, _, _, _ = lstsq(C.T, b, rcond=None)
                    W_new[k*n+i, :] = sol
            
            # Compute residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for rr in range(r):
                            val += U_new[i*m+j, rr] * V_new[j*p+k, rr] * W_new[k*n+i, rr]
                        residual += (val - T[i,j,k])**2
            
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            
            U, V, W = U_new, V_new, W_new
            
            if residual < tol:
                break
        
        # Verify the best solution
        if best_residual < 1e-6:
            # Final verification
            T_check = np.zeros((n, m, p), dtype=float)
            for rr in range(r):
                U_r = best_U[:, rr].reshape(n, m)
                V_r = best_V[:, rr].reshape(m, p)
                W_r = best_W[:, rr].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_check[i,j,k] += U_r[i,j] * V_r[j,k] * W_r[k,i]
            
            if np.allclose(T_check, T, atol=1e-8):
                return best_U, best_V, best_W
        
        return None

    # Build the base decomposition
    U0, V0, W0 = build_block_decomposition()
    base_rank = U0.shape[1]
    
    # Verify base decomposition
    if not verify_decomposition(U0, V0, W0):
        # Should not happen, but return as-is
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = base_rank
    
    # Try to compress to lower ranks using ALS
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) / something... 
    # For 5x5x5, known lower bound is 25 (trivial), but actual minimum is unknown.
    # Try ranks from base_rank-1 down to max(25, base_rank - 10)
    
    max_compression_attempts = 8
    for attempt in range(max_compression_attempts):
        target_rank = base_rank - 1 - attempt
        if target_rank < 25:
            break
        if time.time() - t0 > deadline - 30:
            break
        
        try:
            result = als_compression(U0, V0, W0, target_rank, max_iter=30, tol=1e-12)
            if result is not None:
                U_c, V_c, W_c = result
                if verify_decomposition(U_c, V_c, W_c, seed=123, num_tests=2):
                    best_result = {"U": U_c.tolist(), "V": V_c.tolist(), "W": W_c.tolist()}
                    best_rank = target_rank
                    # Try to go even lower from this point
                    continue
        except Exception:
            break
    
    return best_result
# EVOLVE-BLOCK-END
