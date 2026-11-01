import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Build the matrix multiplication tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # First, construct the rank-28 Strassen decomposition as a baseline
    rank28 = 28
    U28 = np.zeros((n * m, rank28), dtype=int)
    V28 = np.zeros((m * p, rank28), dtype=int)
    W28 = np.zeros((p * n, rank28), dtype=int)

    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    def map_u_a(side, local_idx):
        if side == 'L':
            return [0, 1, 4, 5][local_idx]
        else:
            return [2, 3, 6, 7][local_idx]

    def map_v_b(block, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if block == 'TL':
            return r * 4 + c
        elif block == 'TR':
            return r * 4 + (2 + c)
        elif block == 'BL':
            return (2 + r) * 4 + c
        elif block == 'BR':
            return (2 + r) * 4 + (2 + c)

    def map_w_c(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if side == 'L':
            return c * 2 + r
        else:
            return (2 + c) * 2 + r

    col = 0
    # Apply Strassen to 4 block products
    block_configs = [
        ('L', 'TL', 'L'),  # A_L * B_TL -> C_L
        ('R', 'BL', 'L'),  # A_R * B_BL -> C_L
        ('L', 'TR', 'R'),  # A_L * B_TR -> C_R
        ('R', 'BR', 'R'),  # A_R * B_BR -> C_R
    ]

    for a_side, b_block, c_side in block_configs:
        for u_l, v_l, w_l in strassen_terms:
            for idx, coeff in u_l:
                U28[map_u_a(a_side, idx), col] += coeff
            for idx, coeff in v_l:
                V28[map_v_b(b_block, idx), col] += coeff
            for idx, coeff in w_l:
                W28[map_w_c(c_side, idx), col] += coeff
            col += 1

    # Verify rank-28 solution
    def verify_decomp(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - T[a, b, c]) > tol:
                        return False
        return True

    # Check if rank 28 works
    if not verify_decomp(U28, V28, W28):
        # Should not happen, but fall back to naive if something is wrong
        rank = 32
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to find a rank-27 decomposition
    # Strategy: Look for a linear combination of the 28 rank-1 terms that
    # preserves the tensor but has one fewer non-zero term.
    # 
    # Approach: Try to merge two terms by finding a sparse transformation.
    # If we can find vectors u', v', w' such that
    # u'v'w' = u1v1w1 + u2v2w2 (where u1,v1,w1 and u2,v2,w2 are two of the terms)
    # then we can replace those two terms with one.
    #
    # More generally, we look for a rank-27 decomposition by solving:
    # T = sum_{r=1}^{27} u_r v_r w_r
    #
    # This is a hard non-convex problem. Let's try a heuristic search.
    
    if time.time() < deadline - 10:
        # Try to find rank-27 via iterative perturbation
        # Start from rank-28 and try to zero out one column by adjusting others
        
        best_rank = 28
        best_U, best_V, best_W = U28.copy(), V28.copy(), W28.copy()
        
        # Try each pair of terms to see if they can be merged
        # For a given pair (i, j), we want to find new terms that replace them
        # with fewer terms while preserving the tensor.
        
        # Alternative approach: use the fact that some entries in U, V, W are zero
        # and look for sparse combinations.
        
        # Let's try a different approach: enumerate possible rank-27 decompositions
        # by trying to set one entry in each of U, V, W to zero and solving.
        
        # Actually, let's try a numerical optimization approach.
        # Parameterize a rank-27 decomposition and minimize ||T - sum u_r v_r w_r||^2
        
        if time.time() < deadline - 20:
            # Try to find a rank-27 decomposition via optimization
            # We'll use a simple gradient-free approach since the problem is small.
            
            np.random.seed(42)
            
            # Initialize with the first 27 columns of the rank-28 solution
            U_init = U28[:, :27].astype(float)
            V_init = V28[:, :27].astype(float)
            W_init = W28[:, :27].astype(float)
            
            # Flatten the tensor for comparison
            T_flat = T.flatten()
            
            def decomp_error(U, V, W):
                """Compute ||T - sum_r U[:,r] V[:,r] W[:,r]||^2"""
                rank = U.shape[1]
                T_decomp = np.zeros_like(T)
                for r in range(rank):
                    T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                return np.sum((T_flat - T_decomp.flatten())**2)
            
            # Try a few random initializations and local searches
            found_rank27 = False
            
            for attempt in range(5):
                if time.time() > deadline - 15:
                    break
                    
                if attempt == 0:
                    U_cur = U_init.copy()
                    V_cur = V_init.copy()
                    W_cur = W_init.copy()
                else:
                    # Random perturbation of the rank-27 initialization
                    U_cur = U28[:, :27].astype(float) + np.random.randn(n * m, 27) * 0.1
                    V_cur = V28[:, :27].astype(float) + np.random.randn(m * p, 27) * 0.1
                    W_cur = W28[:, :27].astype(float) + np.random.randn(p * n, 27) * 0.1
                
                lr = 0.01
                # Simple coordinate descent
                for iter in range(200):
                    if time.time() > deadline - 15:
                        break
                    
                    # Compute current decomposition once
                    T_decomp = np.zeros_like(T)
                    for s in range(27):
                        T_decomp += np.einsum('a,b,c->abc', U_cur[:, s], V_cur[:, s], W_cur[:, s])
                    residual = T - T_decomp
                    err = np.sum(residual**2)
                    
                    if err < 1e-10:
                        found_rank27 = True
                        best_U, best_V, best_W = U_cur, V_cur, W_cur
                        best_rank = 27
                        break

                    # Update U
                    for r in range(27):
                        # Gradient for U[:, r]
                        grad_U = np.zeros(n * m)
                        for a in range(n * m):
                            grad_U[a] = -2 * np.sum(residual[a, :, :] * V_cur[:, r, None] * W_cur[:, r, None].T)
                        U_cur[:, r] -= lr * grad_U
                    
                    # Update V
                    for r in range(27):
                        grad_V = np.zeros(m * p)
                        for b in range(m * p):
                            grad_V[b] = -2 * np.sum(residual[:, b, :] * U_cur[:, r, None] * W_cur[:, r, None].T)
                        V_cur[:, r] -= lr * grad_V

                    # Update W
                    for r in range(27):
                        grad_W = np.zeros(p * n)
                        for c in range(p * n):
                            grad_W[c] = -2 * np.sum(residual[:, :, c] * U_cur[:, r, None] * V_cur[:, r, None].T)
                        W_cur[:, r] -= lr * grad_W
                    if err < 1e-10:
                        found_rank27 = True
                        best_U, best_V, best_W = U_cur, V_cur, W_cur
                        best_rank = 27
                        break
                    
                    if time.time() > deadline - 15:
                        break
                
                if found_rank27:
                    break
            
            # Check if we found a valid rank-27 solution
            if found_rank27 and verify_decomp(best_U, best_V, best_W):
                # Round to nearest integers or half-integers
                U_out = np.round(best_U * 2) / 2
                V_out = np.round(best_V * 2) / 2
                W_out = np.round(best_W * 2) / 2
                
                # Verify after rounding
                if verify_decomp(U_out, V_out, W_out):
                    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
        
        # If no rank-27 found, try another approach: look for sparse linear combinations
        # of the 28 terms that can be merged.
        
        if time.time() < deadline - 5:
            # Try to find a 2x2 block of terms that can be merged into 1 term
            # This is equivalent to finding a rank-1 approximation of a 2-term sum.
            
            for i in range(28):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, 28):
                    if time.time() > deadline - 5:
                        break
                    
                    # Try to merge terms i and j
                    # We want to find u', v', w' such that
                    # u'v'w' = u_i v_i w_i + u_j v_j w_j
                    # This is possible if the 2-term sum has rank 1 as a tensor.
                    
                    # Compute the 2-term tensor
                    T_two = np.zeros_like(T)
                    T_two += np.einsum('a,b,c->abc', U28[:, i], V28[:, i], W28[:, i])
                    T_two += np.einsum('a,b,c->abc', U28[:, j], V28[:, j], W28[:, j])
                    
                    # Check if T_two has rank 1
                    # A tensor has rank 1 if it can be written as u v w
                    # We can check this by seeing if all 2x2 minors vanish
                    # For a 3D tensor, this means T[a1,b1,c1]*T[a2,b2,c2] = T[a1,b2,c1]*T[a2,b1,c2] etc.
                    
                    # Simpler check: if T_two is non-zero, pick a non-zero entry and
                    # see if the rest is consistent with rank 1.
                    
                    nonzero = np.argwhere(T_two != 0)
                    if len(nonzero) == 0:
                        continue
                    
                    a0, b0, c0 = nonzero[0]
                    val0 = T_two[a0, b0, c0]
                    if abs(val0) < 1e-10:
                        continue
                    
                    # Check rank 1 condition
                    is_rank1 = True
                    for a1 in range(n * m):
                        for b1 in range(m * p):
                            for c1 in range(p * n):
                                # T[a1,b1,c1] should equal T[a0,b0,c0] * 
                                # (u[a1]/u[a0]) * (v[b1]/v[b0]) * (w[c1]/w[c0])
                                # This is hard to check directly. Let's use a different approach.
                                pass
                    
                    # Skip this complex check for now
                    pass
    
    # Return the best solution found
    # If we found rank 27, return it; otherwise return rank 28
    if best_rank == 27:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
