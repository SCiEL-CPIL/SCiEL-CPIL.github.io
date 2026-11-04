"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    # Build target tensor T[i,j,k] = 1 if (i,j,k) is valid else 0
    # T has shape (n, m, p)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    T_flat = T.flatten()
    NM, MP, PN = n*m, m*p, p*n
    
    def check_exact(U, V, W, rank):
        """Check if U, V, W with given rank exactly reconstruct T"""
        U_r = np.array(U).reshape(n, m, rank)
        V_r = np.array(V).reshape(m, p, rank)
        W_r = np.array(W).reshape(p, n, rank)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-8)
    
    def schoolbook_init():
        """Create the schoolbook (trivial) decomposition"""
        U = np.zeros((n, m, rank_schoolbook))
        V = np.zeros((m, p, rank_schoolbook))
        W = np.zeros((p, n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, col] = 1
                    V[j, k, col] = 1
                    W[k, i, col] = 1
                    col += 1
        return U, V, W
    
    def try_merge(U, V, W, rank, num_trials=20):
        """
        Try to reduce rank by merging pairs of terms.
        For each pair of columns (a,b), try to find a linear combination
        that preserves the tensor.
        """
        if rank <= 1:
            return None
        
        # Try random linear combinations of pairs
        np.random.seed(int(time.time() * 1000) % 10000)
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Pick two random columns to potentially merge
            a, b = np.random.choice(rank, size=2, replace=False)
            
            # Try several random coefficients
            found = False
            for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    # Create new column as alpha*col_a + beta*col_b
                    U_new = U[:, :, a].copy() * alpha + U[:, :, b].copy() * beta
                    V_new = V[:, :, a].copy() * alpha + V[:, :, b].copy() * beta
                    W_new = W[:, :, a].copy() * alpha + W[:, :, b].copy() * beta
                    
                    # Check if this single new column can replace both old columns
                    # Remove columns a and b, add new column
                    new_rank = rank - 1
                    if new_rank < 1:
                        continue
                    
                    # Build new tensors
                    U_test = np.delete(U, [a, b], axis=2)
                    V_test = np.delete(V, [a, b], axis=2)
                    W_test = np.delete(W, [a, b], axis=2)
                    
                    # Insert new column at position a
                    U_test = np.insert(U_test, a, U_new, axis=2)
                    V_test = np.insert(V_test, a, V_new, axis=2)
                    W_test = np.insert(W_test, a, W_new, axis=2)
                    
                    # Check
                    T_rec = np.einsum('ijc,jkc,kic->ijk', U_test, V_test, W_test)
                    if np.allclose(T_rec.flatten(), T_flat, atol=1e-6):
                        return U_test, V_test, W_test, new_rank
            
            # Also try: replace one column with a combination
            for a in range(rank):
                for b in range(rank):
                    if a == b:
                        continue
                    for alpha in [0.5, 1.0, -0.5, -1.0]:
                        U_new = U[:, :, a].copy() * alpha + U[:, :, b].copy() * 1.0
                        V_new = V[:, :, a].copy() * alpha + V[:, :, b].copy() * 1.0
                        W_new = W[:, :, a].copy() * alpha + W[:, :, b].copy() * 1.0
                        
                        U_test = U.copy()
                        V_test = V.copy()
                        W_test = W.copy()
                        U_test[:, :, a] = U_new
                        V_test[:, :, a] = V_new
                        W_test[:, :, a] = W_new
                        
                        T_rec = np.einsum('ijc,jkc,kic->ijk', U_test, V_test, W_test)
                        if np.allclose(T_rec.flatten(), T_flat, atol=1e-6):
                            # We can remove column b
                            new_rank = rank - 1
                            if new_rank < 1:
                                continue
                            U_test = np.delete(U_test, b, axis=2)
                            V_test = np.delete(V_test, b, axis=2)
                            W_test = np.delete(W_test, b, axis=2)
                            return U_test, V_test, W_test, new_rank
    
        return None
    
    def als_decompose(rank, max_iters=50, tol=1e-12, seed=0, init=None):
        """Alternating Least Squares for 3rd order tensor decomposition"""
        np.random.seed(seed)
        
        if init is not None:
            U, V, W = init
        else:
            # Initialize with small random values
            U = np.random.randn(n, m, rank) * 0.3
            V = np.random.randn(m, p, rank) * 0.3
            W = np.random.randn(p, n, rank) * 0.3
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Compute current error
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = T_rec - T
            current_err = np.sum(err**2)
            
            if abs(prev_err - current_err) < tol:
                break
            prev_err = current_err
            
            # Solve for U: fix V, W
            for i in range(n):
                for j in range(m):
                    # A[c, k] = V[j,k,c] * W[k,i,c]
                    A = np.einsum('kc,kc->ck', V[j], W[:, i])  # shape (rank, p)
                    b = T[i, j, :]  # shape (p,)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V: fix U, W
            for j in range(m):
                for k in range(p):
                    A = np.einsum('ic,ic->ci', U[:, j, :], W[k, :, :])  # shape (rank, n)
                    b = T[:, j, k]  # shape (n,)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j, k, :] = v_sol
                    except:
                        pass
            
            # Solve for W: fix U, V
            for k in range(p):
                for i in range(n):
                    A = np.einsum('jc,jc->cj', U[i, :, :], V[:, k, :])  # shape (rank, m)
                    b = T[i, :, k]  # shape (m,)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k, i, :] = w_sol
                    except:
                        pass
        
        # Final check
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        final_err = np.sum((T_rec - T)**2)
        
        if final_err < 1e-4:
            # Try rounding to various grids
            for scale in [2, 4, 8]:
                U_rounded = np.round(U * scale) / scale
                V_rounded = np.round(V * scale) / scale
                W_rounded = np.round(W * scale) / scale
                if check_exact(U_rounded, V_rounded, W_rounded, rank):
                    return (U_rounded.reshape(NM, rank).tolist(), 
                           V_rounded.reshape(MP, rank).tolist(), 
                           W_rounded.reshape(PN, rank).tolist())
            
            # Try integer rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if check_exact(U_int, V_int, W_int, rank):
                return (U_int.reshape(NM, rank).tolist(), 
                       V_int.reshape(MP, rank).tolist(), 
                       W_int.reshape(PN, rank).tolist())
        
        return None
    
    def try_rank_als(rank, num_restarts=15):
        """Try to find decomposition of given rank using ALS"""
        for seed in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            result = als_decompose(rank, max_iters=40, tol=1e-12, seed=seed)
            if result is not None:
                return result
        
        # Try with schoolbook-inspired initialization
        U_sb, V_sb, W_sb = schoolbook_init()
        for seed in range(5):
            if time.time() > deadline - 10:
                return None
            np.random.seed(seed + 1000)
            # Randomly select a subset of schoolbook columns
            selected = np.random.choice(rank_schoolbook, size=min(rank, rank_schoolbook), replace=False)
            U_init = U_sb[:, :, selected]
            V_init = V_sb[:, :, selected]
            W_init = W_sb[:, :, selected]
            
            result = als_decompose(rank, max_iters=40, tol=1e-12, seed=seed, init=(U_init, V_init, W_init))
            if result is not None:
                return result
        
        return None
    
    def try_rank_merge(start_rank, target_rank, max_merges=200):
        """
        Start from schoolbook and try to reduce rank via merging.
        """
        U, V, W = schoolbook_init()
        current_rank = start_rank
        
        # First, try to reduce to target_rank by removing redundant columns
        # Then try merging
        
        merges_done = 0
        while current_rank > target_rank and merges_done < max_merges:
            if time.time() > deadline - 10:
                break
            
            result = try_merge(U, V, W, current_rank, num_trials=10)
            if result is not None:
                U, V, W, current_rank = result
                merges_done += 1
            else:
                # Try ALS to clean up
                result = als_decompose(current_rank, max_iters=20, tol=1e-12, seed=merges_done, init=(U, V, W))
                if result is not None:
                    U = np.array(result[0]).reshape(n, m, current_rank)
                    V = np.array(result[1]).reshape(m, p, current_rank)
                    W = np.array(result[2]).reshape(p, n, current_rank)
                else:
                    break
        
        if current_rank <= target_rank:
            # Round to clean values
            for scale in [2, 4, 8]:
                U_rounded = np.round(U * scale) / scale
                V_rounded = np.round(V * scale) / scale
                W_rounded = np.round(W * scale) / scale
                if check_exact(U_rounded, V_rounded, W_rounded, current_rank):
                    return (U_rounded.reshape(NM, current_rank).tolist(), 
                           V_rounded.reshape(MP, current_rank).tolist(), 
                           W_rounded.reshape(PN, current_rank).tolist())
            
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if check_exact(U_int, V_int, W_int, current_rank):
                return (U_int.reshape(NM, current_rank).tolist(), 
                       V_int.reshape(MP, current_rank).tolist(), 
                       W_int.reshape(PN, current_rank).tolist())
        
        return None
    
    def strassen_2x2():
        """
        Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
        The tensor M_{2,2,2} has entries T[i,j,k] = 1 if i,j,k are valid indices.
        Returns U (4,7), V (4,7), W (4,7) where:
          U is indexed by (i,j), V by (j,k), W by (k,i)
        """
        # Strassen's 7 terms for 2x2 multiplication
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = A*B where C[0,0]=ae+bg, C[0,1]=af+bh, C[1,0]=ce+dg, C[1,1]=cf+dh
        
        # Strassen formulas:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-d)(f+h)
        # m7 = (b-c)(e+g)
        
        # C[0,0] = m1 + m4 - m5 + m7
        # C[0,1] = m3 + m5
        # C[1,0] = m2 + m4
        # C[1,1] = m1 - m2 + m3 + m6
        
        # U[i,j,c] = coefficient of a_{i,j} in term c
        # V[j,k,c] = coefficient of b_{j,k} in term c
        # W[k,i,c] = coefficient of c_{k,i} in term c (output)
        
        U = np.zeros((2, 2, 7))
        V = np.zeros((2, 2, 7))
        W = np.zeros((2, 2, 7))
        
        # Term 0: m1 = (a+d)(e+h) = ae + ah + de + dh
        # a_{0,0}=a, a_{0,1}=b, a_{1,0}=c, a_{1,1}=d
        # b_{0,0}=e, b_{0,1}=f, b_{1,0}=g, b_{1,1}=h
        U[0, 0, 0] = 1  # a
        U[1, 1, 0] = 1  # d
        V[0, 0, 0] = 1  # e
        V[1, 1, 0] = 1  # h
        # Output: C[0,0] += 1, C[0,1] -= 1, C[1,0] -= 1, C[1,1] += 1
        W[0, 0, 0] = 1   # C[0,0]
        W[1, 0, 0] = -1  # C[0,1] -> W[k,i] where k=1,i=0
        W[0, 1, 0] = -1  # C[1,0] -> W[k,i] where k=0,i=1
        W[1, 1, 0] = 1   # C[1,1]
        
        # Term 1: m2 = (c+d)e = ce + de
        U[1, 0, 1] = 1  # c
        U[1, 1, 1] = 1  # d
        V[0, 0, 1] = 1  # e
        # Output: C[1,0] += 1, C[1,1] -= 1
        W[0, 1, 1] = 1   # C[1,0]
        W[1, 1, 1] = -1  # C[1,1]
        
        # Term 2: m3 = a(f-h) = af - ah
        U[0, 0, 2] = 1  # a
        V[0, 1, 2] = 1  # f
        V[1, 1, 2] = -1 # h
        # Output: C[0,1] += 1, C[1,1] += 1
        W[1, 0, 2] = 1   # C[0,1]
        W[1, 1, 2] = 1   # C[1,1]
        
        # Term 3: m4 = d(g-e) = dg - de
        U[1, 1, 3] = 1  # d
        V[1, 0, 3] = 1  # g
        V[0, 0, 3] = -1 # e
        # Output: C[0,0] += 1, C[1,0] += 1
        W[0, 0, 3] = 1   # C[0,0]
        W[0, 1, 3] = 1   # C[1,0]
        
        # Term 4: m5 = (a+b)h = ah + bh
        U[0, 0, 4] = 1  # a
        U[0, 1, 4] = 1  # b
        V[1, 1, 4] = 1  # h
        # Output: C[0,0] -= 1, C[0,1] += 1
        W[0, 0, 4] = -1  # C[0,0]
        W[1, 0, 4] = 1   # C[0,1]
        
        # Term 5: m6 = (c-d)(f+h) = cf + ch - df - dh
        U[1, 0, 5] = 1  # c
        U[1, 1, 5] = -1 # d
        V[0, 1, 5] = 1  # f
        V[1, 1, 5] = 1  # h
        # Output: C[1,1] += 1
        W[1, 1, 5] = 1   # C[1,1]
        
        # Term 6: m7 = (b-c)(e+g) = be + bg - ce - cg
        U[0, 1, 6] = 1  # b
        U[1, 0, 6] = -1 # c
        V[0, 0, 6] = 1  # e
        V[1, 0, 6] = 1  # g
        # Output: C[0,0] += 1
        W[0, 0, 6] = 1   # C[0,0]
        
        return U, V, W
    
    def embed_strassen_to_3x4x6():
        """
        Embed the 2x2 Strassen decomposition into a 3x4x6 decomposition.
        
        Strategy: Partition the 3x4 matrix A into blocks and the 4x6 matrix B into blocks,
        then use Strassen-like combinations.
        
        For 3x4 * 4x6:
        We can think of this as multiplying (3x4) by (4x6).
        
        One approach: Pad to 4x4 * 4x8, use 2x2 block Strassen recursively.
        
        Let's use a direct embedding:
        Split A (3x4) into top 2 rows and bottom 1 row.
        Split B (4x6) into left 2 cols and right 4 cols... this gets complicated.
        
        Alternative: Use the fact that rank(M_{n,m,p}) <= rank(M_{n',m',p'}) 
        if we can embed. But the simplest is to just try ALS with better initialization.
        
        For now, let's create a structured initialization based on Kronecker products.
        """
        # Use a simple Kronecker product structure
        # rank = 3*4*6 / 2 = 36 as a starting point (optimistic)
        
        # Actually, let's just use the Strassen 2x2 as a building block
        # and create a larger decomposition by tiling
        
        # For 3x4x6, we can use:
        # - Two 2x2 blocks in the row dimension (covering rows 0,1 and row 2 separately)
        # - Two 2x2 blocks in the col dimension (covering cols 0,1 and cols 2,3)
        # - Three 2x2 blocks in the output col dimension (covering cols 0,1 and 2,3 and 4,5)
        
        # This is getting complex. Let's use a simpler approach:
        # Create a rank-48 initialization by taking 2/3 of the schoolbook terms
        
        U_sb, V_sb, W_sb = schoolbook_init()
        
        # Select 48 terms from schoolbook in a structured way
        # Keep terms where (i+j+k) % 3 != 2 (removes 1/3 of terms)
        selected_cols = []
        for col in range(rank_schoolbook):
            i = col // (m * p)
            j = (col // p) % m
            k = col % p
            if (i + j + k) % 3 != 2:
                selected_cols.append(col)
        
        # This gives us 48 columns
        U_init = U_sb[:, :, selected_cols]
        V_init = V_sb[:, :, selected_cols]
        W_init = W_sb[:, :, selected_cols]
        
        return U_init, V_init, W_init, len(selected_cols)
    
    def polish_decomposition(U, V, W, rank, max_iters=100, tol=1e-14):
        """
        High-precision polish pass: use alternating least squares with tight tolerance
        to drive the error to machine precision.
        """
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    A = np.einsum('kc,kc->ck', V[j], W[:, i])
                    b = T[i, j, :]
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    A = np.einsum('ic,ic->ci', U[:, j, :], W[k, :, :])
                    b = T[:, j, k]
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j, k, :] = v_sol
                    except:
                        pass
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    A = np.einsum('jc,jc->cj', U[i, :, :], V[:, k, :])
                    b = T[i, :, k]
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k, i, :] = w_sol
                    except:
                        pass
            
            # Check convergence
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = np.sum((T_rec - T)**2)
            if err < tol:
                break
        
        # Final verification
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        final_err = np.sum((T_rec - T)**2)
        
        if final_err < 1e-6:
            # Try rounding
            for scale in [2, 4, 8, 16]:
                U_rounded = np.round(U * scale) / scale
                V_rounded = np.round(V * scale) / scale
                W_rounded = np.round(W * scale) / scale
                if check_exact(U_rounded, V_rounded, W_rounded, rank):
                    return (U_rounded.reshape(NM, rank).tolist(), 
                           V_rounded.reshape(MP, rank).tolist(), 
                           W_rounded.reshape(PN, rank).tolist())
            
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if check_exact(U_int, V_int, W_int, rank):
                return (U_int.reshape(NM, rank).tolist(), 
                       V_int.reshape(MP, rank).tolist(), 
                       W_int.reshape(PN, rank).tolist())
        
        return None
    
    # Strategy:
    # 1. Try structured initializations (Strassen-inspired, Kronecker)
    # 2. Use ALS with polish to find lower rank decompositions
    # 3. Fall back to schoolbook if nothing works
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Phase 1: Try structured initializations and polish
    # Try rank 48 with structured init
    U_init, V_init, W_init, init_rank = embed_strassen_to_3x4x6()
    
    for target_rank in [48, 54, 60, 66]:
        if time.time() > deadline - 20:
            break
        
        # Try to polish from the structured initialization
        if target_rank == init_rank:
            U_test = U_init.copy()
            V_test = V_init.copy()
            W_test = W_init.copy()
        else:
            # Truncate or pad to target_rank
            if target_rank < init_rank:
                U_test = U_init[:, :, :target_rank]
                V_test = V_init[:, :, :target_rank]
                W_test = W_init[:, :, :target_rank]
            else:
                U_test = np.zeros((n, m, target_rank))
                V_test = np.zeros((m, p, target_rank))
                W_test = np.zeros((p, n, target_rank))
                U_test[:, :, :init_rank] = U_init
                V_test[:, :, :init_rank] = V_init
                W_test[:, :, :init_rank] = W_init
        
        result = polish_decomposition(U_test, V_test, W_test, target_rank, max_iters=80)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            break
    
    # Phase 2: ALS search from reasonable ranks with multiple restarts
    if best_result is None:
        for rank in range(62, 47, -1):
            if time.time() > deadline - 20:
                break
            
            result = try_rank_als(rank, num_restarts=8)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
    
    # Phase 3: If Phase 1 and 2 failed, try merge approach
    if best_result is None:
        for target in [55, 50, 48]:
            if time.time() > deadline - 15:
                break
            
            result = try_rank_merge(rank_schoolbook, target, max_merges=100)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = len(U[0]) if U else 0
                break
    
    # Phase 4: If still no result, use schoolbook
    if best_result is None:
        U = [[0] * rank_schoolbook for _ in range(NM)]
        V = [[0] * rank_schoolbook for _ in range(MP)]
        W = [[0] * rank_schoolbook for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank_schoolbook
    
    return best_result
# EVOLVE-BLOCK-END
