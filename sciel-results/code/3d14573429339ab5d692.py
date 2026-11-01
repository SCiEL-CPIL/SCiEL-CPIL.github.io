import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140

    n, m, p = DIMENSIONS

    # Build the target tensor T[i,j,k] for 2x4x4
    # i = row_a * 4 + col_a (A is 2x4)
    # j = row_b * 4 + col_b (B is 4x4)
    # k = col_c * 2 + row_c (C is 2x4, stored as W[k*n+i,r] convention: k = col*n + row)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for row_a in range(n):
        for col_a in range(m):
            i = row_a * m + col_a
            for row_b in range(m):
                for col_b in range(p):
                    j = row_b * p + col_b
                    for row_c in range(n):
                        for col_c in range(p):
                            k = col_c * n + row_c
                            if col_a == row_b and row_a == row_c and col_b == col_c:
                                T[i, j, k] = 1.0

    # Strassen 2x2x2 factors
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ], dtype=float)

    V_s = np.array([
        [1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1]
    ], dtype=float)

    # Derive W_s by solving the linear system for 2x2x2
    T_2 = np.zeros((4, 4, 4), dtype=float)
    for ra in range(2):
        for ca in range(2):
            ii = ra * 2 + ca
            for rb in range(2):
                for cb in range(2):
                    jj = rb * 2 + cb
                    for rc in range(2):
                        for cc in range(2):
                            kk = cc * 2 + rc
                            if ca == rb and ra == rc and cb == cc:
                                T_2[ii, jj, kk] = 1.0

    M_sys = np.zeros((64, 28), dtype=float)
    T_vec = np.zeros(64, dtype=float)
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                eq = ii * 16 + jj * 4 + kk
                T_vec[eq] = T_2[ii, jj, kk]
                for s in range(7):
                    M_sys[eq, kk * 7 + s] = U_s[ii, s] * V_s[jj, s]

    W_vec, _, _, _ = np.linalg.lstsq(M_sys, T_vec, rcond=None)
    W_s = W_vec.reshape(4, 7)

    # Verify Strassen decomposition
    err_s = 0.0
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                val = np.sum(U_s[ii, :] * V_s[jj, :] * W_s[kk, :])
                err_s = max(err_s, abs(val - T_2[ii, jj, kk]))

    if err_s > 1e-9:
        # Fallback: schoolbook rank 32
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Build the rank-28 Strassen decomposition for 2x4x4
    def build_rank28():
        U_full = np.zeros((8, 28), dtype=float)
        V_full = np.zeros((16, 28), dtype=float)
        W_full = np.zeros((8, 28), dtype=float)

        offset = 0
        for bi in range(2):
            for bj in range(2):
                # Map local indices to global
                for ra in range(2):
                    for ca in range(2):
                        local_i = ra * 2 + ca
                        global_i = ra * 4 + (2 * bi + ca)
                        U_full[global_i, offset:offset + 7] = U_s[local_i, :]

                for rb in range(2):
                    for cb in range(2):
                        local_j = rb * 2 + cb
                        global_j = (2 * bi + rb) * 4 + (2 * bj + cb)
                        V_full[global_j, offset:offset + 7] = V_s[local_j, :]

                for rc in range(2):
                    for cc in range(2):
                        local_k = cc * 2 + rc
                        global_k = (2 * bj + cc) * 2 + rc
                        W_full[global_k, offset:offset + 7] = W_s[local_k, :]

                offset += 7

        return U_full, V_full, W_full

    U28, V28, W28 = build_rank28()

    # Verify rank-28 decomposition
    def verify_decomposition(U_full, V_full, W_full, T_target):
        max_err = 0.0
        for i in range(U_full.shape[0]):
            for j in range(V_full.shape[0]):
                for k in range(W_full.shape[0]):
                    val = np.dot(U_full[i, :] * V_full[j, :], W_full[k, :])
                    err = abs(val - T_target[i, j, k])
                    if err > max_err:
                        max_err = err
        return max_err

    err28 = verify_decomposition(U28, V28, W28, T)
    
    if err28 > 1e-9:
        # Fallback: schoolbook rank 32
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Strategy: Try to find a rank-27 decomposition.
    # We have 28 terms. We want to see if we can express the tensor as a sum of 27 rank-1 terms.
    # This is equivalent to finding a vector c in R^28 such that T = sum_{r=1}^{27} c_r * (u_r v_r^T w_r^T) 
    # where the new rank-1 terms are linear combinations of the original 28.
    
    # Actually, a simpler view: The space of decompositions is an affine subspace.
    # Let the 28 Strassen terms be a basis. Any other decomposition with the same U, V structure 
    # (if we fix U and V) is determined by W. But we can also vary U and V.
    
    # A more robust approach for "rank minimization" given a known high-rank decomposition:
    # Check if the tensor lies in the variety of rank <= 27 tensors.
    # This is hard. 
    
    # Alternative Structural Change:
    # Use the "border rank" or "approximate rank" concept? No, we need exact.
    
    # Let's try a different search:
    # We know rank 28 works. 
    # Can we merge two terms? 
    # If we have terms t1 and t2, can we replace them with one term t3?
    # This requires t1 + t2 = t3. This implies the 2-term sum is rank 1.
    # This is very restrictive.
    
    # Let's try to find if there is a linear dependency among the 28 rank-1 tensors 
    # that allows us to drop one term while adjusting the others?
    # No, the tensor T is fixed. T = sum_{i=1}^{28} u_i v_i w_i.
    # If we want rank 27, we need T = sum_{i=1}^{27} u'_i v'_i w'_i.
    
    # Let's attempt a random search for a rank-27 decomposition with integer/half-integer entries.
    # We will fix a random U and V (from a pool of small integers) and solve for W.
    # If the solution W has small integer/half-integer entries, we have a winner.
    
    # However, random search in high dimensions is inefficient.
    
    # Better Idea: Exploit the specific structure of 2x4x4.
    # The tensor is a direct sum of 4 2x2x2 tensors.
    # Rank(T_direct_sum) <= Rank(T1) + Rank(T2) + Rank(T3) + Rank(T4).
    # We know Rank(2x2x2) = 7. So Rank <= 28.
    # Is Rank(2x4x4) < 28?
    # According to literature, the rank of matrix multiplication tensor M_{2,4,4} is not definitively known to be < 28 in simple constructions, but lower bounds exist.
    # Actually, for M_{n,m,p}, if n=1, it's rank m*p. Here n=2.
    
    # Let's try to construct a rank 27 decomposition by "sharing" one product.
    # In the 4 blocks, if we can make one of the Strassen products shared or zero?
    # No, the inputs are generic.
    
    # Let's try a different partition.
    # What if we don't partition into 2x2 blocks?
    # What if we view A as 2x(2+2) and B as (2+2)x(2+2)?
    # This is what we did.
    
    # Let's try to verify if rank 27 is possible by checking the secant variety dimension?
    # Too complex for 120s.
    
    # Let's stick to the robust rank 28 construction but try to optimize the coefficients 
    # to be integers/half-integers (they already are).
    
    # Wait, the prompt asks for a "DIFFERENT strategy".
    # Current: Block-wise Strassen.
    # New: Direct Linear Algebra Search for Rank 27.
    
    # We will set up the linear system for rank 27.
    # Variables: U (8x27), V (16x27), W (8x27).
    # Equations: T[i,j,k] = sum_{r=1}^{27} U[i,r] V[j,r] W[k,r].
    # This is a system of 8*16*8 = 1024 equations.
    # Unknowns: 8*27 + 16*27 + 8*27 = 216 + 432 + 216 = 864.
    # Overdetermined system.
    
    # We can fix U and V to have a specific sparse structure and solve for W?
    # Or fix W?
    
    # Let's try fixing U and V to be "simple" (e.g., derived from Strassen but with one column removed or modified) and solving for W.
    
    # Strategy:
    # 1. Take the rank-28 U28, V28, W28.
    # 2. Try to remove one column (term) and see if the remaining 27 terms can be adjusted to still sum to T.
    #    This is equivalent to asking if the removed term lies in the span of the other 27 rank-1 tensors.
    #    Since the 28 rank-1 tensors are linearly independent (generically), removing one will not work unless we adjust the others.
    #    But we can't "adjust" them arbitrarily; we must find NEW U, V, W.
    
    # Let's try a randomized search for a rank-27 decomposition.
    # We will use the "tensor rank via linear algebra" approach.
    # Fix U to be a random 8x27 matrix with entries in {-1, 0, 1}.
    # Fix V to be a random 16x27 matrix with entries in {-1, 0, 1}.
    # Solve for W: W = (U * V)^+ * T_flattened? No, it's a 3-way product.
    # The equation is: vec(T) = (W^T \otimes V^T \otimes U^T) vec(1)? No.
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
    # This is a linear system in W if U and V are fixed.
    # For each (i,j,k), T[i,j,k] = sum_r (U[i,r] V[j,r]) W[k,r].
    # Let M_{(i,j), k, r} = U[i,r] V[j,r].
    # Then T[i,j,k] = sum_r M_{(i,j), k, r} W[k,r].
    # This is not a standard linear system in W because W[k,r] is coupled with k.
    # However, for a fixed k, T[i,j,k] = sum_r (U[i,r] V[j,r]) W[k,r].
    # This means for each k, we have a system:
    # T[:, :, k] = sum_r W[k,r] (U[:,r] V[:,r]^T)
    # Let A_k be the matrix with columns U[:,r] V[:,r]^T flattened? No.
    # Let's reshape.
    # T_k (8x16 matrix) = sum_{r=1}^{27} W[k,r] (U[:,r] V[:,r]^T)
    # This is a linear system in W[k, :] for each k.
    # The unknowns are W[k, 0], ..., W[k, 26].
    # The equations are 8*16 = 128.
    # The "coefficients" are the 27 rank-1 matrices U[:,r] V[:,r]^T.
    # We can solve for W[k, :] using least squares.
    # If the solution W[k, :] is consistent across all k (i.e., the W matrix we get is valid), and has nice entries, we have a solution.
    
    # We will try this for several random U, V pairs.
    
    def try_random_rank27(num_trials=500):
        for trial in range(num_trials):
            if time.time() - start_time > deadline:
                return None
            
            # Generate random sparse U and V
            # Use small integers to increase chance of integer W
            U_rand = np.random.randint(-1, 2, (8, 27)).astype(float)
            V_rand = np.random.randint(-1, 2, (16, 27)).astype(float)
            
            # Ensure U and V are not zero
            if np.all(U_rand == 0) or np.all(V_rand == 0):
                continue
                
            # Solve for W
            W_sol = np.zeros((8, 27), dtype=float)
            valid = True
            
            for k in range(8):
                # Build the system: T[:, :, k] = sum_r W[k,r] (U[:,r] V[:,r]^T)
                # Flatten T[:, :, k] to a vector of length 128
                T_k_vec = T[:, :, k].flatten()
                
                # Build the matrix A where A[:, r] = vec(U[:,r] V[:,r]^T)
                A_k = np.zeros((128, 27), dtype=float)
                for r in range(27):
                    M_r = np.outer(U_rand[:, r], V_rand[:, r])
                    A_k[:, r] = M_r.flatten()
                
                # Solve A_k @ W_k = T_k_vec
                W_k, residuals, rank, sv = np.linalg.lstsq(A_k, T_k_vec, rcond=None)
                
                # Check if the residual is small
                if len(residuals) > 0 and residuals[0] > 1e-6:
                    valid = False
                    break
                W_sol[k, :] = W_k
            
            if not valid:
                continue
                
            # Verify the full decomposition
            err = verify_decomposition(U_rand, V_rand, W_sol, T)
            if err < 1e-6:
                # Check if W has integer/half-integer entries
                # Round to nearest half-integer
                W_rounded = np.round(W_sol * 2) / 2
                err_rounded = verify_decomposition(U_rand, V_rand, W_rounded, T)
                if err_rounded < 1e-6:
                    return U_rand, V_rand, W_rounded
        
        return None

    result_27 = try_random_rank27(num_trials=200)
    
    if result_27 is not None:
        U_out, V_out, W_out = result_27
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
        
    # If rank 27 not found, return rank 28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
