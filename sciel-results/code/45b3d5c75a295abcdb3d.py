"""Rewritten program for the suite task; attempts rank-27 refinement of Strassen-28."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 140  # Safety margin

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    def strassen_2x2_decomp():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        # C[i,j] -> index j*2 + i
        U[0,0]=1.0; U[3,0]=1.0; V[0,0]=1.0; V[3,0]=1.0; W[0,0]=1.0; W[3,0]=1.0
        U[2,1]=1.0; U[3,1]=1.0; V[0,1]=1.0; W[1,1]=1.0; W[3,1]=-1.0
        U[0,2]=1.0; V[1,2]=1.0; V[3,2]=-1.0; W[2,2]=1.0; W[3,2]=1.0
        U[3,3]=1.0; V[2,3]=1.0; V[0,3]=-1.0; W[0,3]=1.0; W[1,3]=1.0
        U[0,4]=1.0; U[1,4]=1.0; V[3,4]=1.0; W[0,4]=-1.0; W[2,4]=1.0
        U[2,5]=1.0; U[0,5]=-1.0; V[0,5]=1.0; V[1,5]=1.0; W[3,5]=1.0
        U[1,6]=1.0; U[3,6]=-1.0; V[2,6]=1.0; V[3,6]=1.0; W[0,6]=1.0
        return U, V, W

    def build_strassen28():
        U_s, V_s, W_s = strassen_2x2_decomp()
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))
        blocks = [
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+lk,          lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+lk,      lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+(lk+2),      lambda lk,li: (lk+2)*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+(lk+2),  lambda lk,li: (lk+2)*2+li),
        ]
        for bi,(ga,gb,gc) in enumerate(blocks):
            for t in range(7):
                col = bi*7+t
                for li in range(2):
                    for lj in range(2):
                        U_full[ga(li,lj), col] = U_s[li*2+lj, t]
                for lj in range(2):
                    for lk in range(2):
                        V_full[gb(lj,lk), col] = V_s[lj*2+lk, t]
                for lk in range(2):
                    for li in range(2):
                        W_full[gc(lk,li), col] = W_s[lk*2+li, t]
        return U_full, V_full, W_full

    U28, V28, W28 = build_strassen28()
    
    # Fallback check
    if not verify(U28, V28, W28, tol=1e-8):
        # If Strassen 28 is somehow invalid (should not happen), use schoolbook
        r_school = n*m*p
        U_sch = np.zeros((n*m, r_school)); V_sch = np.zeros((m*p, r_school)); W_sch = np.zeros((p*n, r_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sch[i*m+j, col]=1.0; V_sch[j*p+k, col]=1.0; W_sch[k*n+i, col]=1.0; col+=1
        return to_lists(U_sch, V_sch, W_sch, r_school)

    # === Refinement: Attempt Rank 27 ===
    # Strategy: The rank-28 decomposition has 28 columns.
    # We suspect a rank-27 solution exists.
    # We will attempt to find a solution in the space of rank-27 tensors.
    # We initialize with the Strassen-28 solution but drop one column and adjust the rest?
    # No, dropping a column creates a huge error.
    # Instead, we treat the rank-27 problem as:
    # Minimize || sum_{t=0}^{26} U_t V_t W_t - T ||
    # We initialize U, V, W for rank 27 by taking the first 26 columns of Strassen-28 and 
    # randomizing the 27th, or by projecting the 28-column solution onto a 27-column subspace.
    
    # A better approach: Use the "Strassen-like" structure.
    # The tensor M_2,4,4 can be viewed as 2 blocks of M_2,2,2 in the A-dimension? No.
    # It is 4 blocks of M_2,2,2 if we split A (2x4) into two 2x2 blocks?
    # A is 2x4. B is 4x4. C is 4x2.
    # A = [A11 A12; A21 A22] (2x4)
    # B = [B11 B12; B21 B22] (4x4)
    # C = [C11 C21; C12 C22] (4x2) -> C_ij is 2x2
    #
    # C = A B
    # C_11 = A_11 B_11 + A_12 B_21
    # C_12 = A_11 B_12 + A_12 B_22
    # C_21 = A_21 B_11 + A_22 B_21
    # C_22 = A_21 B_12 + A_22 B_22
    #
    # This is exactly 4 independent M_2,2,2 multiplications if A and B blocks were independent.
    # But they share rows/cols.
    # Rank(M_2,4,4) <= 4 * 7 = 28. This is the Strassen bound.
    # Can we do better?
    # R(M_2,4,4) is known to be 27? 
    # According to literature, R(M_{2,4,4}) = 27.
    # We will search for it.

    def try_rank_27(seed):
        rng = np.random.RandomState(seed)
        r = 27
        
        # Initialize using a perturbation of the 28-column solution?
        # It's hard to project 28 to 27.
        # Let's initialize with random integers/halves? No, too far.
        # Let's use the structure:
        # We assume the 27th term is a "correction" term.
        
        # A strong heuristic: 
        # The 28-term Strassen solution has symmetries.
        # We can try to find a linear combination of the 28 terms that sums to zero (a dependency).
        # If sum_{t=0}^{27} alpha_t U_t V_t W_t = 0, and sum alpha_t != 0, we can rescale.
        # But this doesn't reduce rank unless one coefficient is 0? No.
        # If there is a linear dependency among the 28 rank-1 tensors, the rank is < 28.
        # We check if the 28 vectors in the tensor space are linearly dependent.
        
        # Flatten T
        T_flat = T.flatten()
        
        # Build the 28 rank-1 tensors
        basis = np.zeros((28, T_flat.size))
        for t in range(28):
            u = U28[:, t]
            v = V28[:, t]
            w = W28[:, t]
            tensor = np.einsum('a,b,c->abc', u, v, w)
            basis[t] = tensor.flatten()
        
        # Check rank of basis matrix
        # If rank(basis) < 28, then there is a dependency.
        # We can find null space vector z such that sum z_t basis_t = 0.
        # Then we can express one basis tensor in terms of others?
        # No, if they are dependent, the span has dimension < 28.
        # But we need to represent T using rank-1 tensors.
        # The set {basis_t} spans T.
        # If dim(span) = 27, then T can be written as a combination of 27 linearly independent basis tensors.
        # But the coefficients in the combination will be linear combinations of the original rank-1 tensors?
        # No. T = sum_{t=0}^{27} 1.0 * basis_t.
        # If basis is dependent, T is in a 27-dim subspace.
        # Does this mean T has rank 27?
        # The rank of T is the minimum number of rank-1 tensors needed.
        # If the 28 specific rank-1 tensors are dependent, it doesn't automatically mean T has rank 27.
        # It means T can be written as a sum of 27 *linear combinations* of these specific rank-1 tensors.
        # But a linear combination of rank-1 tensors is NOT a rank-1 tensor.
        # So this line of reasoning is flawed for reducing rank directly via linear dependency of the specific decomposition.
        
        # Correct approach:
        # We need to find U, V, W of shape (2, 27), (4, 27), (4, 27) such that:
        # T[a,b,c] = sum_{t=0}^{26} U[a,t] V[b,t] W[c,t]
        
        # Let's use a local search from a "nearby" point.
        # We can start with the 28-term solution and try to "merge" two terms?
        # Or simply run gradient descent on random initializations constrained to small integers?
        
        # Given time constraints, we will run a few iterations of a simple coordinate descent / gradient step
        # on the 27 variables, initialized by dropping the least significant term?
        
        # Actually, a known rank-27 decomposition for 2x4x4 exists.
        # Let's try to find it by solving the linear system for the coefficients if we fix the structure.
        
        # We will attempt a random restart optimization.
        # Objective: Minimize L2 error.
        
        best_err = float('inf')
        best_sol = None
        
        for it in range(50):
            if time.time() > deadline:
                break
            
            # Initialize
            # Start with Strassen 28, but we need 27 columns.
            # We can try to express the 28th column as a combination of the first 27?
            # If U28[:, 27] is in span(U28[:, :27]), etc.
            
            # Let's just use random integer initialization in [-1, 1] range (halves)
            # and refine.
            
            U = rng.randn(n*m, r) * 0.5
            V = rng.randn(m*p, r) * 0.5
            W = rng.randn(p*n, r) * 0.5
            
            # Gradient Descent
            for step in range(200):
                if time.time() > deadline:
                    break
                
                recon = np.einsum('at,bt,ct->abc', U, V, W)
                F = recon - T
                err = np.sum(F*F)
                
                if err < best_err:
                    best_err = err
                    best_sol = (U.copy(), V.copy(), W.copy())
                    
                if err < 1e-8:
                    break
                
                gU = np.einsum('abc,bt,ct->at', F, V, W)
                gV = np.einsum('abc,at,ct->bt', F, U, W)
                gW = np.einsum('abc,at,bt->ct', F, U, V)
                
                # Adaptive Learning Rate
                lr = 0.1 / (1.0 + err**0.5)
                
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
            # Snap to halves and verify
            Us = np.round(best_sol[0]*2)/2
            Vs = np.round(best_sol[1]*2)/2
            Ws = np.round(best_sol[2]*2)/2
            
            if verify(Us, Vs, Ws, tol=1e-8):
                return Us, Vs, Ws
        
        return None

    # Try to find rank 27
    # Note: The search space is huge. Random initialization is unlikely to succeed in 140s.
    # However, if we can find a dependency in the Strassen 28 solution, we might be able to
    # construct a rank 27 solution.
    
    # Let's check the rank of the Strassen 28 basis again, more carefully.
    # If the 28 basis tensors are linearly dependent, say sum z_t T_t = 0.
    # Then T = sum_{t=0}^{27} T_t = sum_{t=0}^{26} T_t + sum_{t=0}^{27} z_t T_t ? No.
    # T = sum_{t=0}^{27} 1 T_t.
    # If sum z_t T_t = 0, then T_k = -1/z_k sum_{j!=k} z_j T_j (if z_k != 0).
    # This expresses T_k as a linear combination of other T_j's.
    # But T is a tensor, and the sum is a tensor sum.
    # This shows the span of {T_0, ..., T_27} has dimension <= 27.
    # It does NOT mean T can be written as a sum of 27 rank-1 tensors.
    # It means T is in the span of 27 specific rank-1 tensors.
    # But the coefficients in that span representation are fixed by T.
    # If T = c_1 T_1 + ... + c_27 T_27, and T_i are rank 1, this is a decomposition of T into 27 rank-1 terms?
    # YES. If T is exactly equal to a linear combination of 27 rank-1 tensors, then the rank of T is <= 27.
    # So, if the 28 Strassen tensors are linearly dependent, the rank of T is <= 27.
    
    # Let's verify linear dependence of the Strassen 28 tensors.
    basis_28 = np.zeros((28, n*m*m*p*p*n))
    for t in range(28):
        u = U28[:, t]; v = V28[:, t]; w = W28[:, t]
        tensor = np.einsum('a,b,c->abc', u, v, w)
        basis_28[t] = tensor.flatten()
    
    # SVD to find rank
    try:
        U_svd, s, Vt = np.linalg.svd(basis_28, full_matrices=False)
        rank_28 = np.sum(s > 1e-8)
    except:
        rank_28 = 28
        
    if rank_28 < 28:
        # The tensors are dependent. T is in their span.
        # We can find a basis of 27 independent tensors from the set.
        # Then T = sum_{j=1}^{27} c_j T'_j.
        # This is a valid rank-27 decomposition!
        # We just need to compute the coefficients c_j.
        
        # Select 27 independent rows from basis_28 (or columns of the transpose)
        # Let's use the 27 vectors with largest singular values? 
        # Or simply take the first 27 that are independent.
        
        # Simpler: Solve basis_28.T @ coeff = T_flat
        # If rank is 27, the system is underdetermined but consistent.
        # We want a sparse solution? Or just ANY solution.
        # If we find coeff such that sum coeff_t T_t = T, and we can zero out one coeff_t by swapping basis?
        # No, we just need to return U, V, W such that sum U_t V_t W_t = T.
        # If T = sum_{t=0}^{26} c_t T_t, then we can define:
        # U_new[:, t] = c_t U28[:, t] for t=0..26
        # V_new[:, t] = V28[:, t] for t=0..26
        # W_new[:, t] = W28[:, t] for t=0..26
        # This gives T = sum_{t=0}^{26} (c_t U28[:, t]) V28[:, t] W28[:, t]
        # This is a rank-27 decomposition (assuming c_t != 0 for all, and independence).
        
        # Let's find the coefficients.
        # Solve basis_28.T (28 x N) x coeff (28 x 1) = T_flat (N x 1)
        # Use least squares, but check residual.
        
        # Since they are dependent, the pseudoinverse will give a minimum norm solution.
        # We can then try to zero out one coefficient by pivoting?
        # Actually, if we just use the minimum norm solution, it might have all 28 coeffs non-zero.
        # But we know there is a dependency.
        # Let's find the null space vector z of basis_28.
        # basis_28 @ z = 0.
        # Then we can add alpha * z to any solution coeff to get another solution.
        # coeff' = coeff + alpha * z.
        # We want one component of coeff' to be 0.
        # Suppose we want coeff'_k = 0.
        # coeff_k + alpha * z_k = 0 => alpha = -coeff_k / z_k.
        # Then coeff' = coeff - (coeff_k / z_k) * z.
        # This sets the k-th term to 0.
        # The remaining 27 terms form a rank-27 decomposition.
        
        # Find null space of basis_28 (28 x N).
        # Null space is left null space of basis_28? No.
        # We want z such that basis_28.T z = 0? No.
        # basis_28 is (28, N).
        # We want z (28,1) such that sum z_t T_t = 0.
        # So basis_28 @ z = 0.
        
        try:
            # Use SVD of basis_28
            U_svd, s, Vt = np.linalg.svd(basis_28, full_matrices=True)
            # Null space is columns of Vt corresponding to zero singular values.
            # s has length min(28, N). If rank < 28, some s are 0.
            # Vt is (min(28,N), N). This is not the null space of basis_28 directly.
            # We need null space of basis_28 (28 x N).
            # It is the null space of basis_28.T (N x 28)? No.
            # basis_28 @ z = 0. z is 28x1.
            # This is the null space of the matrix basis_28.
            # If rank < 28, the null space dimension is > 0.
            
            # Let's find the null space using scipy? No, std lib only.
            # Use np.linalg.svd on basis_28.T (N x 28).
            # Null space of basis_28 is the left null space of basis_28.T?
            # A x = 0. x is in null(A).
            # A = basis_28.
            # SVD of A.T: A.T = U s Vt.
            # The right singular vectors (columns of Vt) corresponding to 0 singular values span the null space of A.
            
            A_t = basis_28.T
            U_A, s_A, Vt_A = np.linalg.svd(A_t, full_matrices=True)
            # s_A has length min(N, 28).
            # If rank < 28, the last (28 - rank) entries of s_A are 0.
            # The corresponding columns of Vt_A (which is 28x28 if full_matrices=True) span the null space.
            
            # Find index of last non-zero singular value
            tol_svd = 1e-8
            num_non_zero = np.sum(s_A > tol_svd)
            
            if num_non_zero < 28:
                # Null space vectors are Vt_A[num_non_zero:, :].T
                # We want one vector z.
                z = Vt_A[num_non_zero, :].T  # 28x1
                
                # Solve for coeff using least squares (minimum norm)
                # basis_28 @ coeff = T_flat
                coeff, residuals, rank_ls, sv = np.linalg.lstsq(basis_28, T_flat, rcond=None)
                
                # Adjust coeff to zero out one component
                # Choose index k with largest |z_k| to avoid division by small numbers
                k = np.argmax(np.abs(z))
                if abs(z[k]) > 1e-8:
                    alpha = -coeff[k] / z[k]
                    coeff_new = coeff + alpha * z
                    coeff_new[k] = 0.0 # Force exact zero
                    
                    # Construct U, V, W for rank 27
                    U_new = np.zeros((n*m, 27))
                    V_new = np.zeros((m*p, 27))
                    W_new = np.zeros((p*n, 27))
                    
                    col_idx = 0
                    for t in range(28):
                        if t == k:
                            continue
                        U_new[:, col_idx] = coeff_new[t] * U28[:, t]
                        V_new[:, col_idx] = V28[:, t]
                        W_new[:, col_idx] = W28[:, t]
                        col_idx += 1
                        
                    # Verify
                    if verify(U_new, V_new, W_new, tol=1e-8):
                        return to_lists(U_new, V_new, W_new, 27)
                        
        except Exception:
            pass

    # If the above didn't work (e.g., rank was 28, or verification failed), return 28
    return to_lists(U28, V28, W28, 28)

# EVOLVE-BLOCK-END
