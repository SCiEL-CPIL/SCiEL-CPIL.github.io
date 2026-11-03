"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a low-rank exact tensor decomposition for 3x4x4 matrix multiplication.
    Returns {'U': ..., 'V': ..., 'W': ...} with shapes (12, rank), (16, rank), (12, rank).
    """
    n, m, p = DIMENSIONS
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Note: The task says W[k*n+i, r]. So W index is k*n + i.
    
    # The standard schoolbook rank is n*m*p = 3*4*4 = 48.
    # We want to minimize this rank.
    
    # Strategy:
    # 1. Try to find a rank < 48 decomposition using iterative numerical methods (ALS).
    # 2. If successful, round to integers/half-integers and verify exactly.
    # 3. If not, return the schoolbook decomposition (rank 48).
    
    # We have a 120s limit. Let's use a tight deadline.
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Helper to check time
    def time_exceeded():
        return (time.time() - start_time) > deadline
    
    # Initialize with schoolbook for fallback
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    best_rank = n*m*p
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Define the target tensor T as a 3D array
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    # Wait, the tensor for matrix multiplication is NOT all 1s.
    # The matrix multiplication tensor T[i, j, k] is 1 if the operation C[i,k] += A[i,j] * B[j,k] exists.
    # Yes, for standard matrix multiplication, every (i,j,k) term in the sum exists.
    # So T[i,j,k] = 1 for all i,j,k.
    # Let's verify the convention.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T is defined such that T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # So yes, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # Reshape T to match U, V, W dimensions for verification
    # T_flat[i*m+j, j*p+k, k*n+i] is not a simple reshape.
    # We need to construct the 3D array T3[i, j, k] = 1.
    # Then check sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == T3[i,j,k]
    
    # Let's use a numerical ALS approach to minimize rank.
    # We start with a rank slightly below schoolbook, e.g., 40, 35, etc.
    # However, ALS for tensor decomposition is non-convex and can get stuck.
    # Given the time limit, we should try a few random initializations with lower ranks.
    
    # Ranks to try: 45, 44, 43, 42, 40, 38, 35, 30, 25, 20
    # If we find a valid one, we stop.
    
    ranks_to_try = [47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 30, 28, 25, 22, 20]
    
    np.random.seed(42)
    
    for r in ranks_to_try:
        if time_exceeded():
            break
            
        # Try multiple random restarts for each rank
        # Fewer restarts for lower ranks if time is running out
        restarts = 5 if r > 40 else 10 if r > 30 else 20
        
        found_valid = False
        for _ in range(restarts):
            if time_exceeded():
                break
                
            # Initialize U, V, W with random values
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # ALS Iterations
            # Minimize || T - sum_c U_c V_c W_c ||_F
            # Unfold T into matrices.
            # T_mat1: (n*m, m*p) -> T[i*m+j, j*p+k] = T[i,j,k] ? No.
            # Let's define the unfolded tensors properly.
            
            # T1: dimensions (n*m, m*p)
            # T1[i*m+j, j*p+k] = T[i,j,k]
            # This requires careful indexing.
            T1 = np.zeros((n*m, m*p))
            T2 = np.zeros((m*p, p*n))
            T3 = np.zeros((p*n, n*m))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T1[i*m+j, j*p+k] = 1.0
                        T2[j*p+k, k*n+i] = 1.0
                        T3[k*n+i, i*m+j] = 1.0
            
            # Iterate
            for it in range(100):
                if time_exceeded():
                    break
                    
                # Update U: T1 - V W^T U => U = (V W^T)^-1 T1
                # V W^T is (m*p, n*m)
                VWt = V @ W.T
                # We want U such that U @ (V W^T)^T = T1 => U @ (W V^T) = T1
                # So U = T1 @ (W V^T)^+
                W_Vt = W @ V.T  # (p*n, n*m) -> Wait.
                # T1 (n*m, m*p) = U (n*m, r) * (V (m*p, r))^T (r, m*p)
                # T1 = U * V^T * W? No.
                # Term c: U[:, c] V[:, c] W[:, c]
                # T1[i*m+j, j*p+k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # This is not a simple matrix multiplication because W depends on k and i, not just a shared index.
                # Wait, the term is a rank-1 tensor.
                # T[i,j,k] = sum_c U[i,j]_c V[j,k]_c W[k,i]_c
                # Note that the indices of the three factors are (i,j), (j,k), (k,i).
                # This is a cyclic dependency. Standard matrix unfolding doesn't apply directly as simple rank-1 matrix sums.
                # However, we can still use ALS by fixing two factors and solving for the third.
                
                # Fix V, W. Solve for U.
                # For a fixed column c, we have a vector u_c.
                # The contribution to T is u_c (outer) v_c (outer) w_c.
                # Specifically, T[i,j,k] += u_c[i,j] v_c[j,k] w_c[k,i].
                # This is a linear system for u_c if v_c, w_c are fixed.
                # Let's construct a matrix A_U such that U * A_U = T_flat?
                # Let's vectorize T into a 1D array of length n*m*p.
                # The linear system is huge.
                
                # Alternative: Use a simpler heuristic or just rely on the schoolbook if ALS is too complex/slow.
                # Given the complexity and time limit, and the fact that Laderman's rank 19 is for 4x4x4,
                # and we are doing 3x4x4, it is likely that the rank is bounded by 48 but might be lower.
                # However, without a guaranteed fast solver, I will stick to a simpler approach or just return schoolbook.
                
                # Let's try a very simple heuristic:
                # Can we share some terms?
                # In 3x4x4, the "i" dimension is small (3).
                # For each i, we have a 4x4 matrix multiplication.
                # If we could use the same U, V structure for all i, but W changes.
                # But U depends on i.
                
                # Let's just do a random search for small improvements? No, too risky.
                
                # I will implement a correct, simple ALS step if possible.
                # Let's construct the linear system for U.
                # Variables: U[i*m+j, c]
                # Constraint: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                # Let M_U[i*m+j, k*n+i] be a matrix? No, the sum is over c.
                # For a fixed i, j, the equation is:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k.
                # This is a system of p equations for the r unknowns U[i*m+j, 0..r-1].
                # This is underdetermined if r > p.
                # We can use least squares.
                
                # This per-(i,j) system is independent for each i,j?
                # Yes, U[i*m+j, :] only appears in equations for that specific i,j.
                # So we can solve r * (n*m) independent least squares problems!
                # This is fast!
                
                # System for U[i*m+j, :]:
                # A_U (p x r) * u (r x 1) = b (p x 1)
                # A_U[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # b[k] = 1
                
                # Let's build these matrices.
                # This is O(n*m*p*r^2) roughly.
                
                # Let's implement this ALS step.
                
                # Pre-compute A_U for all i,j?
                # A_U[i*m+j] is a p x r matrix.
                # A_U[i*m+j][k, c] = V[j*p+k, c] * W[k*n+i, c]
                
                # Similarly for V and W.
                
                # Update U:
                # for i in n:
                #   for j in m:
                #     A = np.zeros((p, r))
                #     for k in p:
                #       for c in r:
                #         A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                #     u_new = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                #     U[i*m+j, :] = u_new
                
                # This is n*m*p*r operations per step.
                # 3*4*4*r = 48r. If r=20, 960 ops. Very fast.
                # 100 iterations * 960 * 3 (for U,V,W) = 288k ops. Trivial.
                
                # Let's do it.
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # A is p x r
                        A = np.zeros((p, r))
                        for k in range(p):
                            # V row j*p+k
                            # W row k*n+i
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A @ u = 1
                        try:
                            u, residuals, rank, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = u
                        except:
                            pass
                
                # Update V
                # For fixed j, k:
                # sum_c V[j*p+k, c] * U[i*m+j, c] * W[k*n+i, c] = 1 for all i
                # A_V (n x r) * v (r x 1) = b (n x 1)
                # A_V[i, c] = U[i*m+j, c] * W[k*n+i, c]
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            v, residuals, rank, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p+k, :] = v
                        except:
                            pass
                
                # Update W
                # For fixed k, i:
                # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j
                # A_W (m x r) * w (r x 1) = b (m x 1)
                # A_W[j, c] = U[i*m+j, c] * V[j*p+k, c]
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            w, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = w
                        except:
                            pass
                
                # Check convergence / validity
                # Calculate error
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (val - 1.0)**2
                            
                if err < 1e-10:
                    # Found a solution. Try to round it.
                    # Round to nearest 0.5
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    err_rounded = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for c in range(r):
                                    val += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                                err_rounded += (val - 1.0)**2
                    
                    if err_rounded < 1e-9:
                        best_rank = r
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        found_valid = True
                        break
                    
                    # If rounding failed, maybe try integer rounding?
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    err_int = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for c in range(r):
                                    val += U_int[i*m+j, c] * V_int[j*p+k, c] * W_int[k*n+i, c]
                                err_int += (val - 1.0)**2
                    if err_int < 1e-9:
                        best_rank = r
                        best_U = U_int
                        best_V = V_int
                        best_W = W_int
                        found_valid = True
                        break
        
        if found_valid:
            # If we found a valid decomposition for this rank, we can stop trying lower ranks immediately?
            # No, we want the MINIMAL rank. So we should continue to lower ranks.
            # But if we found rank 20, we should check 19, 18...
            # So we don't break the outer loop.
            pass

    # Return the best found (or schoolbook)
    # Convert to lists as required by interface? "factor matrices of shapes... with integer/half-integer entries"
    # The example returns lists of lists. I should probably return lists or numpy arrays?
    # "U': ... (n*m, rank)"
    # The seed program returns lists. I will return lists to be safe, or numpy arrays if accepted.
    # Let's check the seed: `return {"U": U, "V": V, "W": W}` where U is a list of lists.
    # I will convert numpy arrays to lists of lists.
    
    def to_list(arr):
        return arr.tolist()
        
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

# EVOLVE-BLOCK-END
