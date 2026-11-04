"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Tensor T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want to find the minimal rank r such that this holds for all i,j,k.
    
    # The schoolbook rank is n*m*p = 2*3*9 = 54.
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    
    # Strategy:
    # 1. Construct the schoolbook decomposition (rank 54) as a guaranteed fallback.
    # 2. Try to find a lower rank decomposition using random initialization and optimization.
    #    Specifically, we can try to solve for the entries of U, V, W such that the
    #    tensor product matches the identity tensor (all 1s).
    #    Since the problem asks for integer/half-integer entries, we will try to snap
    #    the solution to the nearest half-integer and verify.
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """
        Verify if U, V, W form a valid decomposition.
        U, V, W are numpy arrays or lists.
        Returns True if valid, False otherwise.
        """
        # Convert to numpy for speed
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape to 3D tensors for easier broadcasting?
        # U: (n, m, rank)
        # V: (m, p, rank)
        # W: (p, n, rank)
        U_3d = U_arr.reshape(n, m, -1)
        V_3d = V_arr.reshape(m, p, -1)
        W_3d = W_arr.reshape(p, n, -1)
        
        # We need to check for all i, j, k:
        # sum_c U[i, j, c] * V[j, k, c] * W[k, i, c] == 1
        
        # This is a bit tricky to vectorize directly without expanding dimensions too much.
        # Let's use einsum.
        # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        
        # U[i, j, c] * V[j, k, c] -> M[i, j, k, c]
        # Then sum over c with W[k, i, c]
        
        # Alternative: Loop over i, j and vectorize k?
        # Or just loop over i, j, k but use dot products for the inner sum?
        # The rank is small (<=54). n*m*p = 54. 
        # 54 iterations of dot products of length <=54 is very fast.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # u is vector of length rank
                    # v is vector of length rank
                    # w is vector of length rank
                    u = U_arr[i * m + j, :]
                    v = V_arr[j * p + k, :]
                    w = W_arr[k * n + i, :]
                    
                    # val = sum(u * v * w)
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_int(arr):
        """Snap values to nearest half-integer."""
        return [[int(round(x * 2)) / 2.0 for x in row] for row in arr]

    def try_random_search(rank, attempts=100):
        """
        Try to find a decomposition of given rank using random initialization
        and local optimization.
        """
        if time.time() > deadline - 5:
            return None
        
        # We fix W to simplify the gauge freedom? 
        # Actually, let's just use random init for all 3.
        # Objective: minimize sum (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2
        
        best_loss = float('inf')
        best_params = None
        
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_init = np.random.randn(n * m, rank)
            V_init = np.random.randn(m * p, rank)
            W_init = np.random.randn(p * n, rank)
            
            # Flatten parameters
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                U_flat = params[:n * m * rank]
                V_flat = params[n * m * rank : n * m * rank + m * p * rank]
                W_flat = params[n * m * rank + m * p * rank :]
                
                U_mat = U_flat.reshape(n * m, rank)
                V_mat = V_flat.reshape(m * p, rank)
                W_mat = W_flat.reshape(p * n, rank)
                
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Sum over c
                            val = 0.0
                            for c in range(rank):
                                val += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                            loss += (val - 1.0) ** 2
                return loss

            # Use Nelder-Mead or BFGS? Nelder-Mead is derivative-free and robust for small problems.
            # But the dimension is rank*(n*m + m*p + p*n).
            # For rank=27: 27*(6+27+18) = 27*51 = 1377 dimensions. Too high for NM.
            # We need a smarter approach.
            
            # Alternative: Fix two matrices and solve for the third?
            # If we fix U and V, then for each (k,i), we have a linear system for W[k*n+i, :].
            # Equations: sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1 for all j.
            # This is a system of m equations for rank unknowns.
            # If rank >= m, we can solve it (least squares).
            
            # Let's try this "alternating least squares" approach.
            # 1. Initialize W randomly.
            # 2. Iterate:
            #    - Fix V, W. Solve for U.
            #      For each (i,j), sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k.
            #      System of p equations for rank unknowns.
            #    - Fix U, W. Solve for V.
            #      For each (j,k), sum_c V[j*p+k, c] * (U[i*m+j, c] * W[k*n+i, c]) = 1 for all i.
            #      System of n equations for rank unknowns.
            #    - Fix U, V. Solve for W.
            #      For each (k,i), sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1 for all j.
            #      System of m equations for rank unknowns.
            
            W_mat = np.random.randn(p * n, rank)
            U_mat = np.random.randn(n * m, rank)
            V_mat = np.random.randn(m * p, rank)
            
            for it in range(50):
                if time.time() > deadline - 5:
                    return None
                
                # Solve for U
                U_mat = np.zeros((n * m, rank))
                for i in range(n):
                    for j in range(m):
                        # Equations: sum_k ( V[j*p+k, c] * W[k*n+i, c] ) * U[i*m+j, c] = 1
                        # Let A[c] = sum_k V[j*p+k, c] * W[k*n+i, c]
                        # We want sum_c A[c] * U[i*m+j, c] = 1
                        # This is 1 equation for rank unknowns. Under-determined if rank > 1.
                        # We can pick the minimum norm solution or just set U[i*m+j, c] = A[c] / sum(A^2)
                        A = np.zeros(rank)
                        for k in range(p):
                            A += V_mat[j * p + k, :] * W_mat[k * n + i, :]
                        norm_sq = np.sum(A * A)
                        if norm_sq > 1e-9:
                            U_mat[i * m + j, :] = A / norm_sq
                        else:
                            U_mat[i * m + j, :] = 0
                
                # Solve for V
                V_mat_new = np.zeros((m * p, rank))
                for j in range(m):
                    for k in range(p):
                        # Equations: sum_i ( U[i*m+j, c] * W[k*n+i, c] ) * V[j*p+k, c] = 1
                        A = np.zeros(rank)
                        for i in range(n):
                            A += U_mat[i * m + j, :] * W_mat[k * n + i, :]
                        norm_sq = np.sum(A * A)
                        if norm_sq > 1e-9:
                            V_mat_new[j * p + k, :] = A / norm_sq
                        else:
                            V_mat_new[j * p + k, :] = 0
                V_mat = V_mat_new
                
                # Solve for W
                W_mat_new = np.zeros((p * n, rank))
                for k in range(p):
                    for i in range(n):
                        # Equations: sum_j ( U[i*m+j, c] * V[j*p+k, c] ) * W[k*n+i, c] = 1
                        A = np.zeros(rank)
                        for j in range(m):
                            A += U_mat[i * m + j, :] * V_mat[j * p + k, :]
                        norm_sq = np.sum(A * A)
                        if norm_sq > 1e-9:
                            W_mat_new[k * n + i, :] = A / norm_sq
                        else:
                            W_mat_new[k * n + i, :] = 0
                W_mat = W_mat_new
                
                # Check convergence
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = U_mat[i * m + j, :]
                            v = V_mat[j * p + k, :]
                            w = W_mat[k * n + i, :]
                            val = np.dot(u * v, w)
                            loss += (val - 1.0) ** 2
                
                if loss < 1e-6:
                    # Found a real-valued solution. Try to snap to half-integers.
                    U_snap = snap_to_half_int(U_mat.tolist())
                    V_snap = snap_to_half_int(V_mat.tolist())
                    W_snap = snap_to_half_int(W_mat.tolist())
                    
                    if verify(U_snap, V_snap, W_snap):
                        return {"U": U_snap, "V": V_snap, "W": W_snap}
                    
                    # If snapping fails, try a few perturbations? Or just return None for this attempt.
                    # It's possible the real solution is not close to half-integers.
                    # But the problem states entries are integer/half-integer.
                    # So if a solution exists, it should be close to such a grid.
        
        return None

    # Try ranks from lower bound upwards
    # Lower bound is 27.
    # Let's try 27, 28, ..., up to 54.
    # Given time constraints, we can only try a few.
    
    # First, check if rank 27 works with a few random restarts.
    # If not, try 28, etc.
    
    # To save time, we'll limit attempts per rank.
    # Rank 27: 50 attempts.
    # Rank 28: 20 attempts.
    # ...
    
    # Actually, the alternating least squares method above is quite specific.
    # It finds the minimum norm solution at each step, which might not be optimal for finding a global solution.
    # But it's a good heuristic.
    
    # Let's try a few ranks.
    for rank in range(27, 55):
        if time.time() > deadline - 10:
            break
        
        # Decrease attempts as rank increases, assuming higher ranks are easier but we've already spent time.
        attempts = max(5, 50 - (rank - 27) * 2)
        
        res = try_random_search(rank, attempts)
        if res is not None:
            return res
        
        # If we find a real solution that snaps to integers/half-integers, we return it.
        # If not, we continue.
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
