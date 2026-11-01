<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    def check_time():
        return time.time() > deadline
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 54
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_schoolbook
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        if not U or not V or not W:
            return False
        
        # Convert to numpy arrays for efficient checking
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check shapes
        if U_arr.shape != (n * m, U_arr.shape[1]):
            return False
        if V_arr.shape != (m * p, U_arr.shape[1]):
            return False
        if W_arr.shape != (p * n, U_arr.shape[1]):
            return False
            
        rank = U_arr.shape[1]
        if rank == 0:
            return False
            
        # Build the tensor from decomposition
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Build target tensor
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1
        
        return np.allclose(T_decomp, T_target, atol=1e-9)
    
    # Given the lower bound on rank (54), we return the schoolbook decomposition directly.
    # This ensures a valid result within the time limit.
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    def check_time():
        return time.time() > deadline
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 54
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_schoolbook
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        if not U or not V or not W:
            return False
        
        # Convert to numpy arrays for efficient checking
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check shapes
        if U_arr.shape != (n * m, U_arr.shape[1]):
            return False
        if V_arr.shape != (m * p, U_arr.shape[1]):
            return False
        if W_arr.shape != (p * n, U_arr.shape[1]):
            return False
            
        rank = U_arr.shape[1]
        if rank == 0:
            return False
            
        # Build the tensor from decomposition
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Build target tensor
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1
        
        return np.allclose(T_decomp, T_target, atol=1e-9)
    
    def try_rank(r, max_iter=2000, tol=1e-8):
        """Try to find a rank-r decomposition using iterative optimization."""
        if r < 1:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Build target tensor
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1
        
        # Reshape target for efficient computation
        # We need: for all (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T_target[a,b,c]
        # This is equivalent to: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # sums to T_target
        
        # Use alternating minimization
        for iteration in range(max_iter):
            if check_time():
                return None
            
            # Update U: for each a, minimize ||sum_r U[a,r]*V[b,r]*W[c,r] - T_target[a,b,c]||
            # Fix V and W, solve for U
            # T_target[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
            # Let M_r[b,c] = V[b,r]*W[c,r]
            # Then T_target[a,:,:] = sum_r U[a,r] * M_r
            # This is a linear system: U[a,:] @ M = T_target[a,:,:]
            # where M is (r, m*p, p*n) reshaped...
            
            # More efficiently: T_target[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: sum_r U[a,r] * (V[:,r] * W[:,r] reshaped) = T_target[a,:,:]
            # Let's define for each r: outer_r = V[:,r] ⊗ W[:,r] which is (m*p, p*n)
            # Then T_target[a,:,:] = sum_r U[a,r] * outer_r
            # So [outer_0, outer_1, ..., outer_{r-1}] @ U[a,:]^T = T_target[a,:,:]
            
            # Build the matrix A_mat where A_mat[:,r] = vec(V[:,r] ⊗ W[:,r])
            # A_mat is (m*p*p*n, r)
            # And T_target reshaped is (n*m, m*p*p*n)
            # We need: T_flat[a,:] = U[a,:] @ A_mat^T
            # So U = T_flat @ pinv(A_mat)
            
            # But this is huge. Let's use a different approach.
            # Use the fact that we can update one factor at a time using least squares.
            
            # Update U: 
            # For each a (0 to n*m-1):
            #   sum_r U[a,r] * V[b,r] * W[c,r] = T_target[a,b,c] for all b,c
            #   Let G[a,b,c] = V[b,r] * W[c,r] summed over r weighted by U[a,r]
            #   This is: T_target[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
            #   Let B_r = V[:,r] outer W[:,r] (shape m*p x p*n)
            #   Then T_target[a,:,:] = sum_r U[a,r] * B_r
            #   Stacking: T_target[a] = U[a] @ B_stacked where B_stacked is (r, m*p*p*n)
            
            # This is getting complex. Let me use a simpler gradient-based approach.
            
            # Compute current tensor
            T_current = np.einsum('ar,br,cr->abc', U, V, W)
            error = T_current - T_target
            err_norm = np.linalg.norm(error)
            
            if err_norm < tol:
                break
            
            # Gradient descent with adaptive learning rate
            # Gradient w.r.t U[a,r] = sum_{b,c} error[a,b,c] * V[b,r] * W[c,r]
            grad_U = np.einsum('abc,br,cr->ar', error, V, W)
            grad_V = np.einsum('abc,ar,cr->br', error, U, W)
            grad_W = np.einsum('abc,ar,br->cr', error, U, V)
            
            # Adaptive learning rate
            lr_U = 0.1 * err_norm / (np.linalg.norm(grad_U) + 1e-10)
            lr_V = 0.1 * err_norm / (np.linalg.norm(grad_V) + 1e-10)
            lr_W = 0.1 * err_norm / (np.linalg.norm(grad_W) + 1e-10)
            
            U -= lr_U * grad_U
            V -= lr_V * grad_V
            W -= lr_W * grad_W
        
        # Check final error
        T_final = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_final, T_target, atol=1e-7):
            return (U, V, W)
        return None
    
    # Try ranks from low to high
    # Lower bound: max(n*p, m*p, n*m) = max(18, 27, 6) = 27
    # Actually the rank is at least max(n*p, m*p, n*m) for matrix multiplication tensors
    # But it can be higher. Let's try from 27 upward.
    
    for r in range(27, min(55, 55)):
        if check_time():
            break
        
        result = try_rank(r, max_iter=500)
        if result is not None:
            U, V, W = result
            best_U = [[float(x) for x in U[i]] for i in range(n * m)]
            best_V = [[float(x) for x in V[i]] for i in range(m * p)]
            best_W = [[float(x) for x in W[i]] for i in range(p * n)]
            best_rank = r
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize the gradient descent approach will be too slow and may not converge. Let me think of a better approach.

The key insight: for matrix multiplication $M_{n,m,p}$, the tensor rank is known. For $M_{2,3,9}$, let me think about what's achievable.

Actually, a much simpler constructive approach: Since $m=3$, each output element $C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]$. 

If I use $n \cdot p = 18$ rank-1 terms, one for each $(i,k)$ pair, I'd need each term to capture the sum over $j$. But a single rank-1 term $U \otimes V \otimes W$ can only give one product $A[i,j] \cdot B[j,k] \cdot C[k,i]$ for specific $i,j,k$.

So I need at least $n \cdot m \cdot p = 54$ terms if each term handles one $(i,j,k)$ triple. But I can share: if two terms have the same $j$ index in the middle, they can be combined.

Actually no - in a rank-1 term, $U[a,r]$, $V[b,r]$, $W[c,r]$ are fixed for that rank. The term contributes $U[i \cdot m + j, r] \cdot V[j \cdot p + k, r] \cdot W[k \cdot n + i, r]$ to $T[i \cdot m + j, j \cdot p + k, k \cdot n + i]$.

For the target, $T[i \cdot m + j, j \cdot p + k, k \cdot n + i] = 1$ only when the indices are consistent (which they always are by construction), and 0 otherwise.

The key constraint: $U[i \cdot m + j, r] \cdot V[j \cdot p + k, r] \cdot W[k \cdot n + i, r]$ must be zero unless $(i,j,k)$ is a valid triple, and sum to 1 over $r$ for each valid triple.

Since every $(i,j,k)$ with $i \in [0,n), j \in [0,m), k \in [0,p)$ is valid, we need:
$\sum_r U[i \cdot m + j, r] \cdot V[j \cdot p + k, r] \cdot W[k \cdot n + i, r] = 1$ for all $i,j,k$.

And for "invalid" entries (where the indices don't form a valid triple in the tensor support), the sum must be 0. But in our tensor, ALL entries of the form $(i \cdot m + j, j \cdot p + k, k \cdot n + i)$ are 1, and all other entries are 0.

Wait, the tensor is defined on indices $(a, b, c)$ where $a = i \cdot m + j$, $b = j \cdot p + k$, $c = k \cdot n + i$. Not all $(a,b,c)$ triples are in the support. The support is exactly the set of $(i \cdot m + j, j \cdot p + k, k \cdot n + i)$ for $i \in [0,n), j \in [0,m), k \in [0,p)$.

So the constraint is:
- For each $(i,j,k)$: $\sum_r U[i \cdot m + j, r] \cdot V[j \cdot p + k, r] \cdot W[k \cdot n + i, r] = 1$
- For all other $(a,b,c)$: $\sum_r U[a, r] \cdot V[b, r] \cdot W[c, r] = 0$

This is a system of equations. The number of non-zero constraints is $n \cdot m \cdot p = 54$. The number of zero constraints is $n \cdot m \cdot m \cdot p \cdot p \cdot n - n \cdot m \cdot p$ which is huge.

The rank of the tensor is the minimum $r$ such that there exist $U, V, W$ satisfying all constraints.

Let me try a different approach: use the fact that the multiplication tensor has a known structure and try to find a low-rank decomposition using linear algebra.

Actually, let me just try to solve this as a linear system for small ranks. For a given rank $r$, we have $n \cdot m \cdot r + m \cdot p \cdot r + p \cdot n \cdot r$ variables and $n \cdot m \cdot m \cdot p \cdot p \cdot n$ constraints (but most are redundant).

This is getting complex. Let me just use a simpler heuristic: try rank values and use scipy to solve.

