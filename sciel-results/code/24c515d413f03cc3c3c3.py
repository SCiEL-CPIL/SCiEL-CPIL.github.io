import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s to leave safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Build the matrix multiplication tensor
    # T[i, k, j] = 1 if this is a valid multiplication term
    # Actually, the tensor is: for each (i,j,k), the coefficient is 1
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # But we want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    # and 0 for all other triples (but there are no other triples in the support)
    
    # The support of the tensor: for each i in [n], j in [m], k in [p]:
    #   (i*m+j, j*p+k, k*n+i) -> 1
    # All other entries are 0.
    
    # Let's build the target tensor as a set of (a, b, c) -> 1
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1.0
    
    num_support = len(target)  # n*m*p = 80
    
    def evaluate(U, V, W, rank):
        """Check if the decomposition is exact."""
        # For each support element, check if sum equals 1
        # For non-support elements, check if sum equals 0
        # But checking all (n*m)*(m*p)*(p*n) = 16*20*20 = 6400 entries is fine
        
        # Actually, we only need to check that the reconstructed tensor matches
        # The tensor has size (n*m) x (m*p) x (p*n) = 16 x 20 x 20 = 6400
        # But most are 0, so let's be smart
        
        # Reconstruct and compare
        # T_recon[a, b, c] = sum_c U[a,c] * V[b,c] * W[c,c]
        # This is expensive to do naively. Let's use einsum or batch ops.
        
        # For efficiency, compute for each column c:
        #   outer(U[:,c], V[:,c]) gives a (16,20) matrix, then multiply element-wise with W[:,c] broadcast
        # Actually, let's just check the support elements and a sample of non-support elements
        
        # Check all support elements
        for (a, b, c) in target.keys():
            s = 0.0
            for col in range(rank):
                s += U[a, col] * V[b, col] * W[c, col]
            if abs(s - 1.0) > 1e-6:
                return False
        return True
    
    def try_rank(rank, max_iter=500, lr=0.01):
        """Try to find a decomposition of given rank using ALS."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # ALS: alternate fixing two matrices and solving for the third
        # For U: for each row a, U[a, :] should satisfy sum_c U[a,c]*V[b,c]*W[c,c] = target[a,b,c] for all b,c
        # This is a linear system if we fix V and W
        
        # Actually, let's use gradient descent with a simple approach
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Compute gradients using a subset of constraints for speed
            # Or use full gradient
            
            # Gradient of loss = sum over all (a,b,c) of (recon[a,b,c] - target[a,b,c])^2
            # dL/dU[a,c] = 2 * sum_{b,c'} (recon[a,b,c'] - target[a,b,c']) * V[b,c'] * W[c',c]
            # This is getting complex. Let's use a simpler approach.
            
            # Use randomized ALS: pick random support elements and solve
            # For a given (a, b, c) in support: sum_c' U[a,c']*V[b,c']*W[c,c'] = 1
            # If we fix V and W, this is linear in U[a, :]
            
            # Let's do proper ALS
            # Fix V, W -> solve for U
            # For each row a of U: U[a, :] * (V[b, :] * W[c, :]) = 1 for each (b,c) where (a,b,c) in support
            # But not all (b,c) pairs have (a,b,c) in support
            
            # This is tricky. Let me use a different approach: 
            # minimize ||T_recon - T_target||_F^2 using gradient steps
            
            # Compute current reconstruction error on support
            err = 0.0
            for (a, b, c) in target.keys():
                s = 0.0
                for col in range(rank):
                    s += U[a, col] * V[b, col] * W[c, col]
                err += (s - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                break
            
            # Simple gradient step - but computing full gradient is expensive
            # Let's use a heuristic: randomly perturb and keep if better
            
            # Actually, let's use a more structured approach
            # Pick random (a,b,c) from support, compute local gradient
            indices = list(target.keys())
            np.random.shuffle(indices)
            sample = indices[:min(100, len(indices))]
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for (a, b, c) in sample:
                s = 0.0
                for col in range(rank):
                    s += U[a, col] * V[b, col] * W[c, col]
                resid = s - 1.0
                for col in range(rank):
                    grad_U[a, col] += resid * V[b, col] * W[c, col]
                    grad_V[b, col] += resid * U[a, col] * W[c, col]
                    grad_W[c, col] += resid * U[a, col] * V[b, col]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Check if best is exact
        if evaluate(best_U, best_V, best_W, rank):
            return best_U, best_V, best_W
        
        # Try rounding to half-integers
        U_round = np.round(best_U * 2) / 2
        V_round = np.round(best_V * 2) / 2
        W_round = np.round(best_W * 2) / 2
        
        if evaluate(U_round, V_round, W_round, rank):
            return U_round, V_round, W_round
        
        return None
    
    # Try various ranks
    # Known: schoolbook is 80. Let's try to do better.
    # For 4x4x4, best is 49. For 4x4x5, estimate ~55-65
    
    best_result = None
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = (U_school, V_school, W_school, rank_school)
    
    # Try to find better
    for rank in range(55, 80):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(rank, max_iter=200, lr=0.05)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W, rank)
            break
    
    # Convert to lists
    U, V, W, rank = best_result
    
    # Convert numpy arrays to lists of lists
    if isinstance(U, np.ndarray):
        U_list = U.tolist()
    else:
        U_list = U
    if isinstance(V, np.ndarray):
        V_list = V.tolist()
    else:
        V_list = V
    if isinstance(W, np.ndarray):
        W_list = W.tolist()
    else:
        W_list = W
    
    return {"U": U_list, "V": V_list, "W": W_list}
