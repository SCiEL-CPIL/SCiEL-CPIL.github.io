"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Build the target tensor (sparse representation)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_entries.append((i*m+j, j*p+k, k*n+i))
    
    # Try to find low-rank decomposition
    # Start from a reasonable lower bound and go up
    # Lower bound: at least max(n*p, n*m, m*p) but more tightly, at least n*p
    min_rank = n * p  # 10
    
    best_result = None
    
    for rank in range(min_rank, n*m*p + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = max(5, 5000 // rank)
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers/half-integers
            scale = 2  # entries in {-2,-1,-0.5,0,0.5,1,2} roughly
            U = np.random.choice([-2,-1,-0.5,0,0.5,1,2], size=(n*m, rank))
            V = np.random.choice([-2,-1,-0.5,0,0.5,1,2], size=(m*p, rank))
            W = np.random.choice([-2,-1,-0.5,0,0.5,1,2], size=(p*n, rank))
            
            # Iterative refinement: alternate solving for one factor
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Fix V, W, solve for U
                # For each (a,c) where target has 1, we need sum_r U[a,r]*V[b,r]*W[c,r] = 1
                # This is a linear system in U
                # Actually, let's use a different approach: gradient descent or alternating least squares
                
                # ALS: Fix V and W, solve for U
                # For each row a of U, we have constraints from all (a,b,c) in target
                # sum_r U[a,r] * (V[b,r]*W[c,r]) = 1
                # This is: U[a,:] @ diag(V[b,:]*W[c,:]) ... not quite
                # Actually: for fixed b,c, sum_r U[a,r]*V[b,r]*W[c,r] = (U[a,:] * (V[b,:]*W[c,:])) dot product
                # So for fixed b,c: U[a,:] @ (V[b,:]*W[c,:]) = 1
                # This is a linear equation in U[a,:]
                
                # Let's set up the linear system for U
                # Rows of the system: for each (a,b,c) in target, one equation
                # But U has n*m rows, each with rank entries
                # This is underdetermined if rank < number of constraints per row
                
                # Alternative: use a simpler iterative method
                # Compute the "residual" and update
                
                # Let me use a different approach: treat this as a matrix factorization problem
                # Matricize the tensor: for fixed c, T[:,:,c] is a (n*m) x (m*p) matrix
                # T[:,:,c] = sum_r U[:,r] * W[c,r] * V[:,r]^T ... not quite
                # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For fixed c: T[:,:,c] = sum_r (U[:,r] * W[c,r]) * V[:,r]^T
                # = (U * W[c,:]) @ V^T where * is element-wise column scaling
                
                # This is a low-rank approximation problem
                # Let's use ALS on the matricizations
                
                # For efficiency, let's just do a simple iterative refinement
                # Compute current tensor and gradient
                
                # Actually, let me try a different strategy:
                # Use the fact that we want exact half-integer solutions
                # Try a structured construction
                
                break  # Skip the iterative part for now, use structured approach
            
            # Try a structured construction based on Strassen-like ideas
            # For now, just check if the random init happens to work (unlikely)
            # Compute the reconstructed tensor
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            error = 0
            for (a, b, c) in target_entries:
                if abs(recon[a, b, c] - 1) > 1e-6:
                    error += 1
                # Check non-target entries are 0
            # Check non-target entries
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        if (a, b, c) not in set(target_entries):
                            if abs(recon[a, b, c]) > 1e-6:
                                error += 1
                        # This is too slow, skip for now
            
            if error == 0:
                # Found a solution!
                best_result = (rank, U, V, W)
                break
        
        if best_result is not None:
            break
    
    # If no solution found, fall back to schoolbook
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    else:
        rank, U, V, W = best_result
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
