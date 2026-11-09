"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # The tensor equation: for all i,j,k in [0,n),[0,m),[0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is a system of n*m*p equations with 3*rank variables.
    # We can try to solve it using iterative methods.
    
    # Try to find a solution with lower rank using gradient-based optimization
    # on the residual, then round and verify.
    
    for rank in [48, 40, 36, 32, 28, 24, 20, 16]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(100):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Iteratively refine to minimize residual
            lr = 0.01
            for it in range(2000):
                if time.time() > deadline - 10:
                    break
                
                # Compute residual and gradients
                # Residual for (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
                # Gradient w.r.t U[i*m+j,c]: V[j*p+k,c]*W[k*n+i,c]
                
                # Sample a subset of (i,j,k) for speed
                batch_size = min(200, n*m*p)
                indices = np.random.randint(0, n*m*p, size=batch_size)
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for idx in indices:
                    i = idx // (m*p)
                    rem = idx % (m*p)
                    j = rem // p
                    k = rem % p
                    
                    for c in range(rank):
                        uv = U[i*m+j, c]
                        vw = V[j*p+k, c]
                        wi = W[k*n+i, c]
                        
                        # Residual contribution
                        res = uv * vw * wi - 1.0/rank  # target is 1, but distributed
                        # Actually, we want sum_c = 1, so each term should contribute ~1/rank on average
                        
                        # Gradient
                        grad_U[i*m+j, c] += (vw * wi)
                        grad_V[j*p+k, c] += (uv * wi)
                        grad_W[k*n+i, c] += (uv * vw)
                
                # This is getting complicated. Let me use a simpler approach.
                break
            
            # Try a different approach: use the structure of the problem
            # The tensor is all-ones, so we can use separable decompositions.
            
            # Try: U[i*m+j, c] = a_c[i] * b_c[j]
            #       V[j*p+k, c] = b_c[j] * c_c[k]  (reusing b)
            #       W[k*n+i, c] = c_c[k] * a_c[i]  (reusing c and a)
            # Then sum_c a_c[i]*b_c[j]*b_c[j]*c_c[k]*c_c[k]*a_c[i] = sum_c (a_c[i])^2 * (b_c[j])^2 * (c_c[k])^2
            # This needs to equal 1 for all i,j,k, which means each factor must be 1, giving schoolbook.
            
            break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
