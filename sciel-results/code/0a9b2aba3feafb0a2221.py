import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Target tensor T is defined by T[(i,j), (j',k'), (i',k)] = 1 if i==i', j==j', k==k' else 0
    # We want U (nm x r), V (mp x r), W (pn x r) such that
    # sum_c U[i,j,c] * V[j',k',c] * W[i',k,c] == delta(i,i') * delta(j,j') * delta(k,k')
    
    # We can flatten the indices.
    # U index: i*m + j
    # V index: j'*p + k'
    # W index: i'*n + k
    
    # The condition is:
    # For all i, j, j', k', i', k:
    # sum_c U[i*m+j, c] * V[j'*p+k', c] * W[i'*n+k, c] == 1 if (i==i' and j==j' and k==k') else 0
    
    # This is a system of n*m*m*p*p*n equations? No.
    # The number of independent constraints is n*m*p (for the 1s) + (nm*mp*pn - nmp) (for the 0s).
    # This is huge. But we can exploit the structure.
    
    # Instead of checking all, we can check the reconstruction of the tensor.
    # Let's represent the tensor as a 3D array T[i, j, k] ? No, the indices are coupled.
    
    # Let's use a simpler check:
    # The matrix multiplication tensor can be seen as the identity map on the "middle" dimension?
    # Actually, we can verify the decomposition by checking if the induced bilinear map is correct.
    # But that's expensive.
    
    # Alternative: 
    # The tensor T is the sum of nmp rank-1 tensors:
    # T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}
    # We want to approximate this with r rank-1 tensors.
    
    # We will try to find a decomposition with rank r.
    # Start with r = 100 (schoolbook) and try to reduce.
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = n * m * p
    
    # Schoolbook decomposition
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
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n*m*p
    
    # Try to find a lower rank decomposition
    # We will use a random search + local optimization for a few target ranks.
    
    # Define a function to check if a decomposition is valid
    def is_valid(U, V, W, tol=1e-5):
        # Check a subset of constraints to be fast
        # Check all (i,j,k) for the 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(len(U[0])):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        # Check some random 0s
        # There are many 0s. We can check a few random ones.
        # Total entries in U*V*W space is (nm)*(mp)*(pn) = 20*25*20 = 10000
        # We check 100 random ones
        for _ in range(100):
            i = np.random.randint(n)
            j = np.random.randint(m)
            jp = np.random.randint(m)
            kp = np.random.randint(p)
            ip = np.random.randint(n)
            k = np.random.randint(p)
            
            # Skip if it's a 1
            if i == ip and j == jp and k == kp:
                continue
                
            val = 0.0
            for c in range(len(U[0])):
                val += U[i*m+j, c] * V[jp*p+kp, c] * W[ip*n+k, c]
            if abs(val) > tol:
                return False
        return True

    # Define a local optimization step
    def optimize(U, V, W, steps=100):
        # Simple gradient descent on the error
        # Error = sum (sum_c U... - T)^2
        # This is expensive to compute fully.
        # We'll use a few random samples to estimate the gradient.
        
        r = len(U[0])
        lr = 0.01
        
        for step in range(steps):
            # Sample a few constraints
            sample_size = 50
            indices = []
            for _ in range(sample_size):
                i = np.random.randint(n)
                j = np.random.randint(m)
                jp = np.random.randint(m)
                kp = np.random.randint(p)
                ip = np.random.randint(n)
                k = np.random.randint(p)
                indices.append((i, j, jp, kp, ip, k))
                
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for (i, j, jp, kp, ip, k) in indices:
                target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                val = 0.0
                for c in range(r):
                    val += U[i*m+j, c] * V[jp*p+kp, c] * W[ip*n+k, c]
                err = val - target
                
                for c in range(r):
                    # Gradient of err w.r.t U[i*m+j, c] is V[jp*p+kp, c] * W[ip*n+k, c]
                    dU[i*m+j, c] += err * V[jp*p+kp, c] * W[ip*n+k, c]
                    dV[jp*p+kp, c] += err * U[i*m+j, c] * W[ip*n+k, c]
                    dW[ip*n+k, c] += err * U[i*m+j, c] * V[jp*p+kp, c]
                    
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
        return U, V, W

    # Try ranks from best_rank - 1 down to a lower bound
    # Lower bound is roughly n*m*p / max(n,m,p) ? No.
    # For 4x5x5, let's try down to 80.
    
    for target_rank in range(best_rank - 1, max(best_rank - 20, 80), -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            U, V, W = optimize(U, V, W, steps=50)
            
            if is_valid(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
                
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
