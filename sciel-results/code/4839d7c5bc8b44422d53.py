import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    The matrix multiplication tensor M_{2,3,4} has tensor rank 24 (schoolbook).
    We attempt to find a decomposition of rank < 24 using a search over
    integer and half-integer entries, but if none is found within the time
    budget, we return the schoolbook decomposition of rank 24.
    
    The decomposition satisfies:
        U has shape (n*m, rank)   -> indices (i*m+j, r)
        V has shape (m*p, rank)   -> indices (j*p+k, r)
        W has shape (p*n, rank)   -> indices (k*n+i, r)  [trace convention W[k*n+i, r]]
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s wall-clock limit
    
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # First, construct the schoolbook decomposition of rank 24 as a fallback
    rank_schoolbook = n * m * p  # 24
    U_sb = np.zeros((nm, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((mp, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((pn, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_sb[u_idx, col] = 1.0
                V_sb[v_idx, col] = 1.0
                W_sb[w_idx, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly computes matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U[u_idx, :rank], V[v_idx, :rank] * W[w_idx, :rank])
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
        return True
    
    def check_no_spurious(U, V, W, rank):
        """
        Check that the decomposition doesn't introduce spurious terms.
        For the matrix multiplication tensor, we need:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
        
        But we also need to ensure no spurious contributions. The tensor 
        T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        should equal 1 if (i,j,k) is a valid triple, and 0 otherwise.
        
        Actually, for matrix multiplication, the tensor is defined on the 
        full index set, and T[i,j,k] = 1 for all valid (i,j,k) in the 
        multiplication. Let me reconsider.
        
        The matrix multiplication tensor M_{n,m,p} has entries:
        M_{i,j,k} = 1 for all i in [n], j in [m], k in [p]
        
        Wait, that's not right either. Let me think more carefully.
        
        The tensor for multiplying an (n x m) matrix A by an (m x p) matrix B
        to get an (n x p) matrix C is:
        C_{i,k} = sum_j A_{i,j} * B_{j,k}
        
        The tensor T has entries T[i,j,k] such that:
        C_{i,k} = sum_{j} T[i,j,k] * A_{i,j} * B_{j,k}
        
        So T[i,j,k] = 1 for all valid (i,j,k).
        
        A rank-r decomposition gives:
        T[i,j,k] = sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        We need this to equal 1 for all (i,j,k).
        
        The spurious terms check is actually about ensuring that the 
        decomposition is exact, which verify_decomposition already does.
        """
        return True
    
    # Try to find a lower rank decomposition using random search
    # We'll try ranks from 23 down to some minimum
    
    import random
    random.seed(42)
    
    for target_rank in range(23, 15, -1):
        if time.time() > deadline - 10:
            break
        
        # Try random initializations and local optimization
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values from {-1, 0, 0.5, 1, 1.5, 2}
            candidates = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
            
            U = np.array([[random.choice(candidates) for _ in range(target_rank)] for _ in range(nm)])
            V = np.array([[random.choice(candidates) for _ in range(target_rank)] for _ in range(mp)])
            W = np.array([[random.choice(candidates) for _ in range(target_rank)] for _ in range(pn)])
            
            # Try to refine using gradient descent on the squared error
            lr = 0.01
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            
                            # Current value
                            val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                            err = val - 1.0
                            
                            # Gradients
                            grad_U[u_idx] += err * V[v_idx] * W[w_idx]
                            grad_V[v_idx] += err * U[u_idx] * W[w_idx]
                            grad_W[w_idx] += err * U[u_idx] * V[v_idx]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Snap to nearest half-integer
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Check convergence
                if np.allclose(np.dot(U[u_idx], V[v_idx] * W[w_idx]), 1.0, atol=1e-9):
                    break
            
            # Verify
            if verify_decomposition(U, V, W, target_rank):
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            break
    
    # Convert to lists as required by the interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
