import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target Tensor T of shape (dim_u, dim_v, dim_w)
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    start_time = time.time()
    timeout = 110.0 # Leave 10s margin
    
    # Try ranks from a low estimate up to schoolbook
    # Schoolbook is 72.
    # Let's try to find a decomposition with rank < 72.
    # We will use random restarts and gradient descent.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook as fallback
    U_school = np.zeros((dim_u, best_rank))
    V_school = np.zeros((dim_v, best_rank))
    W_school = np.zeros((dim_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    # We will search for ranks between 40 and 71.
    # If we find a valid integer/half-integer decomposition, we return it.
    # Validity check: Reconstruct T and compare.
    
    def check_solution(U, V, W):
        # U, V, W are integer or half-integer.
        # Convert to float for check
        Uf = U.astype(np.float64)
        Vf = V.astype(np.float64)
        Wf = W.astype(np.float64)
        
        # Reconstruct tensor
        # T_recon[u, v, w] = sum_c U[u,c] V[v,c] W[w,c]
        # This is a tensor contraction.
        # We can do this efficiently by looping over c or using einsum.
        # einsum('uc,vc,wc->uvw', U, V, W)
        T_recon = np.einsum('uc,vc,wc->uvw', Uf, Vf, Wf)
        
        # Check if it matches T exactly
        # Since we use integers/half-integers, the result should be exact integers.
        # T is 0 or 1.
        if np.allclose(T_recon, T, atol=1e-9):
            return True
        return False

    # Search strategy:
    # 1. Try to find a real solution for various ranks.
    # 2. If found, try to snap to integers/half-integers.
    
    # Given the complexity, a full search might be too slow.
    # However, for 3x4x6, the rank is likely close to 72 or slightly less.
    # Let's try a simple heuristic:
    # Use the schoolbook decomposition and try to merge columns.
    # This is equivalent to finding a lower rank decomposition.
    
    # Alternatively, use a known decomposition if available.
    # For 3x4x6, is there a known rank?
    # M(3,4,4) <= 38?
    # M(3,4,6) might be <= 55?
    
    # Let's try a random search for integer solutions with rank R.
    # This is a constraint satisfaction problem.
    # We can use a simple local search:
    # Start with schoolbook.
    # Try to remove a column and adjust others?
    
    # Actually, let's just return the schoolbook solution if we can't find a better one quickly.
    # But the task asks for a "stronger algorithm".
    # Let's implement a basic gradient descent on real numbers, then snap.
    
    def solve_real(rank):
        # Initialize U, V, W randomly
        U = np.random.randn(dim_u, rank)
        V = np.random.randn(dim_v, rank)
        W = np.random.randn(dim_w, rank)
        
        # Gradient Descent
        lr = 0.01
        for step in range(2000):
            if time.time() - start_time > timeout:
                return None
            
            # Compute gradient
            # Loss = sum_{u,v,w} (T[u,v,w] - sum_c U[u,c]V[v,c]W[w,c])^2
            # dL/dU[u,c] = -2 * sum_{v,w} (T[u,v,w] - ... ) * V[v,c] * W[w,c]
            
            T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
            diff = T - T_recon
            
            # Gradient for U
            # grad_U[u, c] = -2 * sum_{v,w} diff[u,v,w] V[v,c] W[w,c]
            # We can compute this as:
            # temp[u, v, c] = diff[u, v, :] dot W[:, c] ?
            # grad_U = -2 * np.einsum('uvw,vc,wc->uc', diff, V, W)
            grad_U = -2.0 * np.einsum('uvw,vc,wc->uc', diff, V, W)
            grad_V = -2.0 * np.einsum('uvw,uc,wc->vc', diff, U, W)
            grad_W = -2.0 * np.einsum('uvw,uc,vc->wc', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Optional: normalize to prevent divergence
            # Not strictly necessary for small steps
            
        # Check if converged
        T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
        error = np.linalg.norm(T - T_recon)
        if error < 1e-6:
            return U, V, W
        return None

    # We will try ranks from 50 to 71.
    # If we find a real solution, we try to snap it.
    
    for rank in range(50, 72):
        if time.time() - start_time > timeout - 10:
            break
            
        # Try a few random restarts
        for restart in range(5):
            if time.time() - start_time > timeout - 10:
                break
                
            res = solve_real(rank)
            if res is not None:
                U_real, V_real, W_real = res
                
                # Snap to half-integers
                # Multiply by 2, round, divide by 2
                U_snap = np.round(U_real * 2) / 2
                V_snap = np.round(V_real * 2) / 2
                W_snap = np.round(W_real * 2) / 2
                
                if check_solution(U_snap, V_snap, W_snap):
                    # Found a valid solution!
                    # Convert to list of lists
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    
    # If no better solution found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
