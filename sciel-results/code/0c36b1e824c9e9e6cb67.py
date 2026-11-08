import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    def search_rank(n, m, p, target_rank, max_time=10):
        """
        Search for a decomposition of the n x m x p matrix multiplication tensor
        with the given target rank.
        Returns (U, V, W) as numpy arrays if found, else None.
        """
        local_start = time.time()
        
        # We search in the 3D space U3[n, r], V3[m, r], W3[p, r]
        # and map to interface format U_flat[n*m, r], V_flat[m*p, r], W_flat[p*n, r]
        
        best_error = float('inf')
        best_factors = None
        
        for _ in range(50): # Reduced restarts to save time
            if time.time() - local_start > max_time or time.time() > deadline:
                break
            
            U3 = np.random.randn(n, target_rank)
            V3 = np.random.randn(m, target_rank)
            W3 = np.random.randn(p, target_rank)
            
            for _ in range(100):
                if time.time() - local_start > max_time or time.time() > deadline:
                    break
                
                Approx = np.einsum('ic,jc,kc->ijk', U3, V3, W3)
                Error = Approx - 1.0
                err_norm = np.linalg.norm(Error)
                
                if err_norm < 1e-5:
                    # Found a solution
                    U_flat = np.zeros((n*m, target_rank))
                    V_flat = np.zeros((m*p, target_rank))
                    W_flat = np.zeros((p*n, target_rank))
                    
                    for i in range(n):
                        for j in range(m):
                            U_flat[i*m+j] = U3[i]
                    for j in range(m):
                        for k in range(p):
                            V_flat[j*p+k] = V3[j]
                    for k in range(p):
                        for i in range(n):
                            W_flat[k*n+i] = W3[k]
                    
                    # Round to half-integers
                    U_flat = np.round(U_flat * 2) / 2
                    V_flat = np.round(V_flat * 2) / 2
                    W_flat = np.round(W_flat * 2) / 2
                    
                    # Verify exactness
                    is_exact = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_flat[i*m+j] * V_flat[j*p+k] * W_flat[k*n+i])
                                if abs(val - 1.0) > 1e-6:
                                    is_exact = False
                                    break
                        if not is_exact:
                            break
                    if not is_exact:
                        break
                    
                    if is_exact:
                        return U_flat, V_flat, W_flat
                
                # Gradient Descent
                grad_U = np.einsum('ijk,jc,kc->ic', Error, V3, W3)
                grad_V = np.einsum('ijk,ic,kc->jc', Error, U3, W3)
                grad_W = np.einsum('ijk,ic,jc->kc', Error, U3, V3)
                
                lr = 0.01
                U3 -= lr * grad_U
                V3 -= lr * grad_V
                W3 -= lr * grad_W
                
                if err_norm < best_error:
                    best_error = err_norm
                    best_factors = (U3.copy(), V3.copy(), W3.copy())
        
        return None

    # Search for 2x3x3 decomposition
    U3, V3, W3 = None, None, None
    for r in range(15, 19):
        if time.time() > deadline:
            break
        print(f"Searching for rank {r} for 2x3x3...")
        res = search_rank(2, 3, 3, r, max_time=5)
        if res:
            U3, V3, W3 = res
            print(f"Found rank {r} for 2x3x3")
            break
    
    if U3 is None:
        # Fallback to schoolbook for 2x3x3, rank 18
        print("Falling back to schoolbook for 2x3x3")
        r = 18
        U3 = np.zeros((2, r))
        V3 = np.zeros((3, r))
        W3 = np.zeros((3, r))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U3[i, col] = 1
                    V3[j, col] = 1
                    W3[k, col] = 1
                    col += 1
    else:
        # Ensure they are numpy arrays (in case search returned lists, though we changed it to return arrays)
        U3 = np.array(U3)
        V3 = np.array(V3)
        W3 = np.array(W3)

    # Duplicate for 2x3x6
    # The 2x3x6 tensor is the direct sum of two 2x3x3 tensors.
    r_total = 2 * U3.shape[1]
    U_total = np.zeros((2*3, r_total))
    V_total = np.zeros((3*6, r_total))
    W_total = np.zeros((6*2, r_total))
    
    # First half (k=0,1,2)
    for i in range(2):
        for j in range(3):
            U_total[i*3+j, :U3.shape[1]] = U3[i]
    for j in range(3):
        for k in range(3):
            V_total[j*6+k, :U3.shape[1]] = V3[j]
    for k in range(3):
        for i in range(2):
            W_total[k*2+i, :U3.shape[1]] = W3[k]
    
    # Second half (k=3,4,5)
    offset = U3.shape[1]
    for i in range(2):
        for j in range(3):
            U_total[i*3+j, offset:] = U3[i]
    for j in range(3):
        for k in range(3, 6):
            V_total[j*6+k, offset:] = V3[j]
    for k in range(3, 6):
        for i in range(2):
            W_total[k*2+i, offset:] = W3[k]
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
