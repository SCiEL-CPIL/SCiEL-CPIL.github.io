"""Matrix multiplication 3,3,6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Build the target tensor T[x,y,z] where:
    # x = i*m + j (A index, 9 entries)
    # y = j'*p + k' (B index, 18 entries)  
    # z = k''*n + i'' (C index, 18 entries)
    # T[x,y,z] = 1 iff i=i'', j=j', k=k''
    
    nm, mp, np_ = n * m, m * p, p * n  # 9, 18, 18
    
    T = np.zeros((nm, mp, np_), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                T[x, y, z] = 1.0
    T_flat = T.flatten()
    
    def build_decomp(U, V, W, r):
        """Build U, V, W matrices from coefficient arrays."""
        U_mat = np.zeros((nm, r))
        V_mat = np.zeros((mp, r))
        W_mat = np.zeros((np_, r))
        for c in range(r):
            U_mat[:, c] = U[c]
            V_mat[:, c] = V[c]
            W_mat[:, c] = W[c]
        return U_mat, V_mat, W_mat
    
    def compute_residual(U_mat, V_mat, W_mat, r):
        """Compute the residual tensor for given factor matrices."""
        res = np.zeros_like(T_flat)
        for c in range(r):
            uc = U_mat[:, c]
            vc = V_mat[:, c]
            wc = W_mat[:, c]
            res += np.einsum('i,j,k->ijk', uc, vc, wc).flatten()
        return res - T_flat
    
    def try_round_and_verify(U_mat, V_mat, W_mat, r):
        """Try rounding to integers/half-integers and verify exactness."""
        # Try rounding to nearest integer
        U_r = np.round(U_mat)
        V_r = np.round(V_mat)
        W_r = np.round(W_mat)
        res = compute_residual(U_r, V_r, W_r, r)
        if np.allclose(res, 0, atol=1e-9):
            return U_r, V_r, W_r
        
        # Try rounding to nearest half-integer
        U_r = np.round(U_mat * 2) / 2
        V_r = np.round(V_mat * 2) / 2
        W_r = np.round(W_mat * 2) / 2
        res = compute_residual(U_r, V_r, W_r, r)
        if np.allclose(res, 0, atol=1e-9):
            return U_r, V_r, W_r
        
        return None
    
    # First, try to find a lower-rank decomposition via numerical search
    # Known: 2x Laderman gives rank 46. Try to find something better.
    # Theoretical lower bound for 3x3x6 is at least 27 (since 3x3x3 needs 23)
    # Try ranks from 30 to 45
    
    best_result = None
    best_rank = 46  # Start with 2x Laderman
    
    # Try numerical optimization for various ranks
    ranks_to_try = [30, 32, 34, 36, 38, 40, 42, 44]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try a few random restarts
        max_restarts = 3
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + attempt * 1000 + r)
            
            # Random initialization
            U_init = np.random.randn(r, nm) * 0.5
            V_init = np.random.randn(r, mp) * 0.5
            W_init = np.random.randn(r, np_) * 0.5
            
            # Use gradient descent to minimize residual
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                
                U_mat = U_init.T  # (nm, r)
                V_mat = V_init.T  # (mp, r)
                W_mat = W_init.T  # (np_, r)
                
                # Compute gradient
                # d/dU_c = sum over (y,z) of T[y,z] * V_c[y] * W_c[z] - sum_c' ...
                # More efficiently:
                # grad_U[x,c] = sum_{y,z} (T[x,y,z] - res[x,y,z]) * V_c[y] * W_c[z]
                # But we need the residual first
                
                res = compute_residual(U_mat, V_mat, W_mat, r)
                res_tensor = res.reshape(nm, mp, np_)
                
                # Gradient for U
                grad_U = np.einsum('xyz,yz->xz', res_tensor, V_mat * W_mat)
                # Gradient for V
                grad_V = np.einsum('xyz,xz->yz', res_tensor, U_mat * W_mat)
                # Gradient for W
                grad_W = np.einsum('xyz,xy->xz', res_tensor, U_mat * V_mat)
                
                # Update
                U_init -= lr * grad_U.T
                V_init -= lr * grad_V.T
                W_init -= lr * grad_W.T
                
                # Check convergence
                if np.linalg.norm(res) < 1e-6:
                    break
            
            # Check if we converged
            U_mat = U_init.T
            V_mat = V_init.T
            W_mat = W_init.T
            res = compute_residual(U_mat, V_mat, W_mat, r)
            
            if np.linalg.norm(res) < 1e-6:
                # Try to round
                rounded = try_round_and_verify(U_mat, V_mat, W_mat, r)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    # Final verification
                    res_final = compute_residual(U_r, V_r, W_r, r)
                    if np.allclose(res_final, 0, atol=1e-9):
                        best_result = (U_r, V_r, W_r, r)
                        best_rank = r
                        break  # Found a valid decomposition at this rank
        
        if best_result is not None and best_rank <= r:
            break  # Found something at or below this rank, stop
    
    # If we found a better decomposition, use it
    if best_result is not None:
        U_r, V_r, W_r, r = best_result
        U_list = [[float(x) for x in row] for row in U_r]
        V_list = [[float(x) for x in row] for row in V_r]
        W_list = [[float(x) for x in row] for row in W_r]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: Use 2x Laderman (rank 46)
    # Laderman's rank-23 decomposition for 3x3 x 3x3 matrix multiplication.
    # For 3x3x6, split B into two 3x3 blocks: C = A*B1 + A*B2
    # Use Laderman for each block independently: rank = 23 + 23 = 46
    
    terms23 = [
        ([1,0,0, 0,0,0, 0,0,0],
         [1,0,0, -1,0,0, 1,1,0],
         [1,0,0, 0,0,0, 0,0,1]),
        ([1,1,0, 0,0,0, 0,0,0],
         [0,0,0, 1,0,0, 0,0,0],
         [0,0,1, 0,0,-1, 0,0,0]),
        ([1,0,1, 0,0,0, 0,0,0],
         [0,0,0, 0,0,0, 1,0,0],
         [0,0,1, 0,0,-1, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1],
         [1,1,0, -1,-1,0, 0,0,0],
         [1,0,-1, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 1,0,1],
         [0,1,0, 0,0,0, 0,0,0],
         [0,1,0, 0,-1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 1,0,1],
         [0,0,0, 0,1,0, 0,0,0],
         [0,-1,0, 0,1,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,1],
         [0,0,0, 1,1,0, -1,-1,0],
         [0,1,0, 0,-1,0, 0,0,1]),
        ([1,0,1, 0,-1,0, 1,0,1],
         [1,0,0, 0,0,0, 0,0,1],
         [1,0,0, 0,-1,0, 0,0,-1]),
        ([0,0,1, 0,-1,0, 0,0,1],
         [1,0,0, 0,0,0, 0,0,1],
         [-1,0,0, 0,1,0, -1,0,0]),
        ([1,0,1, 0,-1,0, 0,0,0],
         [0,0,0, 0,0,0, 1,0,1],
         [1,0,0, 0,-1,0, 1,0,0]),
        ([1,0,1, 0,0,0, 0,0,0],
         [0,0,0, -1,0,-1, 1,0,1],
         [0,1,0, 0,-1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 1,0,1],
         [0,1,1, 0,-1,-1, 0,0,0],
         [0,-1,0, 0,0,1, 0,0,-1]),
        ([0,0,0, 0,-1,0, 1,0,1],
         [0,1,1, 0,0,0, 0,0,0],
         [0,0,1, 0,0,-1, -1,0,0]),
        ([0,0,1, 0,-1,0, 0,0,1],
         [1,0,1, 0,0,0, 0,0,0],
         [1,0,0, 0,-1,0, 1,0,0]),
        ([0,0,1, 0,0,0, 0,0,1],
         [1,0,1, -1,0,-1, 0,0,0],
         [0,1,0, 0,-1,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,0,0],
         [0,0,0, 0,1,0, 0,-1,0],
         [0,-1,0, 0,1,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,0,0],
         [1,0,0, -1,0,0, 0,0,0],
         [-1,0,0, 0,1,-1, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0],
         [0,0,0, 0,1,0, 0,-1,0],
         [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,0],
         [0,0,0, 0,0,0, 1,0,-1],
         [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,1,0],
         [0,1,-1, 0,0,0, 0,0,0],
         [0,-1,0, 0,1,1, 0,0,0]),
        ([0,0,0, 0,0,0, 0,1,0],
         [0,0,0, 0,1,-1, 0,0,0],
         [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,0,0],
         [0,0,0, 0,1,1, 0,0,0],
         [-1,0,0, 0,1,-1, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,0],
         [1,0,1, 0,0,0, 0,0,0],
         [0,0,0, 0,0,0, 0,0,0]),
    ]
    
    rank = 46
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((np_, rank))
    
    for t, (alpha, beta, gamma) in enumerate(terms23):
        # First block (columns 0-2 of B)
        for i in range(3):
            for j in range(3):
                U[i*3+j, t] = alpha[i*3+j]
        for j in range(3):
            for k in range(3):
                V[j*6+k, t] = beta[j*3+k]
        for k in range(3):
            for i in range(3):
                W[k*3+i, t] = gamma[k*3+i]
        
        # Second block (columns 3-5 of B)
        for i in range(3):
            for j in range(3):
                U[i*3+j, t+23] = alpha[i*3+j]
        for j in range(3):
            for k in range(3):
                V[j*6+(k+3), t+23] = beta[j*3+k]
        for k in range(3):
            for i in range(3):
                W[(k+3)*3+i, t+23] = gamma[k*3+i]
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
