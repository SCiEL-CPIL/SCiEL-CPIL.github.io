# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def construct():
    n, m, p = 2, 2, 7
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n) = (4, 14, 14)
    # T[i, j, k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # i = a_idx * m + b_idx (a_idx in 0..n-1, b_idx in 0..m-1)
    # j = b_idx * p + c_idx (b_idx in 0..m-1, c_idx in 0..p-1)
    # k = c_idx * n + a_idx (c_idx in 0..p-1, a_idx in 0..n-1)
    
    T = np.zeros((n*m, m*p, p*n))
    for a_idx in range(n):
        for b_idx in range(m):
            for c_idx in range(p):
                i = a_idx * m + b_idx
                j = b_idx * p + c_idx
                k = c_idx * n + a_idx
                T[i, j, k] = 1
    
    # We want to find minimum rank r such that T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    # where u_t ∈ R^{n*m}, v_t ∈ R^{m*p}, w_t ∈ R^{p*n}
    
    # Strategy: Use greedy algorithm with ILP relaxation
    # First, try to find a good initial solution using greedy, then refine
    
    # Greedy approach: iteratively pick the best rank-1 approximation
    best_U = []
    best_V = []
    best_W = []
    residual = T.copy()
    
    max_rank = 30  # Upper bound on search
    
    for rank in range(20, max_rank + 1):
        # Try to find rank decomposition using iterative refinement
        U_list = []
        V_list = []
        W_list = []
        res = T.copy()
        
        for _ in range(rank):
            # Find best rank-1 approximation to residual
            # Use SVD on flattened tensor
            flat_res = res.flatten()
            # This is expensive, so use a simpler heuristic:
            # Find the largest entry in residual and use it as seed
            max_idx = np.unravel_index(np.argmax(np.abs(res)), res.shape)
            i_max, j_max, k_max = max_idx
            
            if abs(res[i_max, j_max, k_max]) < 1e-10:
                break
            
            # Initialize with unit vectors at the max position
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[i_max] = 1
            v[j_max] = 1
            w[k_max] = 1
            
            # Refine using power iteration on the tensor
            for _iter in range(20):
                # Update u: u_new[i] = sum_{j,k} res[i,j,k] * v[j] * w[k]
                temp_u = np.einsum('ijk,j,k->i', res, v, w)
                # Update v: v_new[j] = sum_{i,k} res[i,j,k] * u[i] * w[k]
                temp_v = np.einsum('ijk,i,k->j', res, u, w)
                # Update w: w_new[k] = sum_{i,j} res[i,j,k] * u[i] * v[j]
                temp_w = np.einsum('ijk,i,j->k', res, u, v)
                
                # Normalize
                if np.linalg.norm(temp_u) > 1e-10:
                    u = temp_u / np.linalg.norm(temp_u)
                if np.linalg.norm(temp_v) > 1e-10:
                    v = temp_v / np.linalg.norm(temp_v)
                if np.linalg.norm(temp_w) > 1e-10:
                    w = temp_w / np.linalg.norm(temp_w)
            
            # Compute coefficient
            coeff = np.einsum('ijk,i,j,k', res, u, v, w)
            if abs(coeff) < 1e-10:
                break
            
            # Subtract
            res -= coeff * np.einsum('i,j,k->ijk', u, v, w)
            
            # Store (we'll scale later)
            U_list.append(coeff * u)
            V_list.append(v)
            W_list.append(w)
            
            # Check if residual is close to zero
            if np.max(np.abs(res)) < 1e-8:
                break
        
        # Check if we found a valid decomposition
        if np.max(np.abs(res)) < 1e-8:
            actual_rank = len(U_list)
            # Scale to get integer/half-integer entries if possible
            U_final = np.array(U_list).T  # shape (n*m, actual_rank)
            V_final = np.array(V_list).T  # shape (m*p, actual_rank)
            W_final = np.array(W_list).T  # shape (p*n, actual_rank)
            
            # Try to round to integers or half-integers
            # Multiply by 2 to get integers if they're half-integers
            scale = 2
            U_scaled = U_final * scale
            V_scaled = V_final * scale
            W_scaled = W_final * scale
            
            # Check if close to integers
            if (np.max(np.abs(U_scaled - np.round(U_scaled))) < 1e-6 and
                np.max(np.abs(V_scaled - np.round(V_scaled))) < 1e-6 and
                np.max(np.abs(W_scaled - np.round(W_scaled))) < 1e-6):
                U_int = np.round(U_scaled).astype(int)
                V_int = np.round(V_scaled).astype(int)
                W_int = np.round(W_scaled).astype(int)
                # Verify
                T_recon = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int) / scale
                if np.max(np.abs(T_recon - T)) < 1e-6:
                    # Convert to list format
                    U_out = U_int.tolist()
                    V_out = V_int.tolist()
                    W_out = W_int.tolist()
                    return {"U": U_out, "V": V_out, "W": W_out}
            
            # If not integer, try without scaling
            U_int = np.round(U_final).astype(int)
            V_int = np.round(V_final).astype(int)
            W_int = np.round(W_final).astype(int)
            T_recon = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
            if np.max(np.abs(T_recon - T)) < 1e-6:
                U_out = U_int.tolist()
                V_out = V_int.tolist()
                W_out = W_int.tolist()
                return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: Use the original rank-25 solution
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_W_local = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]
    
    strassen_V_local = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    for b in range(3):
        for r in range(7):
            for i in range(4):
                U[i][col] += strassen_U[r][i]
            
            for loc_idx, coeff in strassen_V_local[r].items():
                if loc_idx == 0:
                    v_idx = 0 * p + (2 * b)
                elif loc_idx == 1:
                    v_idx = 0 * p + (2 * b + 1)
                elif loc_idx == 2:
                    v_idx = 1 * p + (2 * b)
                elif loc_idx == 3:
                    v_idx = 1 * p + (2 * b + 1)
                V[v_idx][col] += coeff
            
            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0:
                    w_idx = (2 * b) * n + 0
                elif loc_idx == 1:
                    w_idx = (2 * b + 1) * n + 0
                elif loc_idx == 2:
                    w_idx = (2 * b) * n + 1
                elif loc_idx == 3:
                    w_idx = (2 * b + 1) * n + 1
                W[w_idx][col] += coeff
            
            col += 1
            
    schoolbook_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    
    schoolbook_V_local = [
        {0: 1},
        {1: 1},
        {0: 1},
        {1: 1}
    ]
    
    schoolbook_W_local = [
        {0: 1},
        {0: 1},
        {1: 1},
        {1: 1}
    ]
    
    for r in range(4):
        for i in range(4):
            U[i][col] += schoolbook_U[r][i]
            
        for loc_idx, coeff in schoolbook_V_local[r].items():
            if loc_idx == 0:
                v_idx = 0 * p + 6
            elif loc_idx == 1:
                v_idx = 1 * p + 6
            V[v_idx][col] += coeff
            
        for loc_idx, coeff in schoolbook_W_local[r].items():
            if loc_idx == 0:
                w_idx = 6 * n + 0
            elif loc_idx == 1:
                w_idx = 6 * n + 1
            W[w_idx][col] += coeff
            
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
