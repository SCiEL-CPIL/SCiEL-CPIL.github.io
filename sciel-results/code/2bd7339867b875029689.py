"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Build the list of nonzero tensor entries
    # T[a, b, c] = 1 where a = k*n+i, b = i*m+j, c = j*p+k
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                entries.append((a, b, c, 1))
    
    # Try to find a lower rank decomposition using alternating least squares
    # For a given rank r, we want U (n*m × r), V (m*p × r), W (p*n × r)
    # such that sum_r U[b,r] * V[c,r] * W[a,r] = T[a,b,c] for all entries
    
    # This is a 3-way tensor completion problem
    
    def solve_rank(r, max_iter=50, seed=0):
        """Try to find a rank-r decomposition using ALS."""
        if time.time() > deadline - 30:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n * m, r) * 0.01
        V = rng.randn(m * p, r) * 0.01
        W = rng.randn(p * n, r) * 0.01
        
        for iteration in range(max_iter):
            if time.time() > deadline - 20:
                return None
            
            # Update U: for each b, solve for U[b, :]
            # Given V and W, the equation is:
            # sum_r U[b,r] * V[c,r] * W[a,r] = T[a,b,c]
            # For fixed b, this is linear in U[b, :]
            # Let M_b be a matrix where M_b[(a,c), r] = V[c,r] * W[a,r]
            # Then M_b @ U[b, :] = t_b where t_b[(a,c)] = T[a,b,c]
            
            for b in range(n * m):
                # Collect all entries with this b
                mask = [e for e in entries if e[1] == b]
                if not mask:
                    continue
                
                # Build the system
                A_mat = np.zeros((len(mask), r))
                t_vec = np.zeros(len(mask))
                for idx, (a, bb, c, t) in enumerate(mask):
                    for r_idx in range(r):
                        A_mat[idx, r_idx] = V[c, r_idx] * W[a, r_idx]
                    t_vec[idx] = t
                
                # Solve least squares
                if len(mask) >= r:
                    try:
                        U[b, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass
            
            # Similarly for V and W
            for c in range(m * p):
                mask = [e for e in entries if e[2] == c]
                if not mask:
                    continue
                A_mat = np.zeros((len(mask), r))
                t_vec = np.zeros(len(mask))
                for idx, (a, b, cc, t) in enumerate(mask):
                    for r_idx in range(r):
                        A_mat[idx, r_idx] = U[b, r_idx] * W[a, r_idx]
                    t_vec[idx] = t
                if len(mask) >= r:
                    try:
                        V[c, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass
            
            for a in range(p * n):
                mask = [e for e in entries if e[0] == a]
                if not mask:
                    continue
                A_mat = np.zeros((len(mask), r))
                t_vec = np.zeros(len(mask))
                for idx, (aa, b, c, t) in enumerate(mask):
                    for r_idx in range(r):
                        A_mat[idx, r_idx] = U[b, r_idx] * V[c, r_idx]
                    t_vec[idx] = t
                if len(mask) >= r:
                    try:
                        W[a, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass
        
        # Verify the solution
        for a, b, c, t in entries:
            val = np.sum(U[b, :] * V[c, :] * W[a, :])
            if abs(val - t) > 1e-4:
                return None
        
        return U, V, W
    
    # Try ranks from lower to higher
    best_result = None
    for r in range(40, min(rank_school + 1, 106)):
        if time.time() > deadline - 40:
            break
        
        result = solve_rank(r, max_iter=30, seed=r)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = U_school, V_school, W_school
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for x in row:
                # Round to nearest half-integer
                half = round(x * 2) / 2
                if abs(x - half) < 1e-6:
                    row_list.append(int(half) if half == int(half) else half)
                else:
                    row_list.append(float(x))
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
