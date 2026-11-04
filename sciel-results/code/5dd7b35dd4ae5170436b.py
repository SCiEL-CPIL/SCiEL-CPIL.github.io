import time
import sys
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # n=4, m=4, p=9
    n, m, p = 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen 2x2 terms
    strassen_2x2 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        def add_inner(A_blocks, B_blocks, C_blocks):
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                for loc, coeff in u_c.items():
                    li = loc // 2
                    lj = loc % 2
                    for br, bc, bc_coeff in A_blocks:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bc_coeff
                        
                for loc, coeff in v_c.items():
                    lj = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in B_blocks:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bc_coeff
                        
                for loc, coeff in w_c.items():
                    li = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in C_blocks:
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * bc_coeff
                        
                add_term(u_vec, v_vec, w_vec)
                
        add_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        add_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        add_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        add_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
        add_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        add_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        add_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])

    # Part 1: A * B1 (cols 0-3)
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (cols 4-7)
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (col 8)
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    rank = len(U_cols)
    
    # Convert to matrices
    def to_matrices(U_cols, V_cols, W_cols, rank):
        U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
        V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
        W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
        return U, V, W

    U, V, W = to_matrices(U_cols, V_cols, W_cols, rank)
    
    # Verification function
    def verify_decomposition(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0
                    for c in range(rank):
                        s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                    if s != 1:
                        return False
        # Check zeros
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    if j != jp:
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = jp * p + k
                            w_idx = k * n + i
                            s = 0
                            for c in range(rank):
                                s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                            if s != 0:
                                return False
        return True

    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook if Strassen failed (should not happen)
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Attempt to improve rank using linear system search
    # We try to find a support of size R < 114 that works.
    # We use the Strassen support as a seed and try to remove terms.
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Define the linear system for a given support
    # Variables: coefficients for each rank-1 term in the support.
    # Equations: For each (i,j,k) in the tensor, sum(c * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]) = T[i,j,k]
    # This is non-linear in the coefficients if we don't fix the supports.
    # BUT, if we fix the *supports* (which indices are non-zero in U, V, W for each term),
    # the equation for T[i,j,k] is linear in the coefficients?
    # No, T[i,j,k] = sum_c coeff_c * U_sup[c][i*m+j] * V_sup[c][j*p+k] * W_sup[c][k*n+i]
    # If U_sup, V_sup, W_sup are fixed (0/1 masks), then the term is coeff_c * (product of masks).
    # So yes, it is linear in coeff_c.
    
    # However, the masks themselves are part of the search.
    # We will fix the masks to be subsets of the Strassen masks.
    
    strassen_masks_U = [tuple(1 if x != 0 else 0 for x in u) for u in U_cols]
    strassen_masks_V = [tuple(1 if x != 0 else 0 for x in v) for v in V_cols]
    strassen_masks_W = [tuple(1 if x != 0 else 0 for x in w) for w in W_cols]
    
    # Precompute the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
                
    # Function to check if a set of masks (indices) has a solution
    def check_support(indices):
        R = len(indices)
        if R == 0: return None
        # Build the linear system A x = b
        # A is (n*m*m*p*p*n) x R ? No, the number of equations is n*m*m*p*p*n?
        # No, the tensor has n*m*m*p*p*n entries?
        # T[u,v,w] where u in 0..15, v in 0..35, w in 0..35.
        # Total equations: 16 * 36 * 36 = 20736.
        # We only need to satisfy the non-zero entries of T (which are 144) and the zero entries (rest).
        # Actually, we need to satisfy ALL entries to ensure exactness.
        
        # However, 20736 equations is large. We can sample or use all.
        # Let's use all non-zero entries of T and a sample of zero entries to keep it fast?
        # No, we need exactness. We must check all.
        # But 20736 x R system is solvable quickly if R is small.
        
        # We construct the matrix A.
        # A[eq, col] = U_mask[col][u] * V_mask[col][v] * W_mask[col][w]
        # where (u,v,w) corresponds to eq.
        
        # To speed up, we only iterate over non-zero T entries for the 'b' vector,
        # but we must ensure the zero entries are also 0.
        # We can solve for the non-zero entries first, then check the zero entries.
        
        eqs = []
        b = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    eqs.append((u,v,w))
                    b.append(1)
        
        # Add some zero equations to constrain the solution
        zero_eqs = []
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    if j != jp:
                        for k in range(p):
                            u = i*m+j
                            v = jp*p+k
                            w = k*n+i
                            zero_eqs.append((u,v,w))
        
        # Combine
        all_eqs = eqs + zero_eqs
        b_full = [1 if e in eqs else 0 for e in all_eqs]
        
        # Build A
        # A is len(all_eqs) x R
        A = np.zeros((len(all_eqs), R))
        for col, idx in enumerate(indices):
            um = strassen_masks_U[idx]
            vm = strassen_masks_V[idx]
            wm = strassen_masks_W[idx]
            for row, (u,v,w) in enumerate(all_eqs):
                val = um[u] * vm[v] * wm[w]
                if val != 0:
                    A[row, col] = val
        
        # Solve A x = b_full
        try:
            x, residuals, rank_A, s = np.linalg.lstsq(A, np.array(b_full), rcond=None)
        except:
            return None
            
        # Check residual
        if residuals.size > 0 and residuals[0] > 1e-9:
            return None
            
        # Check if solution is "nice" (integers/half-integers)
        # We can check if x is close to 0.5 * integer
        x_nice = np.round(x * 2) / 2
        if np.allclose(x, x_nice, atol=1e-6):
            # Verify with exact arithmetic on the nice solution
            # Construct U, V, W with x_nice coefficients
            U_new = [[0]*R for _ in range(16)]
            V_new = [[0]*R for _ in range(36)]
            W_new = [[0]*R for _ in range(36)]
            
            for col, idx in enumerate(indices):
                coeff = x_nice[col]
                if coeff == 0: continue
                # Get original vectors from U_cols etc
                # Note: U_cols[idx] is the vector for that term
                # We need to scale it by coeff
                # But wait, the masks were derived from the Strassen vectors which had coeffs 1/-1.
                # So U_cols[idx] is the actual vector.
                # We just need to scale the column.
                for r in range(16):
                    U_new[r][col] = U_cols[idx][r] * coeff
                for r in range(36):
                    V_new[r][col] = V_cols[idx][r] * coeff
                for r in range(36):
                    W_new[r][col] = W_cols[idx][r] * coeff
            
            # Final verification
            if verify_decomposition(U_new, V_new, W_new):
                return (U_new, V_new, W_new)
                
        return None

    # Search loop
    # Try to remove 1, 2, 3... terms from the Strassen support
    current_indices = list(range(rank))
    
    # Heuristic: Try removing terms from the "schoolbook" part first?
    # Or random removal.
    
    # To keep it fast, we limit the number of trials.
    trials = 0
    max_trials = 500
    
    while time.time() < deadline - 10 and trials < max_trials:
        trials += 1
        # Try to reduce rank by 1
        # Pick a random term to remove
        if len(current_indices) <= 100: break # Don't go too low
        
        idx_to_remove = current_indices[np.random.randint(len(current_indices))]
        new_indices = [i for i in current_indices if i != idx_to_remove]
        
        res = check_support(new_indices)
        if res:
            U, V, W = res
            best_rank = len(new_indices)
            current_indices = new_indices
            trials = 0 # Reset trials on success? Or continue?
            # Continue searching for even lower rank
        else:
            # If failed, try another random removal
            pass
            
    return {"U": U, "V": V, "W": W}
