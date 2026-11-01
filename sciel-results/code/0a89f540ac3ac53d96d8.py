import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = 2, 2, 4
    
    def get_rank14_solution():
        """Return the known rank-14 decomposition."""
        rank = 14
        U = np.zeros((4, rank))
        V = np.zeros((8, rank))
        W = np.zeros((8, rank))
        
        # Left block (k=0,1) - Strassen
        # M1 = (A00+A11)(B00+B11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[5,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # M2 = (A10+A11)B00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[1,1] = 1; W[3,1] = -1
        
        # M3 = A00(B01-B11)
        U[0,2] = 1
        V[1,2] = 1; V[5,2] = -1
        W[2,2] = 1; W[3,2] = 1
        
        # M4 = A11(B10-B00)
        U[3,3] = 1
        V[4,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[1,3] = 1
        
        # M5 = (A00+A01)B11
        U[0,4] = 1; U[1,4] = 1
        V[5,4] = 1
        W[0,4] = -1; W[2,4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1,6] = 1; U[3,6] = -1
        V[4,6] = 1; V[5,6] = 1
        W[0,6] = 1
        
        # Right block (k=2,3) - Strassen
        # N1 = (A00+A11)(B02+B13)
        U[0,7] = 1; U[3,7] = 1
        V[2,7] = 1; V[7,7] = 1
        W[4,7] = 1; W[7,7] = 1
        
        # N2 = (A10+A11)B02
        U[2,8] = 1; U[3,8] = 1
        V[2,8] = 1
        W[5,8] = 1; W[7,8] = -1
        
        # N3 = A00(B03-B13)
        U[0,9] = 1
        V[3,9] = 1; V[7,9] = -1
        W[6,9] = 1; W[7,9] = 1
        
        # N4 = A11(B12-B02)
        U[3,10] = 1
        V[6,10] = 1; V[2,10] = -1
        W[4,10] = 1; W[5,10] = 1
        
        # N5 = (A00+A01)B13
        U[0,11] = 1; U[1,11] = 1
        V[7,11] = 1
        W[4,11] = -1; W[6,11] = 1
        
        # N6 = (A10-A00)(B02+B03)
        U[2,12] = 1; U[0,12] = -1
        V[2,12] = 1; V[3,12] = 1
        W[7,12] = 1
        
        # N7 = (A01-A11)(B12+B13)
        U[1,13] = 1; U[3,13] = -1
        V[6,13] = 1; V[7,13] = 1
        W[4,13] = 1
        
        return U, V, W
    
    def verify(U, V, W, tol=1e-9):
        """Verify the decomposition is exact."""
        rank = U.shape[1]
        # For each (i,j,k) triple, compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # and check it equals 1
        u_idx = []
        v_idx = []
        w_idx = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx.append(i*m+j)
                    v_idx.append(j*p+k)
                    w_idx.append(k*n+i)
        u_idx = np.array(u_idx)
        v_idx = np.array(v_idx)
        w_idx = np.array(w_idx)
        
        U_sel = U[u_idx, :]
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        prod = U_sel * V_sel * W_sel
        vals = prod.sum(axis=1)
        return bool(np.all(np.abs(vals - 1.0) <= tol))
    
    def residual_sq_vec(U, V, W, u_idx, v_idx, w_idx):
        """Vectorized residual computation."""
        U_sel = U[u_idx, :]
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        prod = U_sel * V_sel * W_sel
        vals = prod.sum(axis=1)
        return float(np.sum((vals - 1.0)**2))
    
    def local_search(U, V, W, u_idx, v_idx, w_idx, max_steps=100):
        """Perform local search with integer/half-integer entries."""
        rank = U.shape[1]
        current_val = residual_sq_vec(U, V, W, u_idx, v_idx, w_idx)
        
        if current_val < 1e-9:
            return U, V, W, current_val
        
        steps = 0
        no_improve = 0
        while steps < max_steps and time.time() < deadline:
            steps += 1
            # Pick a random matrix and entry
            mat_idx = np.random.randint(0, 3)
            if mat_idx == 0:
                row = np.random.randint(0, 4)
                col = np.random.randint(0, rank)
                old_val = U[row, col]
                candidates = []
                if old_val != 0:
                    candidates.append(-old_val)
                candidates.extend([0, 1, -1, 0.5, -0.5])
                candidates = [c for c in candidates if c != old_val]
                if not candidates:
                    candidates = [1.0]
                new_val = candidates[np.random.randint(0, len(candidates))]
                U[row, col] = new_val
                new_val_res = residual_sq_vec(U, V, W, u_idx, v_idx, w_idx)
                if new_val_res < current_val - 1e-12:
                    current_val = new_val_res
                    no_improve = 0
                else:
                    U[row, col] = old_val
                    no_improve += 1
            elif mat_idx == 1:
                row = np.random.randint(0, 8)
                col = np.random.randint(0, rank)
                old_val = V[row, col]
                candidates = []
                if old_val != 0:
                    candidates.append(-old_val)
                candidates.extend([0, 1, -1, 0.5, -0.5])
                candidates = [c for c in candidates if c != old_val]
                if not candidates:
                    candidates = [1.0]
                new_val = candidates[np.random.randint(0, len(candidates))]
                V[row, col] = new_val
                new_val_res = residual_sq_vec(U, V, W, u_idx, v_idx, w_idx)
                if new_val_res < current_val - 1e-12:
                    current_val = new_val_res
                    no_improve = 0
                else:
                    V[row, col] = old_val
                    no_improve += 1
            else:
                row = np.random.randint(0, 8)
                col = np.random.randint(0, rank)
                old_val = W[row, col]
                candidates = []
                if old_val != 0:
                    candidates.append(-old_val)
                candidates.extend([0, 1, -1, 0.5, -0.5])
                candidates = [c for c in candidates if c != old_val]
                if not candidates:
                    candidates = [1.0]
                new_val = candidates[np.random.randint(0, len(candidates))]
                W[row, col] = new_val
                new_val_res = residual_sq_vec(U, V, W, u_idx, v_idx, w_idx)
                if new_val_res < current_val - 1e-12:
                    current_val = new_val_res
                    no_improve = 0
                else:
                    W[row, col] = old_val
                    no_improve += 1
            
            if no_improve > 30:
                break
        
        return U, V, W, current_val
    
    # Precompute index arrays
    u_idx = []
    v_idx = []
    w_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx.append(i*m+j)
                v_idx.append(j*p+k)
                w_idx.append(k*n+i)
    u_idx = np.array(u_idx)
    v_idx = np.array(v_idx)
    w_idx = np.array(w_idx)
    
    # Try rank 13 from rank-14 base
    def try_rank13_from_rank14():
        """Start from rank-14 solution, try to merge two columns into one for rank 13."""
        rank13 = 13
        best_val = float('inf')
        best_result = None
        
        U14, V14, W14 = get_rank14_solution()
        
        # Strategy: try to express one column as a combination of others
        # or start from rank-14 with one column zeroed and adjust
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Start with rank-14, pick a random column to "remove"
            # and try to distribute its contributions
            U = U14[:, :rank13].copy()
            V = V14[:, :rank13].copy()
            W = W14[:, :rank13].copy()
            
            # Also try: start with rank-14, zero out one column, then search
            if np.random.random() < 0.5:
                U = U14.copy()
                V = V14.copy()
                W = W14.copy()
                # Zero out a random column
                col_to_zero = np.random.randint(0, 14)
                U[:, col_to_zero] = 0
                V[:, col_to_zero] = 0
                W[:, col_to_zero] = 0
                # Remove that column and add a new random one
                remaining = [c for c in range(14) if c != col_to_zero]
                U = np.column_stack([U[:, remaining], np.random.randint(-1, 2, (4, 1))])
                V = np.column_stack([V[:, remaining], np.random.randint(-1, 2, (8, 1))])
                W = np.column_stack([W[:, remaining], np.random.randint(-1, 2, (8, 1))])
            
            U, V, W, val = local_search(U, V, W, u_idx, v_idx, w_idx, max_steps=80)
            
            if val < best_val:
                best_val = val
                best_result = (U.copy(), V.copy(), W.copy())
                
            if best_val < 1e-9:
                if verify(best_result[0], best_result[1], best_result[2]):
                    return best_result
        
        return best_result
    
    # Try rank 12
    def try_rank12():
        """Try to find a rank-12 solution."""
        rank12 = 12
        best_val = float('inf')
        best_result = None
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-1, 2, (4, rank12))
            V = np.random.randint(-1, 2, (8, rank12))
            W = np.random.randint(-1, 2, (8, rank12))
            
            U, V, W, val = local_search(U, V, W, u_idx, v_idx, w_idx, max_steps=50)
            
            if val < best_val:
                best_val = val
                best_result = (U.copy(), V.copy(), W.copy())
                
            if best_val < 1e-9:
                if verify(best_result[0], best_result[1], best_result[2]):
                    return best_result
        
        return best_result
    
    # Try rank 12 first (theoretical minimum)
    result = try_rank12()
    if result is not None:
        U, V, W = result
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 13 from rank-14 base
    result = try_rank13_from_rank14()
    if result is not None:
        U, V, W = result
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank 14 (known valid solution)
    U, V, W = get_rank14_solution()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
