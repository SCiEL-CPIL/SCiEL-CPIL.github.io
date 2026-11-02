import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 2, 2, 12
    
    # Target tensor: T[a,b,c] = 1 if a=(i,k), b=(k,j), c=(i,j)
    # Build the list of non-zero entries
    # a = i*m+k, b = k*p+j, c = j*n+i
    # For each (i,k,j) with i in {0,1}, k in {0,1}, j in {0..11}
    
    # Number of constraints: n*m*p = 2*2*12 = 48 non-zero entries
    # Total possible (a,b,c) triples: 4*24*24 = 2304
    # But we only need to satisfy the 48 non-zero ones and the rest are automatically 0
    # if the structure is correct... actually no, we need all 2304 to be satisfied.
    
    # Let's use the constraint: for all a,b,c: sum_r U[a,r]V[b,r]W[c,r] = T[a,b,c]
    # This is equivalent to: for all c: (U ⊙ V)[a,b,c] = T[a,b,c] where ⊙ is Khatri-Rao
    
    # Alternative: fix W, then for each c, we need U ⊙ V to match T[:,:,c]
    # U ⊙ V is a 4x24 matrix (Khatri-Rao product of U (4xr) and V (24xr))
    # Actually, U ⊙ V has entries (U⊙V)[a,b] = sum_r U[a,r]V[b,r]
    # And we need (U⊙V)[a,b] * W[c,r] summed over r... no that's not right.
    
    # Let me think differently. The constraint is:
    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # For fixed c: sum_r (U[:,r] ⊗ V[:,r])_flattened * W[c,r] = T[:,:,c]_flattened
    # This is a linear system in W[c,:] given U and V.
    
    # So: let M_c be the matrix where M_c[:,r] = (U[:,r] ⊗ V[:,r])_flattened (size 4*24=96)
    # Then M_c @ W[c,:] = T[:,:,c]_flattened (size 96)
    # Similarly for fixing U or V.
    
    # Let's try ALS:
    # 1. Initialize U, V, W randomly
    # 2. Fix V, W: solve for U. For each a: sum_r U[a,r] (V[:,r] ⊗ W[:,r])_flattened = T[a,:,:]_flattened
    #    This is: (V⊙W) @ U[a,:] = T[a,:,:]_flattened
    #    Where (V⊙W) is 24*24 x r = 576 x r
    # 3. Similarly for V and W.
    
    # Let's implement this.
    
    def build_target_tensor():
        T = np.zeros((4, 24, 24))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    T[a, b, c] = 1.0
        return T
    
    T = build_target_tensor()
    
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    val = 0.0
                    for rr in range(r):
                        val += U[a, rr] * V[b, rr] * W[c, rr]
                    if abs(val - 1) > 1e-9:
                        return False
        # Check zeros
        for a in range(4):
            for b in range(24):
                for c in range(24):
                    if T[a,b,c] == 0:
                        val = 0.0
                        for rr in range(r):
                            val += U[a, rr] * V[b, rr] * W[c, rr]
                        if abs(val) > 1e-9:
                            return False
        return True
    
    # Try ALS for various ranks
    best_result = None
    
    for target_rank in [24, 28, 30, 32, 35, 38, 40, 42]:
        if time.time() > deadline - 30:
            break
        
        for trial in range(10):
            if time.time() > deadline - 30:
                break
            
            r = target_rank
            np.random.seed(42 + target_rank * 100 + trial)
            U = np.random.randn(4, r)
            V = np.random.randn(24, r)
            W = np.random.randn(24, r)
            
            # ALS iterations
            for it in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Solve for U: for each a, (V⊙W) @ U[a,:] = T[a,:,:].flatten()
                # V⊙W: Khatri-Rao product, shape (24*24, r)
                # Actually, we need: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                # For fixed a: sum_r U[a,r] (V[b,r] W[c,r]) = T[a,b,c]
                # Let M[a] be a matrix where M[a][:,r] = (V[:,r] * W[:,r])_flattened (size 24*24=576)
                # Then M @ U[a,:] = T[a,:,:].flatten()
                
                # Build M: shape (576, r)
                M = np.zeros((24*24, r))
                for rr in range(r):
                    M[:, rr] = (V[:, rr] * W[:, rr]).flatten()
                
                # Solve for each a
                for a in range(4):
                    rhs = T[a, :, :].flatten()
                    U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                
                # Solve for V: for each b, (U⊙W) @ V[b,:] = T[:,b,:].flatten()
                M = np.zeros((4*24, r))
                for rr in range(r):
                    M[:, rr] = (U[:, rr] * W[:, rr]).flatten()
                
                for b in range(24):
                    rhs = T[:, b, :].flatten()
                    V[b, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                
                # Solve for W: for each c, (U⊙V) @ W[c,:] = T[:,:,c].flatten()
                M = np.zeros((4*24, r))
                for rr in range(r):
                    M[:, rr] = (U[:, rr] * V[:, rr]).flatten()
                
                for c in range(24):
                    rhs = T[:, :, c].flatten()
                    W[c, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            
            # Verify
            if verify(U, V, W):
                if best_result is None or r < best_result[0]:
                    best_result = (r, U, V, W)
                    break  # Found a solution at this rank, try lower ranks
            
            # Check error
            max_err = 0
            for a in range(4):
                for b in range(24):
                    for c in range(24):
                        val = 0
                        for rr in range(r):
                            val += U[a, rr] * V[b, rr] * W[c, rr]
                        if T[a,b,c] == 1:
                            max_err = max(max_err, abs(val - 1))
                        else:
                            max_err = max(max_err, abs(val))
            
            if max_err < 1e-6:
                # Close to solution, try to polish
                for polish_it in range(100):
                    if time.time() > deadline - 30:
                        break
                    for a in range(4):
                        M = np.zeros((24*24, r))
                        for rr in range(r):
                            M[:, rr] = (V[:, rr] * W[:, rr]).flatten()
                        U[a, :] = np.linalg.lstsq(M, T[a,:,:].flatten(), rcond=None)[0]
                    for b in range(24):
                        M = np.zeros((4*24, r))
                        for rr in range(r):
                            M[:, rr] = (U[:, rr] * W[:, rr]).flatten()
                        V[b, :] = np.linalg.lstsq(M, T[:,b,:].flatten(), rcond=None)[0]
                    for c in range(24):
                        M = np.zeros((4*24, r))
                        for rr in range(r):
                            M[:, rr] = (U[:, rr] * V[:, rr]).flatten()
                        W[c, :] = np.linalg.lstsq(M, T[:,:,c].flatten(), rcond=None)[0]
                
                if verify(U, V, W):
                    if best_result is None or r < best_result[0]:
                        best_result = (r, U, V, W)
                    break
    
    if best_result is not None:
        r, U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 42
    rank_42 = 42
    U42 = np.zeros((n * m, rank_42), dtype=int)
    V42 = np.zeros((m * p, rank_42), dtype=int)
    W42 = np.zeros((p * n, rank_42), dtype=int)
    
    U_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    for block_idx in range(6):
        b00 = 2 * block_idx
        b01 = 2 * block_idx + 1
        b10 = 12 + 2 * block_idx
        b11 = 12 + 2 * block_idx + 1
        c00 = 4 * block_idx
        c01 = 4 * block_idx + 2
        c10 = 4 * block_idx + 1
        c11 = 4 * block_idx + 3
        
        for t in range(7):
            col = block_idx * 7 + t
            for i in range(4):
                U42[i, col] = U_forms[t][i]
            
            if t == 0:
                V42[b00, col] += 1; V42[b11, col] += 1
                W42[c00, col] += 1; W42[c11, col] += 1
            elif t == 1:
                V42[b00, col] += 1
                W42[c10, col] += 1; W42[c11, col] -= 1
            elif t == 2:
                V42[b01, col] += 1; V42[b11, col] -= 1
                W42[c01, col] += 1; W42[c11, col] += 1
            elif t == 3:
                V42[b10, col] += 1; V42[b00, col] -= 1
                W42[c00, col] += 1; W42[c10, col] += 1
            elif t == 4:
                V42[b11, col] += 1
                W42[c00, col] -= 1; W42[c01, col] += 1
            elif t == 5:
                V42[b00, col] += 1; V42[b01, col] += 1
                W42[c11, col] += 1
            elif t == 6:
                V42[b10, col] += 1; V42[b11, col] += 1
                W42[c00, col] += 1
    
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
