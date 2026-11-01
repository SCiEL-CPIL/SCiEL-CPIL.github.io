"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Build rank-21 Strassen decomposition
    rank21 = 21
    U21 = np.zeros((n * m, rank21))
    V21 = np.zeros((m * p, rank21))
    W21 = np.zeros((p * n, rank21))
    
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
        [(0, 0, 1)],                 # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
        [(1, 1, 1)],                 # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
    ]
    
    W_map = [
        [(0, 0, 1), (1, 1, 1)],   # k=0 (M1) -> C00, C11
        [(1, 0, 1), (1, 1, -1)],  # k=1 (M2) -> C10, C11
        [(0, 1, 1), (1, 1, 1)],   # k=2 (M3) -> C01, C11
        [(0, 0, 1), (1, 0, 1)],   # k=3 (M4) -> C00, C10
        [(0, 0, -1), (0, 1, 1)],  # k=4 (M5) -> C00, C01
        [(1, 1, 1)],              # k=5 (M6) -> C11
        [(0, 0, 1)]               # k=6 (M7) -> C00
    ]
    
    for r in range(3):
        for k in range(7):
            col = r * 7 + k
            for idx, val in U_coeffs[k].items():
                U21[idx, col] = val
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                v_idx = j_local * p + global_k
                V21[v_idx, col] = val
            for i, j_local, val in W_map[k]:
                global_k = 2 * r + j_local
                w_idx = global_k * n + i
                W21[w_idx, col] = val
    
    # Build the full matrix multiplication tensor T
    # T[(i,j),(j,k),(i,k)] = 1 for all valid (i,j,k)
    # Flatten: T_flat[idx] where idx = ((i*m + j)*p + k) gives 0..23
    T = np.ones(n * m * p)
    
    # Compute the tensor contribution of each rank-1 term
    # Term r contributes: U21[:, r] ⊗ V21[:, r] ⊗ W21[:, r]
    # The (i,j,k) entry of this tensor is U21[i*m+j, r] * V21[j*p+k, r] * W21[k*n+i, r]
    
    def tensor_entry(U_col, V_col, W_col, i, j, k):
        return U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
    
    # Linear algebra approach: model as a linear code over F_2
    # The rank-21 decomposition gives us 21 rank-1 tensors that sum to T
    # We want to find if there's a subset of 20 that still sums to T
    # (i.e., one of the rank-1 terms is redundant)
    
    # For each pair of columns to remove, check if the residual has rank 1
    # If so, we can replace those 2 columns with 1 new column
    
    def has_rank_1(X_flat):
        """Check if a tensor with entries X_flat[((i*m+j)*p+k)] has tensor rank 1."""
        # Find first non-zero entry
        nonzero_idx = None
        for idx in range(len(X_flat)):
            if abs(X_flat[idx]) > 1e-10:
                nonzero_idx = idx
                break
        
        if nonzero_idx is None:
            return None  # Zero tensor, not rank 1
        
        # Decompose idx into (i, j, k)
        i = (nonzero_idx // p) // m
        j = (nonzero_idx // p) % m
        k = nonzero_idx % p
        
        # For rank 1: X[i',j',k'] = X[i,j,k] * (u[i',j']/u[i,j]) * (v[j',k']/v[j,k]) * (w[k',i']/w[k,i])
        # Check all quadruples for consistency
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                lhs = X_flat[(i*m+j)*p+k] * X_flat[(i2*m+j2)*p+k2]
                                rhs = X_flat[(i2*m+j)*p+k] * X_flat[(i*m+j2)*p+k2]
                                if abs(lhs - rhs) > 1e-8:
                                    return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for j2 in range(m):
                        for k2 in range(p):
                            lhs = X_flat[(i*m+j)*p+k] * X_flat[(i*m+j2)*p+k2]
                            rhs = X_flat[(i*m+j2)*p+k] * X_flat[(i*m+j)*p+k2]
                            if abs(lhs - rhs) > 1e-8:
                                return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            lhs = X_flat[(i*m+j)*p+k] * X_flat[(i2*m+j)*p+k2]
                            rhs = X_flat[(i2*m+j)*p+k] * X_flat[(i*m+j)*p+k2]
                            if abs(lhs - rhs) > 1e-8:
                                return False
        
        return True
    
    # Search for rank-20: try removing 2 columns and replacing with 1
    # This is equivalent to finding a linear dependency in the code
    
    # First, check if any single column removal gives rank-1 residual
    for r_remove in range(rank21):
        if time.time() > deadline - 20:
            break
        
        # Compute residual tensor: T - sum of all terms except r_remove
        X_flat = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    total = 0
                    for r in range(rank21):
                        if r == r_remove:
                            continue
                        total += U21[i*m+j, r] * V21[j*p+k, r] * W21[k*n+i, r]
                    X_flat[idx] = T[idx] - total
        
        # Check if X_flat has rank 1
        if has_rank_1(X_flat):
            # Extract factors
            anchor_idx = None
            for idx in range(len(X_flat)):
                if abs(X_flat[idx]) > 1e-10:
                    anchor_idx = idx
                    break
            
            if anchor_idx is None:
                continue
            
            i_a = (anchor_idx // p) // m
            j_a = (anchor_idx // p) % m
            k_a = anchor_idx % p
            
            U_new = np.zeros(n * m)
            V_new = np.zeros(m * p)
            W_new = np.zeros(p * n)
            
            # Try gauge: W[k*n+i] = 1 for all (k,i)
            valid = True
            for j in range(m):
                M = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        M[i, k] = X_flat[(i*m+j)*p+k]
                
                if n > 0 and p > 0:
                    if abs(M[0,0]) > 1e-10:
                        for i in range(n):
                            for k in range(p):
                                if abs(M[i,k] - M[i,0]*M[0,k]/M[0,0]) > 1e-8:
                                    valid = False
                                    break
                            if not valid:
                                break
                    else:
                        row0_all_zero = all(abs(M[0,k]) < 1e-10 for k in range(p))
                        col0_all_zero = all(abs(M[i,0]) < 1e-10 for i in range(n))
                        if not (row0_all_zero or col0_all_zero):
                            valid = False
                        else:
                            for i in range(n):
                                for k in range(p):
                                    if not row0_all_zero and not col0_all_zero:
                                        pass
                                    elif row0_all_zero:
                                        if abs(M[i,k]) > 1e-10 and abs(M[i,0]) < 1e-10:
                                            valid = False
                                    elif col0_all_zero:
                                        if abs(M[i,k]) > 1e-10 and abs(M[0,k]) < 1e-10:
                                            valid = False
                            if not valid:
                                break
                if not valid:
                    break
            
            if valid:
                # Extract factors with gauge W[k*n+i] = 1
                # Then U[i*m+j] * V[j*p+k] = X_flat[(i*m+j)*p+k]
                # Set U[i*m+j] = X_flat[(i*m+j)*p+0] for k=0, then V[j*p+0] = 1
                # For k>0: V[j*p+k] = X_flat[(i*m+j)*p+k] / U[i*m+j]
                
                # This is still coupled. Let's use a different approach.
                # Set V[j*p+k] = X_flat[(i0*m+j)*p+k] for fixed i0, then
                # U[i*m+j] = X_flat[(i*m+j)*p+0] / V[j*p+0]
                
                i0 = 0
                for j in range(m):
                    for k in range(p):
                        V_new[j*p+k] = X_flat[(i0*m+j)*p+k]
                    
                    # Now find U[i*m+j] for each i
                    for i in range(n):
                        # U[i*m+j] * V[j*p+k] = X_flat[(i*m+j)*p+k] for all k
                        # Find k where V[j*p+k] != 0
                        found = False
                        for k in range(p):
                            if abs(V_new[j*p+k]) > 1e-10:
                                U_new[i*m+j] = X_flat[(i*m+j)*p+k] / V_new[j*p+k]
                                found = True
                                break
                        if not found:
                            U_new[i*m+j] = 0
                        # Verify consistency
                        for k in range(p):
                            if abs(V_new[j*p+k]) > 1e-10:
                                if abs(U_new[i*m+j] * V_new[j*p+k] - X_flat[(i*m+j)*p+k]) > 1e-8:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Set W[k*n+i] = 1 for all (k,i)
                    W_new[:] = 1.0
                    
                    # Verify the full decomposition
                    U_final = np.zeros((n * m, 20))
                    V_final = np.zeros((m * p, 20))
                    W_final = np.zeros((p * n, 20))
                    
                    # Copy all columns except r_remove
                    col_idx = 0
                    for r in range(rank21):
                        if r == r_remove:
                            continue
                        U_final[:, col_idx] = U21[:, r]
                        V_final[:, col_idx] = V21[:, r]
                        W_final[:, col_idx] = W21[:, r]
                        col_idx += 1
                    
                    # Add the new column
                    U_final[:, 19] = U_new
                    V_final[:, 19] = V_new
                    W_final[:, 19] = W_new
                    
                    # Verify
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = 0
                                for r in range(20):
                                    total += U_final[i*m+j, r] * V_final[j*p+k, r] * W_final[k*n+i, r]
                                if abs(total - T[(i*m+j)*p+k]) > 1e-8:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    # If no rank-20 solution found, return rank 21
    U = U21.tolist()
    V = V21.tolist()
    W = W21.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
