import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Construct schoolbook solution as fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
                
    best_result = {
        'U': U_sb.tolist(),
        'V': V_sb.tolist(),
        'W': W_sb.tolist()
    }
    best_rank = schoolbook_rank

    # Search for lower rank slice decomposition
    # The tensor is a direct sum of p slices of M_{n,m,1}
    # Slice k has:
    # a = i*m + j
    # b = j (local index in slice, mapped to j*p + k in global)
    # c = i (local index in slice, mapped to k*n + i in global)
    
    # Define the slice tensor T_slice of shape (n*m, m, n)
    # T_slice[a, b_local, c_local] = 1 if a = i*m + j, b_local = j, c_local = i
    # Here b_local is in [0, m), c_local is in [0, n)
    
    diag_slice_indices = []
    for i in range(n):
        for j in range(m):
            a = i * m + j
            b_local = j
            c_local = i
            diag_slice_indices.append((a, b_local, c_local))
            
    all_slice_indices = []
    for a in range(n * m):
        for b_local in range(m):
            for c_local in range(n):
                all_slice_indices.append((a, b_local, c_local))
                
    for idx in all_slice_indices:
        if idx not in diag_slice_indices:
            # zero_indices for slice
            pass

    def verify_slice(U_s, V_s, W_s, tol=1e-8):
        # U_s: (n*m, r)
        # V_s: (m, r)
        # W_s: (n, r)
        r = U_s.shape[1]
        if V_s.shape != (m, r) or W_s.shape != (n, r):
            return False
            
        # Compute tensor R_s of shape (n*m, m, n)
        R_s = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
        
        for a, b, c in diag_slice_indices:
            if abs(R_s[a, b, c] - 1.0) > tol:
                return False
                
        for a in range(n * m):
            for b in range(m):
                for c in range(n):
                    if (a, b, c) not in diag_slice_indices:
                        if abs(R_s[a, b, c]) > tol:
                            return False
        return True

    def build_full_solution(U_s, V_s, W_s):
        r_slice = U_s.shape[1]
        R = p * r_slice
        
        U_full = np.zeros((n * m, R))
        V_full = np.zeros((m * p, R))
        W_full = np.zeros((p * n, R))
        
        for k in range(p):
            for r_idx in range(r_slice):
                col = k * r_slice + r_idx
                
                # U is same for all slices
                U_full[:, col] = U_s[:, r_idx]
                
                # V: place V_s[:, r_idx] at positions j*p + k for j in 0..m-1
                for j in range(m):
                    b_global = j * p + k
                    V_full[b_global, col] = V_s[j, r_idx]
                    
                # W: place W_s[:, r_idx] at positions k*n + i for i in 0..n-1
                for i in range(n):
                    c_global = k * n + i
                    W_full[c_global, col] = W_s[i, r_idx]
                    
        return {
            'U': U_full.tolist(),
            'V': V_full.tolist(),
            'W': W_full.tolist()
        }

    # Try ranks for the slice
    # Known rank of M_{3,3,1} is 9. Try 8 down to 1.
    for r_slice in range(8, 0, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        max_trials = 50
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            U_s = np.random.randn(n * m, r_slice)
            V_s = np.random.randn(m, r_slice)
            W_s = np.random.randn(n, r_slice)
            
            # Optimization
            lr = 0.01
            steps = 500
            for step in range(steps):
                if time.time() > deadline - 5:
                    break
                    
                R_s = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
                
                E = np.zeros_like(R_s)
                for a, b, c in diag_slice_indices:
                    E[a, b, c] = R_s[a, b, c] - 1.0
                for a in range(n * m):
                    for b in range(m):
                        for c in range(n):
                            if (a, b, c) not in diag_slice_indices:
                                E[a, b, c] = R_s[a, b, c]
                                
                loss = np.sum(E**2)
                if loss < 1e-10:
                    break
                    
                grad_U = 2 * np.einsum('abc,br,cr->ar', E, V_s, W_s)
                grad_V = 2 * np.einsum('abc,ar,cr->br', E, U_s, W_s)
                grad_W = 2 * np.einsum('abc,ar,br->cr', E, U_s, V_s)
                
                U_s -= lr * grad_U
                V_s -= lr * grad_V
                W_s -= lr * grad_W
                
            # Check convergence
            R_s = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
            max_err = 0
            for a, b, c in diag_slice_indices:
                max_err = max(max_err, abs(R_s[a, b, c] - 1.0))
            for a in range(n * m):
                for b in range(m):
                    for c in range(n):
                        if (a, b, c) not in diag_slice_indices:
                            max_err = max(max_err, abs(R_s[a, b, c]))
                            
            if max_err < 1e-8:
                # Try rounding to integers/half-integers
                U_r = np.round(U_s * 2) / 2.0
                V_r = np.round(V_s * 2) / 2.0
                W_r = np.round(W_s * 2) / 2.0
                
                if verify_slice(U_r, V_r, W_r):
                    result = build_full_solution(U_r, V_r, W_r)
                    # Verify full solution
                    # (Skipped for speed, assuming slice is correct)
                    best_result = result
                    best_rank = p * r_slice
                    found = True
                    break
                    
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
