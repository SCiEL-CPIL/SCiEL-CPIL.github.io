import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor non-zero entries
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                entries.append((u, v, w))
    
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((U_dim, rank_school))
    V_school = np.zeros((V_dim, rank_school))
    W_school = np.zeros((W_dim, rank_school))
    for idx, (u, v, w) in enumerate(entries):
        U_school[u, idx] = 1
        V_school[v, idx] = 1
        W_school[w, idx] = 1
        
    def to_list(mat):
        return [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]
        
    # Verify function
    def verify(U, V, W):
        # Check if U, V, W produce the exact tensor
        # Reconstruct tensor from factors
        # T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
        # Check all non-zero entries are 1
        # Check all zero entries are 0? 
        # The tensor is defined to be 0 elsewhere.
        # We must ensure no "ghost" non-zeros.
        # Since the tensor is sparse, we check:
        # 1. For each (u,v,w) in entries, sum is 1.
        # 2. For a sample of (u,v,w) NOT in entries, sum is 0.
        # To be safe, we can check all possible (u,v,w) if the space is small.
        # U_dim * V_dim * W_dim = 9 * 15 * 15 = 2025. This is small.
        # We can just reconstruct the whole tensor and compare.
        
        rank = U.shape[1]
        # Efficient reconstruction using einsum or broadcasting
        # T = np.einsum('ur,vr,wr->uvw', U, V, W)
        # This creates a 9x15x15 array.
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Check entries
        for u, v, w in entries:
            if not np.isclose(T_recon[u, v, w], 1.0, atol=1e-9):
                return False
        
        # Check zeros
        # Create a mask of expected non-zeros
        mask = np.zeros((U_dim, V_dim, W_dim), dtype=bool)
        for u, v, w in entries:
            mask[u, v, w] = True
            
        # Check that all non-masked entries are close to 0
        if not np.all(np.isclose(T_recon[~mask], 0.0, atol=1e-9)):
            return False
        return True

    start_time = time.time()
    deadline = start_time + 110 # Leave 10s margin
    
    # Try to find lower rank
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Search ranks from 30 to 44
    for rank in range(30, 45):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(U_dim, rank)
            V = np.random.randn(V_dim, rank)
            W = np.random.randn(W_dim, rank)
            
            # ALS iterations
            for it in range(100):
                # Update U
                for u in range(U_dim):
                    # Find all (v, w) such that T[u,v,w] != 0
                    # From our construction, for a fixed u=i*m+j, 
                    # v = j*p+k, w = k*n+i for k in 0..p-1
                    # So there are p such pairs.
                    vs = []
                    ws = []
                    for k in range(p):
                        v = (u % m) * p + k
                        w = k * n + (u // m)
                        vs.append(v)
                        ws.append(w)
                    
                    # Construct the linear system A x = b
                    # x is U[u, :]
                    # b is T[u, v, w] which is 1 for these entries
                    # A[r, col] = V[v, r] * W[w, r]
                    # Wait, the equation is sum_r U[u,r] V[v,r] W[w,r] = 1
                    # Let x_r = U[u,r].
                    # sum_r x_r (V[v,r] W[w,r]) = 1
                    # For each (v,w) pair, we have one equation.
                    # A is p x rank.
                    A = np.zeros((p, rank))
                    for idx in range(p):
                        A[idx, :] = V[vs[idx], :] * W[ws[idx], :]
                    b = np.ones(p)
                    
                    # Solve A x = b
                    # Use least squares
                    try:
                        U[u, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
                        
                # Update V
                for v in range(V_dim):
                    # v = j*p + k
                    j = v // p
                    k = v % p
                    # Non-zeros for fixed v:
                    # u = i*m + j, w = k*n + i for i in 0..n-1
                    us = []
                    ws = []
                    for i in range(n):
                        u = i * m + j
                        w = k * n + i
                        us.append(u)
                        ws.append(w)
                    
                    A = np.zeros((n, rank))
                    for idx in range(n):
                        A[idx, :] = U[us[idx], :] * W[ws[idx], :]
                    b = np.ones(n)
                    try:
                        V[v, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass

                # Update W
                for w in range(W_dim):
                    # w = k*n + i
                    k = w // n
                    i = w % n
                    # Non-zeros for fixed w:
                    # u = i*m + j, v = j*p + k for j in 0..m-1
                    us = []
                    vs = []
                    for j in range(m):
                        u = i * m + j
                        v = j * p + k
                        us.append(u)
                        vs.append(v)
                    
                    A = np.zeros((m, rank))
                    for idx in range(m):
                        A[idx, :] = U[us[idx], :] * V[vs[idx], :]
                    b = np.ones(m)
                    try:
                        W[w, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
                        
            # Check error
            T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.max(np.abs(T_recon - (np.zeros((U_dim, V_dim, W_dim))))) # Placeholder
            # Better error check:
            err_val = 0
            for u, v, w in entries:
                err_val = max(err_val, abs(T_recon[u, v, w] - 1))
            # Check zeros
            mask = np.zeros((U_dim, V_dim, W_dim), dtype=bool)
            for u, v, w in entries:
                mask[u, v, w] = True
            err_val = max(err_val, np.max(np.abs(T_recon[~mask])))
            
            if err_val < 1e-6:
                # Try rounding
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded):
                    best_rank = rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    found = True
                    break
                    
        if found:
            # Found a valid decomposition for this rank, try lower?
            # We are iterating 30..44, so we found the lowest so far.
            # We can break if we want the absolute lowest, but maybe a lower one exists?
            # No, we start from 30 and go up. So the first found is the lowest.
            break
            
    # If no better found, use schoolbook
    if best_rank == rank_school:
        U_out, V_out, W_out = U_school, V_school, W_school
    else:
        U_out, V_out, W_out = best_U, best_V, best_W
        
    return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
