import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Indices for non-zero entries of T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # u in [0, nm), v in [0, mp), w in [0, pn)
    
    # Precompute the list of (u, v, w) indices for non-zero entries
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                nz_indices.append((u, v, w))
    
    # Total number of entries in the tensor
    total_entries = nm * mp * pn
    
    # Target tensor values for non-zero entries are 1.
    # For all other entries, they are 0.
    
    # We will try to find a decomposition of rank r.
    # Start with r = 119 (one less than schoolbook 120).
    
    def snap_to_half_int(arr):
        # Round to nearest 0.5
        return np.round(arr * 2) / 2.0

    def verify(U, V, W, r):
        # Check if U, V, W exactly decompose T
        # We only need to check non-zero entries of T and a sample of zeros?
        # No, must be exact.
        # Reconstructing the full tensor is expensive.
        # But we can check the non-zero entries and ensure the zero entries are 0.
        # Actually, if the non-zero entries match and the rank is low, it's likely correct,
        # but we must be sure.
        
        # Efficient check:
        # Compute the sum for each non-zero index.
        # And compute the sum for a set of zero indices.
        # If any non-zero index != 1 or any zero index != 0, it fails.
        
        # Let's compute the full contraction for the non-zero indices.
        # U: (nm, r), V: (mp, r), W: (pn, r)
        
        # We can compute the tensor T_recon as a sparse set or just check indices.
        # For each (u, v, w) in nz_indices:
        #   sum(U[u, :] * V[v, :] * W[w, :]) should be 1.
        
        # For zero indices, we need to ensure they are 0.
        # We can pick a random subset of zero indices to check during search,
        # but for final verification, we should check all?
        # 14400 checks * r ops is fast.
        
        # Let's check all non-zero indices.
        for u, v, w in nz_indices:
            val = np.dot(U[u, :], V[v, :] * W[w, :])
            if not np.isclose(val, 1.0, atol=1e-9):
                return False
        
        # Check a sample of zero indices to ensure they are 0.
        # If the rank is low, the zero entries are likely 0 if the non-zero ones are 1.
        # But to be safe, let's check all zero indices?
        # 14400 - 120 = 14280 indices.
        # 14280 * 119 * 3 ops ~ 5 million ops. Fast.
        
        # Generate all indices
        all_u = np.arange(nm)
        all_v = np.arange(mp)
        all_w = np.arange(pn)
        
        # This loop is slow in Python. Vectorize?
        # We can compute the full tensor T_recon using broadcasting?
        # T_recon[u, v, w] = sum_c U[u,c] V[v,c] W[w,c]
        # This is a tensor contraction.
        # T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
        # This creates a (20, 30, 24) array. 14400 elements.
        # This is very fast in NumPy.
        
        T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
        
        # Check non-zero entries
        for u, v, w in nz_indices:
            if not np.isclose(T_recon[u, v, w], 1.0, atol=1e-9):
                return False
        
        # Check zero entries
        # Create a mask of non-zero entries
        mask = np.zeros((nm, mp, pn), dtype=bool)
        for u, v, w in nz_indices:
            mask[u, v, w] = True
            
        # Check if all entries where mask is False are 0
        if not np.allclose(T_recon[~mask], 0.0, atol=1e-9):
            return False
            
        return True

    def linearize_and_update(U, V, W, r, T_target):
        # T_target is the full tensor (nm, mp, pn)
        # Current error E = T_target - T_current
        T_current = np.einsum('uc,vc,wc->uvw', U, V, W)
        E = T_target - T_current
        
        # We want to find dU, dV, dW such that:
        # sum_c (dU[u,c] V[v,c] W[w,c] + U[u,c] dV[v,c] W[w,c] + U[u,c] V[v,c] dW[w,c]) approx -E[u,v,w]
        
        # This is a linear system.
        # Variables: dU (nm*r), dV (mp*r), dW (pn*r)
        # Total vars = r * (nm + mp + pn)
        
        # We can set up the system using the non-zero entries of E?
        # No, E is dense (mostly).
        # We should use all entries or a subset.
        # Using all 14400 entries gives a 14400 x ~8800 system.
        
        # To speed up, we can use a subset of rows (equations).
        # Let's pick the 2000 largest entries of E.
        
        flat_E = E.flatten()
        # Indices of largest 2000 entries
        k = min(2000, len(flat_E))
        idx = np.argsort(np.abs(flat_E))[-k:]
        
        # Construct the linear system A x = b
        # b = -E.flatten()[idx]
        b = -flat_E[idx]
        
        # A has rows corresponding to idx, columns to variables.
        # This construction is slow in Python loops.
        # Alternative: Use a simpler update rule.
        
        # Simple Gradient Descent on the error norm?
        # Error = || T_target - T_current ||_F^2
        # Gradient with respect to U[u,c]:
        # dE/du = -2 * sum_{v,w} E[u,v,w] V[v,c] W[w,c]
        
        # We can compute the gradient efficiently using einsum.
        # grad_U = -2 * np.einsum('uvw,vc,wc->uc', E, V, W)
        # grad_V = -2 * np.einsum('uvw,uc,wc->vc', E, U, W)
        # grad_W = -2 * np.einsum('uvw,uc,vc->wc', E, U, V)
        
        # Then take a step:
        # U -= lr * grad_U
        # V -= lr * grad_V
        # W -= lr * grad_W
        
        # This is a full gradient step.
        # We can use this! It's much faster to compute than building A.
        
        grad_U = -2.0 * np.einsum('uvw,vc,wc->uc', E, V, W)
        grad_V = -2.0 * np.einsum('uvw,uc,wc->vc', E, U, W)
        grad_W = -2.0 * np.einsum('uvw,uc,vc->wc', E, U, V)
        
        return grad_U, grad_V, grad_W

    # Create the target tensor
    T_target = np.zeros((nm, mp, pn), dtype=float)
    for u, v, w in nz_indices:
        T_target[u, v, w] = 1.0
        
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    best_U, best_V, best_W = None, None, None
    best_rank = 120
    
    # Schoolbook solution
    U_school = np.zeros((nm, 120))
    V_school = np.zeros((mp, 120))
    W_school = np.zeros((pn, 120))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1.0
                V_school[v, col] = 1.0
                W_school[w, col] = 1.0
                col += 1
                
    # Try ranks from 119 down to 100
    for r in range(119, 99, -1):
        if time.time() > deadline:
            break
            
        # Initialize U, V, W with first r terms of schoolbook
        U = U_school[:, :r].copy()
        V = V_school[:, :r].copy()
        W = W_school[:, :r].copy()
        
        # Run gradient descent
        lr = 0.01
        for it in range(200):
            if time.time() > deadline:
                break
                
            T_current = np.einsum('uc,vc,wc->uvw', U, V, W)
            E = T_target - T_current
            err = np.linalg.norm(E)
            
            if err < 1e-9:
                # Verify and snap
                U_snap = snap_to_half_int(U)
                V_snap = snap_to_half_int(V)
                W_snap = snap_to_half_int(W)
                if verify(U_snap, V_snap, W_snap, r):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                # If snap fails, maybe try different lr or break
                break
            
            grad_U = -2.0 * np.einsum('uvw,vc,wc->uc', E, V, W)
            grad_V = -2.0 * np.einsum('uvw,uc,wc->vc', E, U, W)
            grad_W = -2.0 * np.einsum('uvw,uc,vc->wc', E, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Snap periodically to keep values close to grid?
            # No, snapping during descent can trap in local minima.
            # But we need to end up on the grid.
            # Let's just descend, then snap at the end.
            
        # After descent, try to snap and verify
        U_snap = snap_to_half_int(U)
        V_snap = snap_to_half_int(V)
        W_snap = snap_to_half_int(W)
        if verify(U_snap, V_snap, W_snap, r):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
    # If no better rank found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
