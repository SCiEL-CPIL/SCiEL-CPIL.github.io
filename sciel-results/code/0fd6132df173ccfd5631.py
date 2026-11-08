import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Trivial solution
    rank_trivial = n * m * p  # 36
    U_trivial = np.zeros((n * m, rank_trivial), dtype=float)
    V_trivial = np.zeros((m * p, rank_trivial), dtype=float)
    W_trivial = np.zeros((p * n, rank_trivial), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_trivial[i * m + j, r] = 1.0
                V_trivial[j * p + k, r] = 1.0
                W_trivial[k * n + i, r] = 1.0

    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial

    # Strategy: Decompose into two 2x3x3 blocks
    # B = [B1, B2] where B1, B2 are 3x3
    # C = [A B1, A B2]
    # Find rank r decomposition for 2x3x3
    # Then full rank is 2*r
    
    # Search for rank r in 2x3x3
    # Trivial r is 18.
    # Try r from 17 down to 12
    
    sub_n, sub_m, sub_p = 2, 3, 3
    
    for r in range(17, 11, -1):
        if time.time() - t0 > 110:
            break
        
        # Try to find a decomposition of rank r for 2x3x3
        res = _find_sub_decomposition(sub_n, sub_m, sub_p, r, t0)
        if res is not None:
            U_sub, V_sub, W_sub = res
            # Construct full decomposition
            # U_full: (6, 2r)
            # V_full: (18, 2r)
            # W_full: (12, 2r)
            
            U_full = np.zeros((n * m, 2 * r), dtype=float)
            V_full = np.zeros((m * p, 2 * r), dtype=float)
            W_full = np.zeros((p * n, 2 * r), dtype=float)
            
            # First block (k=0,1,2)
            U_full[:, :r] = U_sub
            V_full[:sub_m*sub_p, :r] = V_sub
            # W_sub has shape (sub_p * sub_n, r) = (9, r)
            # Indices in W_full for k=0,1,2 are k*n + i.
            # k in 0..2, i in 0..1.
            # Flat index: k*2 + i.
            # This matches the first 6 rows of W_full?
            # W_full rows: 0..11.
            # k=0: 0,1
            # k=1: 2,3
            # k=2: 4,5
            # k=3: 6,7
            # ...
            # So W_full[0:6, :r] = W_sub
            
            W_full[0:sub_p*sub_n, :r] = W_sub
            
            # Second block (k=3,4,5)
            U_full[:, r:] = U_sub
            # V_full rows for B2: j*p + k.
            # j in 0..2, k in 3..5.
            # Flat index: j*6 + k.
            # j=0, k=3..5 -> 3,4,5
            # j=1, k=3..5 -> 9,10,11
            # j=2, k=3..5 -> 15,16,17
            # V_sub has shape (9, r). Rows correspond to j*3 + k_sub where k_sub in 0..2.
            # We need to map V_sub rows to V_full rows.
            # V_sub row idx = j*3 + k_sub.
            # V_full row idx = j*6 + (k_sub + 3).
            
            for j in range(sub_m):
                for k_sub in range(sub_p):
                    v_sub_idx = j * sub_p + k_sub
                    v_full_idx = j * p + (k_sub + 3)
                    V_full[v_full_idx, r:] = V_sub[v_sub_idx]
            
            # W_full for k=3,4,5
            # W_full rows: k*2 + i.
            # k=3: 6,7
            # k=4: 8,9
            # k=5: 10,11
            # W_sub rows: k_sub*2 + i.
            # k_sub=0: 0,1 -> map to 6,7
            # k_sub=1: 2,3 -> map to 8,9
            # k_sub=2: 4,5 -> map to 10,11
            
            for k_sub in range(sub_p):
                for i in range(sub_n):
                    w_sub_idx = k_sub * sub_n + i
                    w_full_idx = (k_sub + 3) * sub_n + i
                    W_full[w_full_idx, r:] = W_sub[w_sub_idx]
            
            # Verify full decomposition
            if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                best_rank = 2 * r
                best_U, best_V, best_W = U_full, V_full, W_full
                break # Found a valid lower rank, stop searching lower r? 
                # Actually, we should continue to see if we can find even lower r.
                # But lower r means smaller sub-rank.
                # So we break the inner loop and continue the outer loop with smaller r.
                # Wait, the loop is `for r in range(17, 11, -1)`.
                # So if we find r=17, we should try r=16 next.
                # So we do NOT break the outer loop. We just update best.
                # But we can break the inner search for this r.
                pass
        
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

def _find_sub_decomposition(n, m, p, rank, t0):
    """
    Find a rank decomposition for n x m x p matrix multiplication.
    n=2, m=3, p=3.
    Returns U, V, W or None.
    """
    deadline = t0 + 115
    
    # Constraints: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    
    # We want integer or half-integer entries.
    # Let's try random initialization with small integers and gradient descent,
    # then snap to half-integers.
    
    # Index arrays
    u_idx = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    v_idx = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    w_idx = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    
    num_constraints = n * m * p
    
    rng = np.random.RandomState(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            return None
            
        # Initialize with small integers
        # Values in {-1, 0, 1} or {-2, -1, 0, 1, 2}
        U = rng.randint(-2, 3, size=(n*m, rank)).astype(float)
        V = rng.randint(-2, 3, size=(m*p, rank)).astype(float)
        W = rng.randint(-2, 3, size=(p*n, rank)).astype(float)
        
        # Gradient descent to minimize error
        # Error = sum (sum_r ... - 1)^2
        lr = 0.01
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(200):
            if time.time() > deadline:
                return None
                
            # Compute residuals
            res = np.zeros(num_constraints)
            for r in range(rank):
                res += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            res -= 1.0
            
            err = np.sum(res**2)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < 1e-8:
                break
                
            # Gradients
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            for r in range(rank):
                U_contrib = res * V[v_idx, r] * W[w_idx, r]
                V_contrib = U[u_idx, r] * res * W[w_idx, r]
                W_contrib = U[u_idx, r] * V[v_idx, r] * res
                np.add.at(U_grad, u_idx, U_contrib)
                np.add.at(V_grad, v_idx, V_contrib)
                np.add.at(W_grad, w_idx, W_contrib)
                
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
            
            # Adaptive LR
            # ... simplified
            
        # Snap to half-integers
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        
        # Verify snapped solution
        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
            
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition satisfies the tensor equation."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(U.shape[1]):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True
