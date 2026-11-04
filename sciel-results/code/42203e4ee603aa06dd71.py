import time
import numpy as np
import sys

DIMENSIONS = [4, 4, 9]

# ---------------------------------------------------------------------------
# Reference solution (Rank 114) from the current program
# ---------------------------------------------------------------------------
_STRASSEN_222 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # m1
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # m2
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # m3
    ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),  # m4
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # m5
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # m6
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # m7
]

def _add_strassen_block(U, V, W, col, a_base, b_base, c_base):
    for (u, v, w) in _STRASSEN_222:
        for t in range(4):
            if u[t]:
                U[a_base + t][col] += u[t]
            if v[t]:
                V[b_base + t][col] += v[t]
            if w[t]:
                W[c_base + t][col] += w[t]
        col += 1
    return col

def _add_444(U, V, W, col, a_base, b_base, c_base):
    U_sums = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_sums = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, -1, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_sums = [
        [1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    for t in range(7):
        u, v, w = U_sums[t], V_sums[t], W_sums[t]
        for s in range(4):
            if u[s]:
                U[a_base + s][col] += u[s]
            if v[s]:
                V[b_base + s][col] += v[s]
            if w[s]:
                W[c_base + s][col] += w[s]
        col += 1
    return col

def _schoolbook_block(U, V, W, col, a_base, b_base, c_base, ar, ac, bc):
    for i in range(ar):
        for j in range(ac):
            for k in range(bc):
                U[a_base + i * ac + j][col] += 1
                V[b_base + j * bc + k][col] += 1
                W[c_base + i * bc + k][col] += 1
                col += 1
    return col

def _build_rank114(n, m, p):
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    col = _add_444(U, V, W, col, 0, 0, 0)
    col = _add_444(U, V, W, col, 0, 4, 4)
    col = _schoolbook_block(U, V, W, col, 0, 8, 8, 4, 4, 1)
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True

# ---------------------------------------------------------------------------
# New Strategy: Greedy SVD Snapping
# ---------------------------------------------------------------------------

def _snap_to_half_int(vec):
    """Snap a vector to the nearest half-integer values."""
    # Multiply by 2, round, divide by 2
    return np.round(vec * 2) / 2

def _greedy_decompose(n, m, p, max_rank, deadline):
    """
    Attempts to find a decomposition of rank <= max_rank using greedy SVD snapping.
    Returns (U, V, W, rank) or None if it fails.
    """
    # Target tensor T
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    U_list = []
    V_list = []
    W_list = []
    
    remaining = T.copy()
    
    # We will try to build up to max_rank terms
    for rank_step in range(max_rank):
        if time.time() > deadline:
            return None
            
        # 1. Find best rank-1 approximation of 'remaining'
        # Flatten remaining to 2D for SVD? No, use iterative SVD on modes.
        # Reshape to (nm, mp*pn) for mode A, etc.
        
        # Mode A: (nm, mp*pn)
        A_mat = remaining.reshape(n * m, -1)
        # Mode B: (mp, nm*pn)
        B_mat = remaining.transpose(1, 0, 2).reshape(m * p, -1)
        # Mode C: (pn, nm*mp)
        C_mat = remaining.transpose(2, 0, 1).reshape(p * n, -1)
        
        # Get leading singular vectors
        try:
            _, _, Vt_A = np.linalg.svd(A_mat, full_matrices=False)
            u_A = Vt_A[0]
            _, _, Vt_B = np.linalg.svd(B_mat, full_matrices=False)
            v_B = Vt_B[0]
            _, _, Vt_C = np.linalg.svd(C_mat, full_matrices=False)
            w_C = Vt_C[0]
        except:
            return None
            
        # 2. Snap to half-integers
        u_A_snap = _snap_to_half_int(u_A)
        v_B_snap = _snap_to_half_int(v_B)
        w_C_snap = _snap_to_half_int(w_C)
        
        # Check if snapping resulted in zero vectors (bad)
        if np.all(u_A_snap == 0) or np.all(v_B_snap == 0) or np.all(w_C_snap == 0):
             # Try random small integers as a fallback for this step?
             # Or break if we can't make progress
             if rank_step > 10:
                 return None
             # Random fallback
             u_A_snap = np.random.choice([-1, 0, 1], size=n*m)
             v_B_snap = np.random.choice([-1, 0, 1], size=m*p)
             w_C_snap = np.random.choice([-1, 0, 1], size=p*n)
             # Ensure non-zero
             if np.all(u_A_snap == 0): u_A_snap[0] = 1
             if np.all(v_B_snap == 0): v_B_snap[0] = 1
             if np.all(w_C_snap == 0): w_C_snap[0] = 1

        # 3. Subtract the rank-1 term
        term = np.einsum('a,b,c->abc', u_A_snap, v_B_snap, w_C_snap)
        remaining -= term
        
        # Store the term
        U_list.append(u_A_snap)
        V_list.append(v_B_snap)
        W_list.append(w_C_snap)
        
        # 4. Check if remaining is zero (within tolerance)
        if np.max(np.abs(remaining)) < 1e-9:
            # Success!
            U = np.array(U_list).T # Shape (nm, rank)
            V = np.array(V_list).T
            W = np.array(W_list).T
            return U, V, W, rank_step + 1
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # 1. Try the new Greedy SVD Snapping strategy
    # We aim for rank < 114. Let's try 110.
    # If it fails, we fall back to 114.
    
    # Note: Greedy SVD is stochastic in the sense of tie-breaking, but deterministic if we fix seeds or use standard SVD.
    # However, snapping can lead to local minima.
    
    best_result = None
    best_rank = 114 # Fallback
    
    # Try a few times with different random seeds for the fallback snapping?
    # Actually, SVD is deterministic. But we can try different initializations or random perturbations.
    
    # Let's try rank 110, 105, 100
    for target_rank in [110, 105, 100]:
        if time.time() > deadline:
            break
            
        # We can try to "jitter" the SVD vectors before snapping to escape local minima
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Modify _greedy_decompose to accept a jitter factor?
            # For simplicity, let's just call the standard one.
            # If it fails, we might need a more robust search.
            
            # Let's add a small random noise to the remaining tensor before SVD to perturb the singular vectors
            # This is effectively random restarts.
            
            # We need a version of _greedy_decompose that allows noise.
            # To keep code simple, we'll just run the standard one.
            # If standard fails, we assume it's hard.
            
            res = _greedy_decompose(n, m, p, target_rank, deadline)
            if res:
                U, V, W, r = res
                # Verify
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break # Found a valid one for this target rank, try lower? No, just return best.
        
        # If we found a valid result with rank < 114, we can stop.
        if best_result and best_rank < 114:
            return best_result
            
    # 2. Fallback to Rank 114
    res = _build_rank114(n, m, p)
    if time.time() < deadline and _verify(res["U"], res["V"], res["W"], n, m, p):
        return res
        
    # 3. Final Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Verified:", _verify(res['U'], res['V'], res['W'], 4, 4, 9))
