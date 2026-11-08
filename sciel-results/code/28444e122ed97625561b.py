"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_target(n, m, p):
    """Build the target tensor T[a,b,c] with 1s at required positions."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_np(U, V, W, n, m, p, rank, target=None):
    """Fast numpy-based verification."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    if target is None:
        target = _build_target(n, m, p)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    diff = T - target
    return np.max(np.abs(diff)) < 1e-9

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank55(n, m, p):
    """Block-based rank-55 decomposition for 2x3 x 3x10."""
    rank = 55
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    col = 0
    for t in range(5):
        k0 = 2*t
        k1 = 2*t + 1

        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0*3+0][col] += 1
        U[1*3+1][col] += 1
        V[0*10+k0][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1
        W[k1*2+1][col] += 1
        col += 1

        # M2: (A10 + A11) * B0,k0
        U[1*3+0][col] += 1
        U[1*3+1][col] += 1
        V[0*10+k0][col] += 1
        W[k0*2+1][col] += 1
        W[k1*2+1][col] -= 1
        col += 1

        # M3: A00 * (B0,k1 - B1,k1)
        U[0*3+0][col] += 1
        V[0*10+k1][col] += 1
        V[1*10+k1][col] -= 1
        W[k1*2+0][col] += 1
        W[k1*2+1][col] += 1
        col += 1

        # M4: A11 * (B1,k0 - B0,k0)
        U[1*3+1][col] += 1
        V[1*10+k0][col] += 1
        V[0*10+k0][col] -= 1
        W[k0*2+0][col] += 1
        W[k0*2+1][col] += 1
        col += 1

        # M5: (A00 + A01) * B1,k1
        U[0*3+0][col] += 1
        U[0*3+1][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] -= 1
        W[k1*2+0][col] += 1
        col += 1

        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[1*3+0][col] += 1
        U[0*3+0][col] -= 1
        V[0*10+k0][col] += 1
        V[0*10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1

        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[0*3+1][col] += 1
        U[1*3+1][col] -= 1
        V[1*10+k0][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1
        col += 1

        # Direct terms for row 2 (A02, A12) with B2
        U[0*3+2][col] += 1
        V[2*10+k0][col] += 1
        W[k0*2+0][col] += 1
        col += 1

        U[0*3+2][col] += 1
        V[2*10+k1][col] += 1
        W[k1*2+0][col] += 1
        col += 1

        U[1*3+2][col] += 1
        V[2*10+k0][col] += 1
        W[k0*2+1][col] += 1
        col += 1

        U[1*3+2][col] += 1
        V[2*10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}

def _try_merge_rank55_to_54(n, m, p, deadline):
    """Try to merge two terms from rank-55 into one, achieving rank-54."""
    base = _rank55(n, m, p)
    U0 = np.array(base["U"], dtype=np.float64)
    V0 = np.array(base["V"], dtype=np.float64)
    W0 = np.array(base["W"], dtype=np.float64)
    target = _build_target(n, m, p)
    
    # Current tensor from rank-55
    T55 = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(55):
        T55 += np.einsum('a,b,c->abc', U0[:,r], V0[:,r], W0[:,r])
    
    # Try all pairs of terms to see if they can be merged
    # Two terms r1, r2 can be merged if there exist coefficients such that
    # U[r1]*V[r1]*W[r1] + U[r2]*V[r2]*W[r2] = U_new*V_new*W_new
    # This is generally not possible, but we can try specific patterns
    
    # Alternative: try removing one term and adjusting others
    # For each term r, try to find a replacement that covers its contribution
    # using modifications to other terms
    
    best_result = None
    best_residual = np.inf
    
    # Strategy: for each pair (r1, r2), try to replace both with a single term
    # by finding U_new, V_new, W_new such that the sum matches
    # This is a rank-2 to rank-1 reduction, which is generally impossible
    # unless the two tensors are proportional
    
    # Better strategy: try to find a term r* that can be eliminated by
    # adjusting the coefficients of other terms
    # For each r*, compute what T - U[r*]⊗V[r*]⊗W[r*] looks like
    # and check if it can be represented with rank 54
    
    # Most practical: try local search from rank-55
    # Remove one random term and try to fix the residual
    
    rng = np.random.default_rng(123)
    
    for remove_r in range(55):
        if time.time() > deadline - 15:
            break
        
        # Start with rank-55 minus term remove_r
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        U[remove_r] = 0
        V[remove_r] = 0
        W[remove_r] = 0
        
        # Compute current residual
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(55):
            if r != remove_r:
                T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        residual = T - target
        
        # Try to fix by modifying other terms
        # For each other term r', try adjusting its entries
        for iteration in range(200):
            if time.time() > deadline - 15:
                break
            
            # Find the term with largest contribution to residual
            # Try modifying random entries
            r_mod = rng.integers(55)
            if r_mod == remove_r:
                r_mod = (r_mod + 1) % 55
            
            # Try modifying U, V, or W entry
            factor = rng.integers(3)
            if factor == 0:
                idx = rng.integers(n*m)
                old_val = U[idx, r_mod]
                # Try setting to different values
                for new_val in [old_val + 1, old_val - 1, -old_val, 0, 1, -1]:
                    if new_val == old_val:
                        continue
                    U[idx, r_mod] = new_val
                    T_new = T + new_val * np.einsum('b,c->bc', V[:,r_mod], W[:,r_mod]) * U[idx, r_mod] / new_val if new_val != 0 else T
                    # Simpler: recompute contribution
                    old_contrib = old_val * np.outer(V[:,r_mod], W[:,r_mod]) if old_val != 0 else 0
                    new_contrib = new_val * np.outer(V[:,r_mod], W[:,r_mod]) if new_val != 0 else 0
                    T_diff = new_contrib - old_contrib
                    T_updated = T + T_diff
                    new_residual = T_updated - target
                    new_norm = np.sum(new_residual**2)
                    if new_norm < np.sum(residual**2) - 1e-12:
                        T = T_updated
                        residual = new_residual
                        break
                    else:
                        U[idx, r_mod] = old_val
            elif factor == 1:
                idx = rng.integers(m*p)
                old_val = V[idx, r_mod]
                for new_val in [old_val + 1, old_val - 1, -old_val, 0, 1, -1]:
                    if new_val == old_val:
                        continue
                    V[idx, r_mod] = new_val
                    old_contrib = old_val * np.outer(U[:,r_mod], W[:,r_mod]) if old_val != 0 else 0
                    new_contrib = new_val * np.outer(U[:,r_mod], W[:,r_mod]) if new_val != 0 else 0
                    T_diff = new_contrib - old_contrib
                    T_updated = T + T_diff
                    new_residual = T_updated - target
                    new_norm = np.sum(new_residual**2)
                    if new_norm < np.sum(residual**2) - 1e-12:
                        T = T_updated
                        residual = new_residual
                        break
                    else:
                        V[idx, r_mod] = old_val
            else:
                idx = rng.integers(p*n)
                old_val = W[idx, r_mod]
                for new_val in [old_val + 1, old_val - 1, -old_val, 0, 1, -1]:
                    if new_val == old_val:
                        continue
                    W[idx, r_mod] = new_val
                    old_contrib = old_val * np.outer(U[:,r_mod], V[:,r_mod]) if old_val != 0 else 0
                    new_contrib = new_val * np.outer(U[:,r_mod], V[:,r_mod]) if new_val != 0 else 0
                    T_diff = new_contrib - old_contrib
                    T_updated = T + T_diff
                    new_residual = T_updated - target
                    new_norm = np.sum(new_residual**2)
                    if new_norm < np.sum(residual**2) - 1e-12:
                        T = T_updated
                        residual = new_residual
                        break
                    else:
                        W[idx, r_mod] = old_val
            
            if np.max(np.abs(residual)) < 1e-9:
                # Found a solution!
                U[remove_r] = 0
                V[remove_r] = 0
                W[remove_r] = 0
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if np.max(np.abs(residual)) < best_residual:
            best_residual = np.max(np.abs(residual))
    
    return None

def _try_rank54_structured(n, m, p, deadline):
    """Structured search for rank-54 using pair grouping."""
    # Group the 10 columns into 5 pairs
    # For each pair, we need to multiply 2x3 by 3x2
    # The 2x3 x 3x2 multiplication has rank at most 2*3*2 = 12
    # But we can do better using Strassen-like methods
    
    # Actually, let's try a different grouping
    # Group into 3 groups: sizes 4, 3, 3
    # Or try to find a rank-54 decomposition directly
    
    # Use the fact that 2x3 x 3x10 can be decomposed as:
    # - 2x3 x 3x4 (rank ~?)
    # - 2x3 x 3x6 (rank ~?)
    
    # Let me try: split B into B1 (3x4) and B2 (3x6)
    # 2x3 x 3x4: schoolbook rank = 2*3*4 = 24
    # 2x3 x 3x6: schoolbook rank = 2*3*6 = 36
    # Total: 60, not helpful
    
    # Better: use the block structure
    # A = [[A00, A01, A02], [A10, A11, A12]]
    # B = [[B0], [B1], [B2]] where Bj is 1x10
    
    # C0 = A00*B0 + A01*B1 + A02*B2
    # C1 = A10*B0 + A11*B1 + A12*B2
    
    # The current rank-55 uses 11 terms per pair of columns
    # To get rank-54, we need to save 1 term total
    
    # Try: can we share one more term across blocks?
    # The direct terms for row 2 (A02, A12) are 4 terms per block
    # Maybe we can share some of these
    
    # Alternative: try a completely different decomposition
    # Use the fact that we only need to multiply by a 3x10 matrix
    
    # Let's try: express the computation using 54 scalar multiplications
    # This requires finding 54 trilinear forms that span the multiplication
    
    # For now, fall back to the local search approach
    return _try_merge_rank55_to_54(n, m, p, deadline)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin

    # Try rank-54 structured search
    if time.time() < deadline - 30:
        try:
            res54 = _try_rank54_structured(n, m, p, deadline)
            if res54 is not None:
                if _verify_np(res54["U"], res54["V"], res54["W"], n, m, p, 54):
                    return res54
        except Exception:
            pass

    # Try rank-55 decomposition with verification
    try:
        res55 = _rank55(n, m, p)
        if _verify_np(res55["U"], res55["V"], res55["W"], n, m, p, 55):
            return res55
    except Exception:
        pass

    # Fallback: schoolbook rank-60
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
