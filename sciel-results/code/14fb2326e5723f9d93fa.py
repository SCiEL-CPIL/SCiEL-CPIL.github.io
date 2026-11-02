"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Scan ranks from 23 down to 12.
    2. Use efficient ALS with multiple diverse initializations (random and structured).
    3. Snap solutions to half-integers and verify exactness.
    4. Fall back to schoolbook rank 24.
    """
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    deadline = time.time() + 140  # 140s budget, 10s margin
    
    # Try ranks from 23 down to 12
    # Rank 24 is the schoolbook fallback.
    for rank in range(23, 11, -1):
        if time.time() > deadline:
            break
        
        # Determine number of trials based on rank
        # Higher ranks are easier to find, so we can afford fewer trials?
        # Or use all time for the first few ranks?
        # Let's use a fixed generous number of trials, e.g., 50, for each rank.
        # If rank 23 is found quickly, we return.
        # If not, we move to 22.
        
        max_trials = 50
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Generate unique seed for diversity
            seed = rank * 10000 + trial * 100 + 42
            np.random.seed(seed)
            
            # Initialization Strategy
            if trial % 3 == 0:
                # Random Gaussian
                U = np.random.randn(nm, rank) * 0.5
                V = np.random.randn(mp, rank) * 0.5
                W = np.random.randn(pn, rank) * 0.5
            elif trial % 3 == 1:
                # Structured: Sparse ones with noise
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                for c in range(rank):
                    U[np.random.randint(nm), c] = 1.0
                    V[np.random.randint(mp), c] = 1.0
                    W[np.random.randint(pn), c] = 1.0
                U += np.random.randn(nm, rank) * 0.1
                V += np.random.randn(mp, rank) * 0.1
                W += np.random.randn(pn, rank) * 0.1
            else:
                # Schoolbook-like initialization for high ranks
                # If rank >= 24, use schoolbook. For rank < 24, use first 'rank' terms + noise
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                
                # Fill with schoolbook terms
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < rank:
                                U[i * m + j, col] = 1.0
                                V[j * p + k, col] = 1.0
                                W[k * n + i, col] = 1.0
                                col += 1
                            else:
                                break
                
                # Add noise to break symmetry
                U += np.random.randn(nm, rank) * 0.05
                V += np.random.randn(mp, rank) * 0.05
                W += np.random.randn(pn, rank) * 0.05
            
            # Optimize using ALS
            U, V, W, loss = _optimize_als(T, U, V, W, nm, mp, pn, rank, deadline)
            
            if loss < 1e-8:
                # Try to snap to half-integers
                result = _snap_and_verify(T, U, V, W, nm, mp, pn)
                if result is not None:
                    return result
                
                # If snapping fails, try integer snapping
                result = _snap_and_verify_int(T, U, V, W, nm, mp, pn)
                if result is not None:
                    return result
                
                # If both fail, the solution might be real-valued but not half-integer.
                # We cannot use it. Continue searching.
                # However, if loss is extremely low, maybe we can try to find a nearby half-integer solution?
                # For now, just continue.

    # Fallback: schoolbook rank 24
    return _schoolbook(n, m, p)


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _optimize_als(T, U, V, W, nm, mp, pn, rank, deadline):
    """
    Efficient Alternating Least Squares optimization.
    """
    # Precompute reshaped T for fast access
    T_flat_u = T.reshape(nm, mp * pn)
    T_flat_v = T.reshape(mp, nm * pn) # Note: T is (nm, mp, pn). 
    # For V update: T_{b, a, d} = sum_c U_{a,c} W_{d,c} V_{b,c}
    # Fix b. T[:, b, :] is (nm, pn).
    # We need to solve for V[:, b] given U, W.
    # T_{a, d} = sum_c V_{b,c} (U_{a,c} W_{d,c})
    # Let M_{a,d,c} = U_{a,c} W_{d,c}.
    # T_{a,d} = sum_c V_{b,c} M_{a,d,c}
    # This is a least squares problem for V_{b,:}.
    # Vectorize: T_flat (nm*pn, ) = V_b (rank, )^T * M_flat (nm*pn, rank)?
    # Actually, T_flat (nm*pn, 1) = M_flat.T (rank, nm*pn) * V_b (rank, 1)? No.
    # T_{a,d} = sum_c U_{a,c} V_{b,c} W_{d,c}.
    # Let X_{a,d} = T_{a,d}.
    # Let A_{a,d,c} = U_{a,c} W_{d,c}.
    # X_{a,d} = sum_c A_{a,d,c} V_{b,c}.
    # This is linear in V_{b,:}.
    # We can form a matrix M of size (nm*pn, rank) where M_{(a,d), c} = U_{a,c} W_{d,c}.
    # Then X_flat = M.T @ V_b.
    # V_b = lstsq(M.T, X_flat).
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Update U
        # T_{a,b,d} = sum_c U_{a,c} V_{b,c} W_{d,c}
        # Fix a. T_{a,:,d} = sum_c U_{a,c} (V_{b,c} W_{d,c})
        # M_{(b,d), c} = V_{b,c} W_{d,c}
        # T_flat_a (mp*pn, ) = M.T (rank, mp*pn) @ U_a (rank, )
        # U_a = lstsq(M.T, T_flat_a)
        
        VW = np.einsum('bc,dc->bdc', V, W) # (mp, pn, rank)
        VW_flat = VW.reshape(mp * pn, rank)
        T_flat_a = T.reshape(nm, mp * pn)
        
        # Solve for each a
        # U[:, a] = lstsq(VW_flat.T, T_flat_a[a])
        # Vectorized: U = lstsq(VW_flat.T, T_flat_a.T).T ?
        # VW_flat.T is (rank, mp*pn). T_flat_a.T is (mp*pn, nm).
        # We want U (nm, rank).
        # U = lstsq(VW_flat.T, T_flat_a.T) gives (rank, nm).
        try:
            U_new = np.linalg.lstsq(VW_flat.T, T_flat_a.T, rcond=None)[0].T
        except:
            U_new = U
        
        # Update V
        # T_{a,b,d} = sum_c V_{b,c} (U_{a,c} W_{d,c})
        # M_{(a,d), c} = U_{a,c} W_{d,c}
        # T_flat_b (nm*pn, ) = M.T (rank, nm*pn) @ V_b (rank, )
        # V = lstsq(M.T, T_flat_b.T).T
        
        UW = np.einsum('ac,dc->adc', U, W) # (nm, pn, rank)
        UW_flat = UW.reshape(nm * pn, rank)
        T_flat_b = T.reshape(mp, nm * pn)
        
        try:
            V_new = np.linalg.lstsq(UW_flat.T, T_flat_b.T, rcond=None)[0].T
        except:
            V_new = V
        
        # Update W
        # T_{a,b,d} = sum_c W_{d,c} (U_{a,c} V_{b,c})
        # M_{(a,b), c} = U_{a,c} V_{b,c}
        # T_flat_d (nm*mp, ) = M.T (rank, nm*mp) @ W_d (rank, )
        # W = lstsq(M.T, T_flat_d.T).T
        
        UV = np.einsum('ac,bc->abc', U, V) # (nm, mp, rank)
        UV_flat = UV.reshape(nm * mp, rank)
        T_flat_d = T.reshape(pn, nm * mp)
        
        try:
            W_new = np.linalg.lstsq(UV_flat.T, T_flat_d.T, rcond=None)[0].T
        except:
            W_new = W
        
        U, V, W = U_new, V_new, W_new
        
        # Compute loss
        tensor = np.einsum('ac,bc,dc->abd', U, V, W)
        loss = np.sum((tensor - T) ** 2)
        
        if loss < 1e-12:
            break
    
    return U, V, W, loss


def _snap_and_verify(T, U, V, W, nm, mp, pn):
    """
    Snap to half-integers and verify exact decomposition.
    """
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    if _verify(T, U_snap, V_snap, W_snap, nm, mp, pn):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return None

def _snap_and_verify_int(T, U, V, W, nm, mp, pn):
    """
    Snap to integers and verify exact decomposition.
    """
    U_snap = np.round(U)
    V_snap = np.round(V)
    W_snap = np.round(W)
    
    if _verify(T, U_snap, V_snap, W_snap, nm, mp, pn):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return None


def _verify(T, U, V, W, nm, mp, pn):
    """Verify exact decomposition."""
    tensor = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(tensor, T, atol=1e-9)
# EVOLVE-BLOCK-END
