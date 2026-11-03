import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    return T_target

def _verify(U, V, W, n, m, p, T_target=None, tol=1e-8):
    """Check if U,V,W form a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if T_target is None:
        T_target = _build_target_tensor(n, m, p)
    
    # Compute reconstructed tensor
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check difference
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _als_polish(U, V, W, T_target, n, m, p, max_iter=100, tol=1e-12):
    """Polish a decomposition using ALS."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    rank = U.shape[1]
    
    prev_cost = np.inf
    
    for it in range(max_iter):
        # Solve for U: fix V, W
        VW = np.einsum('br,cr->bcr', V, W)
        VW_flat = VW.transpose(2, 0, 1).reshape(rank, dim_V * dim_W)
        T_flat = T_target.reshape(dim_U, dim_V * dim_W)
        U_new, _, _, _ = np.linalg.lstsq(VW_flat.T, T_flat.T, rcond=None)
        U = U_new.T
        
        # Solve for V: fix U, W
        UW = np.einsum('ar,cr->acr', U, W)
        UW_flat = UW.transpose(2, 0, 1).reshape(rank, dim_U * dim_W)
        T_flat2 = T_target.reshape(dim_U * dim_W, dim_V)
        V_new, _, _, _ = np.linalg.lstsq(UW_flat.T, T_flat2.T, rcond=None)
        V = V_new.T
        
        # Solve for W: fix U, V
        UV = np.einsum('ar,br->abr', U, V)
        UV_flat = UV.transpose(2, 0, 1).reshape(rank, dim_U * dim_V)
        T_flat3 = T_target.reshape(dim_U * dim_V, dim_W)
        W_new, _, _, _ = np.linalg.lstsq(UV_flat.T, T_flat3.T, rcond=None)
        W = W_new.T
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        cost = np.sum((T_recon - T_target)**2)
        
        if abs(prev_cost - cost) < tol * max(1.0, cost):
            break
        prev_cost = cost
    
    return U, V, W, cost

def _try_merge_schoolbook(n, m, p, target_rank, T_target, max_time=5.0):
    """
    Try to merge schoolbook columns to achieve target_rank.
    Each schoolbook column corresponds to one (i,j,k) triple.
    We try to combine multiple schoolbook terms into one rank term.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Build schoolbook decomposition
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_U, schoolbook_rank))
    V_sb = np.zeros((dim_V, schoolbook_rank))
    W_sb = np.zeros((dim_W, schoolbook_rank))
    
    col = 0
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1.0
                V_sb[b, col] = 1.0
                W_sb[c, col] = 1.0
                triples.append((i, j, k))
                col += 1
    
    # Try to merge columns by finding linear dependencies
    # Strategy: randomly partition the 24 schoolbook terms into groups,
    # then try to find coefficients for each group
    
    import itertools
    import random
    
    best_result = None
    start_time = time.time()
    
    # Generate random partitions of the 24 items into target_rank groups
    rng = np.random.RandomState(42)
    
    for attempt in range(50):
        if time.time() > start_time + max_time:
            break
        
        # Create a random partition
        labels = rng.randint(0, target_rank, size=schoolbook_rank)
        
        # For each group, try to find coefficients
        U_new = np.zeros((dim_U, target_rank))
        V_new = np.zeros((dim_V, target_rank))
        W_new = np.zeros((dim_W, target_rank))
        
        valid = True
        for r in range(target_rank):
            # Find all schoolbook columns in this group
            mask = (labels == r)
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                # Empty group - assign zero
                continue
            
            # For each index in the group, add the schoolbook column
            # with a random coefficient
            for idx in indices:
                coeff = rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
                U_new[:, r] += coeff * U_sb[:, idx]
                V_new[:, r] += coeff * V_sb[:, idx]
                W_new[:, r] += coeff * W_sb[:, idx]
        
        # Polish with ALS
        try:
            U_pol, V_pol, W_pol, cost = _als_polish(
                U_new, V_new, W_new, T_target, n, m, p, max_iter=50
            )
            
            if cost < 1e-6:
                # Try rounding
                for mult in [1, 2, 4]:
                    U_rounded = np.round(U_pol * mult) / mult
                    V_rounded = np.round(V_pol * mult) / mult
                    W_rounded = np.round(W_pol * mult) / mult
                    
                    if _verify(U_rounded, V_rounded, W_rounded, n, m, p, T_target):
                        return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                
                # Try integer rounding
                U_int = np.round(U_pol)
                V_int = np.round(V_pol)
                W_int = np.round(W_pol)
                if _verify(U_int, V_int, W_int, n, m, p, T_target):
                    return (U_int.tolist(), V_int.tolist(), W_int.tolist())
                    
        except Exception:
            pass
    
    return None

def _structured_search(n, m, p, target_rank, T_target, max_time=5.0):
    """
    Structured search: start from schoolbook and merge terms.
    Uses the structure of the matrix multiplication tensor.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Build schoolbook
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_U, schoolbook_rank))
    V_sb = np.zeros((dim_V, schoolbook_rank))
    W_sb = np.zeros((dim_W, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1.0
                V_sb[b, col] = 1.0
                W_sb[c, col] = 1.0
                col += 1
    
    # Try different merging strategies
    start_time = time.time()
    rng = np.random.RandomState(0)
    
    # Strategy 1: Random linear combinations of groups
    for attempt in range(100):
        if time.time() > start_time + max_time:
            break
        
        # Randomly assign each schoolbook term to a rank slot
        assignment = rng.randint(0, target_rank, size=schoolbook_rank)
        
        U_new = np.zeros((dim_U, target_rank))
        V_new = np.zeros((dim_V, target_rank))
        W_new = np.zeros((dim_W, target_rank))
        
        for r in range(target_rank):
            mask = (assignment == r)
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            # Assign random coefficients
            coeffs = rng.choice([-1, 1], size=len(indices)) * rng.uniform(0.5, 1.5, size=len(indices))
            
            for idx, coeff in zip(indices, coeffs):
                U_new[:, r] += coeff * U_sb[:, idx]
                V_new[:, r] += coeff * V_sb[:, idx]
                W_new[:, r] += coeff * W_sb[:, idx]
        
        # Polish
        try:
            U_pol, V_pol, W_pol, cost = _als_polish(
                U_new, V_new, W_new, T_target, n, m, p, max_iter=30
            )
            
            if cost < 1e-6:
                for mult in [1, 2, 4]:
                    U_rounded = np.round(U_pol * mult) / mult
                    V_rounded = np.round(V_pol * mult) / mult
                    W_rounded = np.round(W_pol * mult) / mult
                    
                    if _verify(U_rounded, V_rounded, W_rounded, n, m, p, T_target):
                        return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                
                U_int = np.round(U_pol)
                V_int = np.round(V_pol)
                W_int = np.round(W_pol)
                if _verify(U_int, V_int, W_int, n, m, p, T_target):
                    return (U_int.tolist(), V_int.tolist(), W_int.tolist())
                    
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = 2, 3, 4
    T_target = _build_target_tensor(n, m, p)
    
    # Try ranks from 16 up to 23 (schoolbook is 24)
    # 16 is the theoretical lower bound for 2x3x4
    for rank in range(16, 24):
        if time.time() > _DEADLINE - 5:
            break
        
        # Try structured search
        result = _structured_search(n, m, p, rank, T_target, max_time=3.0)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Try merge schoolbook
        result = _try_merge_schoolbook(n, m, p, rank, T_target, max_time=3.0)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() > _DEADLINE - 10:
            break
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
