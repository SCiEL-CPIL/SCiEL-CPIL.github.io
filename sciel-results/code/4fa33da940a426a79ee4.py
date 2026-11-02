import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def _verify_full(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for u in range(n * m):
        i = u // m
        j = u % m
        for v in range(m * p):
            j_v = v // p
            k = v % p
            for w in range(p * n):
                k_w = w // n
                i_w = w % n
                target = 1.0 if (i == i_w and j == j_v and k == k_w) else 0.0
                s = 0.0
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[w][c]
                if abs(s - target) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _greedy_construct(max_rank, deadline):
    """
    Greedy construction: start with zero, add rank-1 terms that best match residual.
    Use half-integer values.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Current decomposition
    U = np.zeros((nm, max_rank))
    V = np.zeros((mp, max_rank))
    W = np.zeros((pn, max_rank))
    residual = T.copy()
    
    for c in range(max_rank):
        if time.time() > deadline:
            break
        
        # Find best rank-1 term u ⊗ v ⊗ w that approximates residual
        # Try all possible half-integer values in a limited range
        # For efficiency, try a subset
        
        best_score = -np.inf
        best_u, best_v, best_w = None, None, None
        
        # Candidate values: -1, -0.5, 0, 0.5, 1
        candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        # To make this tractable, we'll do a coordinate descent on the rank-1 term
        # Initialize with random or greedy
        u_vec = np.zeros(nm)
        v_vec = np.zeros(mp)
        w_vec = np.zeros(pn)
        
        # Greedy initialization: set each coordinate to maximize alignment with residual
        # For u[i], v[j], w[k], the contribution to T[u,v,w] is u[i]*v[j]*w[k]
        # We want sum to approximate residual
        
        # Simple approach: for each coordinate, try the 5 candidate values
        # This is 5^(nm+mp+pn) which is too large
        # Instead, do iterative coordinate descent
        
        # Initialize with small random values
        for _ in range(3):
            u_vec = np.random.choice(candidates, size=nm)
            v_vec = np.random.choice(candidates, size=mp)
            w_vec = np.random.choice(candidates, size=pn)
            
            # Coordinate descent
            for iteration in range(20):
                if time.time() > deadline:
                    break
                improved = False
                # Update u
                for idx in range(nm):
                    best_val = u_vec[idx]
                    best_obj = -np.inf
                    for val in candidates:
                        u_test = u_vec.copy()
                        u_test[idx] = val
                        # Compute alignment: sum over all (u,v,w) of residual[u,v,w] * u[u]*v[v]*w[w]
                        # This is expensive, so approximate
                        # Actually, the objective is to maximize <residual, u⊗v⊗w>
                        # = sum_{u,v,w} residual[u,v,w] * u[u] * v[v] * w[w]
                        # For fixed v,w, this is linear in u[u]
                        # So the best u[idx] is the one that maximizes:
                        # val * (sum_{v,w} residual[idx,v,w] * v[v] * w[w])
                        coeff = np.sum(residual[idx, :, :] * v_vec[None, :] * w_vec[:, None])
                        if coeff > 0:
                            best_val = max(candidates, key=lambda x: x * coeff)
                        elif coeff < 0:
                            best_val = min(candidates, key=lambda x: x * coeff)
                        else:
                            best_val = 0.0
                        if best_val != u_vec[idx]:
                            u_vec[idx] = best_val
                            improved = True
                # Similarly for v and w
                for idx in range(mp):
                    best_val = v_vec[idx]
                    coeff = np.sum(residual[:, idx, :] * u_vec[:, None] * w_vec[None, :])
                    if coeff > 0:
                        best_val = max(candidates, key=lambda x: x * coeff)
                    elif coeff < 0:
                        best_val = min(candidates, key=lambda x: x * coeff)
                    else:
                        best_val = 0.0
                    if best_val != v_vec[idx]:
                        v_vec[idx] = best_val
                        improved = True
                for idx in range(pn):
                    best_val = w_vec[idx]
                    coeff = np.sum(residual[:, :, idx] * u_vec[:, None] * v_vec[None, :])
                    if coeff > 0:
                        best_val = max(candidates, key=lambda x: x * coeff)
                    elif coeff < 0:
                        best_val = min(candidates, key=lambda x: x * coeff)
                    else:
                        best_val = 0.0
                    if best_val != w_vec[idx]:
                        w_vec[idx] = best_val
                        improved = True
                if not improved:
                    break
            
            # Compute score
            term = np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
            score = np.sum(residual * term)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u_vec.copy(), v_vec.copy(), w_vec.copy()
        
        if best_u is None:
            break
        
        U[:, c] = best_u
        V[:, c] = best_v
        W[:, c] = best_w
        
        # Update residual
        term = np.einsum('i,j,k->ijk', best_u, best_v, best_w)
        residual -= term
        
        # Check if residual is close to zero
        if np.max(np.abs(residual)) < 1e-9:
            # Truncate unused columns
            actual_rank = c + 1
            U = U[:, :actual_rank].tolist()
            V = V[:, :actual_rank].tolist()
            W = W[:, :actual_rank].tolist()
            if _verify_full(U, V, W):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try greedy construction with various target ranks
    for target_rank in range(30, best_rank):
        if time.time() > deadline:
            break
        result = _greedy_construct(target_rank, deadline)
        if result is not None:
            U, V, W = result
            if len(U[0]) < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = len(U[0])
                if best_rank <= 30:
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
