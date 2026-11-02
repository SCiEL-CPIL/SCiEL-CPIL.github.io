import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # Build Strassen rank-21 baseline
    us = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0]
    ]
    vs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0]
    ]
    ws = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    rank_21 = 21
    U21 = np.zeros((dim_u, rank_21))
    V21 = np.zeros((dim_v, rank_21))
    W21 = np.zeros((dim_w, rank_21))

    for block in range(3):
        for r in range(7):
            c = block * 7 + r
            U21[:, c] = us[r]
            V21[2*block, c] = vs[r][0]
            V21[2*block+1, c] = vs[r][1]
            V21[6+2*block, c] = vs[r][2]
            V21[6+2*block+1, c] = vs[r][3]
            W21[4*block, c] = ws[r][0]
            W21[4*block+2, c] = ws[r][1]
            W21[4*block+1, c] = ws[r][2]
            W21[4*block+3, c] = ws[r][3]

    best_result = {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
    best_rank = 21

    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    def check_exact(U, V, W, tol=1e-9):
        """Check if decomposition is exact and entries are half-integers."""
        for mat in [U, V, W]:
            for val in mat.flat:
                if abs(val - round(val * 2) / 2) > tol:
                    return False
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        if np.max(np.abs(T_approx - T)) < tol:
            return True
        return False

    def refine(U, V, W):
        """Round to nearest half-integer."""
        return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2

    def residual_norm(U, V, W):
        """Compute residual norm of tensor decomposition."""
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.linalg.norm(T_approx - T)

    # Strategy: structured random search over rank-20 decompositions
    # Use sparse small-integer parameterization with rapid residual check
    # The 2x2x6 tensor has specific symmetries that may allow rank-20 solutions
    
    rank_20 = 20
    np.random.seed(42)
    
    # Try structured random search for rank-20
    # Parameterize with sparse entries from {-1, 0, 1} and {-1, 0, 1, 2}
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Initialize with sparse random entries
        U = np.random.randint(-1, 2, size=(dim_u, rank_20))
        V = np.random.randint(-1, 2, size=(dim_v, rank_20))
        W = np.random.randint(-1, 2, size=(dim_w, rank_20))
        
        # Add some structure: make columns mostly sparse
        for c in range(rank_20):
            # Zero out some entries to maintain sparsity
            mask_u = np.random.random(dim_u) > 0.5
            mask_v = np.random.random(dim_v) > 0.5
            mask_w = np.random.random(dim_w) > 0.5
            U[mask_u, c] = 0
            V[mask_v, c] = 0
            W[mask_w, c] = 0
        
        # Quick residual check
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        resid = np.linalg.norm(T_approx - T)
        
        if resid < 1e-6:
            U_r, V_r, W_r = refine(U, V, W)
            if check_exact(U_r, V_r, W_r):
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = rank_20
                return best_result

    # Try rank-19 with sparser initialization
    rank_19 = 19
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        U = np.random.randint(-1, 2, size=(dim_u, rank_19))
        V = np.random.randint(-1, 2, size=(dim_v, rank_19))
        W = np.random.randint(-1, 2, size=(dim_w, rank_19))
        
        # More aggressive sparsification
        for c in range(rank_19):
            mask_u = np.random.random(dim_u) > 0.4
            mask_v = np.random.random(dim_v) > 0.4
            mask_w = np.random.random(dim_w) > 0.4
            U[mask_u, c] = 0
            V[mask_v, c] = 0
            W[mask_w, c] = 0
        
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        resid = np.linalg.norm(T_approx - T)
        
        if resid < 1e-6:
            U_r, V_r, W_r = refine(U, V, W)
            if check_exact(U_r, V_r, W_r):
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = rank_19
                return best_result

    # Try rank-18 with even sparser initialization
    rank_18 = 18
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        U = np.random.randint(-1, 2, size=(dim_u, rank_18))
        V = np.random.randint(-1, 2, size=(dim_v, rank_18))
        W = np.random.randint(-1, 2, size=(dim_w, rank_18))
        
        # Very sparse
        for c in range(rank_18):
            mask_u = np.random.random(dim_u) > 0.3
            mask_v = np.random.random(dim_v) > 0.3
            mask_w = np.random.random(dim_w) > 0.3
            U[mask_u, c] = 0
            V[mask_v, c] = 0
            W[mask_w, c] = 0
        
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        resid = np.linalg.norm(T_approx - T)
        
        if resid < 1e-6:
            U_r, V_r, W_r = refine(U, V, W)
            if check_exact(U_r, V_r, W_r):
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = rank_18
                return best_result

    return best_result
