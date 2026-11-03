import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Initial solution: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute
    A_sym = a + a.T
    AS_diag = np.diag(A_sym)
    a_diag = np.diag(a)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Local Search
    while time.time() < deadline:
        # Compute permuted B
        C = b[p[:, None], p[None, :]]
        C_diag = np.diag(C)
        
        # Compute M = A_sym @ C.T
        M = A_sym @ C.T
        
        # Compute Q = (A_sym * C).sum(axis=1)
        Q = (A_sym * C).sum(axis=1)
        
        # T = M - Q[:, None] - Q[None, :] + M.T
        T = M - Q[:, None] - Q[None, :] + M.T
        
        # Corrections for k=u, v
        # Term1: (AS_diag[u] - A_sym[u,v]) * (C[v,u] - C[u,u])
        Term1 = (AS_diag[:, None] - A_sym) * (C.T - C_diag[:, None])
        
        # Term2: (A_sym[u,v] - AS_diag[v]) * (C[v,v] - C[u,v])
        Term2 = (A_sym - AS_diag[None, :]) * (C_diag[None, :] - C)
        
        Corr = Term1 + Term2
        T_prime = T - Corr
        
        # Diagonal Delta
        # TermA: (a_uu - a_vv) * (C_vv - C_uu)
        TermA = (a_diag[:, None] - a_diag[None, :]) * (C_diag[None, :] - C_diag[:, None])
        
        # TermB: (a_uv - a_vu) * (C_vu - C_uv)
        TermB = (a - a.T) * (C.T - C)
        
        D_diag = TermA + TermB
        
        Delta = T_prime + D_diag
        
        # Mask diagonal
        np.fill_diagonal(Delta, np.inf)
        
        # Find best swap
        min_val = np.min(Delta)
        
        if min_val >= 0:
            break # Local optimum reached
            
        # Get indices of min
        u, v = np.unravel_index(np.argmin(Delta), Delta.shape)
        
        # Perform swap
        p[u], p[v] = p[v], p[u]
        
    return p.tolist()
