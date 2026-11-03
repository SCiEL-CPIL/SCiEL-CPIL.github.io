"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    
    c = a + a.T
    
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    best_p = p.copy()
    current_cost = calc_cost(best_p)
    best_cost = current_cost
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    while time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            min_delta = 0
            min_i = -1
            min_j = -1
            
            Bp = b[np.ix_(p, p)]
            
            V = np.sum(c * Bp, axis=1)
            U = c @ Bp
            T = Bp @ c.T
            
            Main = U - V[:, None] - V[None, :] + T
            
            DiagB = np.diag(Bp)
            DiagA = np.diag(a)
            Base = (DiagA[:, None] - DiagA[None, :]) * (DiagB[None, :] - DiagB[:, None])
            
            C_ii = c.diagonal()
            
            M1 = (C_ii[:, None] - c) * (Bp - DiagB[:, None])
            M2 = (c - C_ii[None, :]) * (DiagB[None, :] - Bp)
            
            Corr = M1 + M2
            
            Delta = Base + Main - Corr
            np.fill_diagonal(Delta, np.inf)
            
            min_idx = np.unravel_index(np.argmin(Delta), Delta.shape)
            min_i, min_j = min_idx
            min_delta = Delta[min_i, min_j]
            
            if min_delta < 0:
                p[min_i], p[min_j] = p[min_j], p[min_i]
                current_cost += min_delta
                improved = True
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
        if time.time() < deadline:
            idxs = np.random.choice(n, 4, replace=False)
            p[idxs[0]], p[idxs[1]] = p[idxs[1]], p[idxs[0]]
            p[idxs[2]], p[idxs[3]] = p[idxs[3]], p[idxs[2]]
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
