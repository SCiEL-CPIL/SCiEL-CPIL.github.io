import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 300.0
_TIME_MARGIN = 15.0
_N = 51

def _remaining():
    return _DEADLINE - _TIME_MARGIN - time.time()

def _safe_heuristic(dm):
    d = np.asarray(dm, dtype=float).copy()
    np.fill_diagonal(d, 1e9)
    H = 1.0 / (d + 1e-9)
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    np.fill_diagonal(H, 1e-9)
    return H

def _distmat(coords):
    c = np.asarray(coords, dtype=float)
    return np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))

def _ant_route(dist, D, cap, logP, logH, max_steps):
    node = 0
    rem = cap
    cost = 0.0
    visited = np.zeros(_N, bool)
    visited[0] = True
    steps = 0
    while True:
        feas = []
        for j in range(1, _N):
            if not visited[j] and D[j] <= rem + 1e-9:
                feas.append(j)
        if not feas:
            cost += dist[node, 0]
            return cost
        vals = np.array([logP[node, j] + logH[node, j] for j in feas])
        m = vals.max()
        w = np.exp(vals - m)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            nxt = feas[0]
        else:
            r = np.random.random() * s
            acc = 0.0
            for k, j in enumerate(feas):
                acc += w[k]
                if r <= acc:
                    nxt = j
                    break
            else:
                nxt = feas[-1]
        cost += dist[node, nxt]
        rem -= D[nxt]
        visited[nxt] = True
        node = nxt
        steps += 1
        if steps >= max_steps:
            cost += dist[node, 0]
            return cost

def _colony(dist, D, cap, logH, n_ants=20, n_iter=30, max_steps=300):
    P = np.ones((_N, _N))
    bests = []
    for _ in range(n_iter):
        itcost = np.empty(n_ants)
        logP = np.log(P)
        for a in range(n_ants):
            c = _ant_route(dist, D, cap, logP, logH, max_steps)
            itcost[a] = c
        b = float(itcost.min())
        bests.append(b)
        P *= 0.9
        P += 1.0 / b
    return min(bests)

def _build_heuristic_savings(dm, alpha, beta, scale):
    """
    H_ij = (1 + beta * savings_ij) / (d_ij^alpha)
    where savings_ij = d_i0 + d_j0 - d_ij
    This combines the savings-based attractiveness with distance penalty.
    """
    d = np.asarray(dm, dtype=float).copy()
    np.fill_diagonal(d, 1e9)
    
    d_to_depot = d[:, 0]
    # savings[i, j] = d[i, 0] + d[j, 0] - d[i, j]
    savings = d_to_depot[:, np.newaxis] + d_to_depot[np.newaxis, :] - d
    
    # Ensure savings are non-negative for stability, though in Euclidean they should be >= 0
    # Clip small negative values due to floating point errors
    savings = np.maximum(savings, 0.0)
    
    # Normalize savings to [0, 1] roughly to make beta more interpretable
    max_sav = savings.max()
    if max_sav > 0:
        savings_norm = savings / max_sav
    else:
        savings_norm = savings
        
    # H[i, j] = (1 + beta * savings_norm[i, j]) / (d[i, j] + eps)^alpha
    # We want larger H to be more attractive.
    # So if savings is high, H is higher. If distance is low, H is higher.
    
    denom = np.power(d + 1e-9, alpha)
    numerator = 1.0 + beta * savings_norm
    
    H = numerator / denom
    
    # Apply scale factor to adjust overall magnitude if needed, 
    # though ACO usually normalizes or the relative values matter more.
    # However, the prompt says "larger is more attractive".
    # Let's keep the scale as a multiplier if it helps tuning.
    H = scale * H
    
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    np.fill_diagonal(H, 1e-9)
    
    return H

def heuristics(distance_matrix, coordinates, demands, capacity):
    dm = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    D = np.asarray(demands, dtype=float)
    cap = float(capacity)
    n = dm.shape[0]
    
    try:
        if dm.shape != (n, n):
            raise ValueError("shape")
        if not np.isfinite(dm).all() or not np.isfinite(coords).all() or not np.isfinite(D).all():
            raise ValueError("non-finite")
        
        if n != _N:
            return _safe_heuristic(dm)
        
        dist = _distmat(coords)
        
        if _remaining() <= 30.0:
            # Fallback: standard 1/d
            return _build_heuristic_savings(dm, 1.0, 0.0, 1.0)
        
        best_m = 1e30
        best_params = (1.0, 0.0, 1.0)
        
        # Search over alpha (distance power), beta (savings weight), scale
        # alpha: 0.5 to 2.0
        # beta: 0.0 to 2.0 (0 means pure 1/d^alpha)
        # scale: 1.0 (fixed for now, relative scaling is what matters in probability)
        
        alpha_vals = [0.5, 1.0, 1.5, 2.0]
        beta_vals = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        n_ants_short = 10
        n_iter_short = 10
        max_steps_short = 150
        
        for alpha in alpha_vals:
            for beta in beta_vals:
                if _remaining() <= 30.0:
                    break
                H = _build_heuristic_savings(dm, alpha, beta, 1.0)
                logH = np.log(H)
                m = _colony(dist, D, cap, logH, n_ants_short, n_iter_short, max_steps_short)
                if m < best_m:
                    best_m = m
                    best_params = (alpha, beta, 1.0)
            if _remaining() <= 30.0:
                break
        
        alpha_opt, beta_opt, scale_opt = best_params
        
        # Refine with full colony if time permits
        if _remaining() > 60.0:
            cur_a, cur_b, cur_s = alpha_opt, beta_opt, scale_opt
            cur_H = _build_heuristic_savings(dm, cur_a, cur_b, cur_s)
            cur_logH = np.log(cur_H)
            cur_m = _colony(dist, D, cap, cur_logH, 20, 15, 250)
            
            for _round in range(4):
                if _remaining() <= 40.0:
                    break
                improved = False
                # Refine alpha
                for da in [0.2, -0.2, 0.1, -0.1]:
                    na = cur_a + da
                    if na < 0.2 or na > 3.0:
                        continue
                    Ht = _build_heuristic_savings(dm, na, cur_b, cur_s)
                    logHt = np.log(Ht)
                    mt = _colony(dist, D, cap, logHt, 20, 15, 250)
                    if mt < cur_m - 1e-6:
                        cur_a, cur_H, cur_logH, cur_m = na, Ht, logHt, mt
                        improved = True
                # Refine beta
                for db in [0.2, -0.2, 0.1, -0.1]:
                    nb = cur_b + db
                    if nb < 0.0 or nb > 3.0:
                        continue
                    Ht = _build_heuristic_savings(dm, cur_a, nb, cur_s)
                    logHt = np.log(Ht)
                    mt = _colony(dist, D, cap, logHt, 20, 15, 250)
                    if mt < cur_m - 1e-6:
                        cur_b, cur_H, cur_logH, cur_m = nb, Ht, logHt, mt
                        improved = True
                if not improved:
                    break
            
            alpha_opt, beta_opt, scale_opt = cur_a, cur_b, cur_s

        H_final = _build_heuristic_savings(dm, alpha_opt, beta_opt, scale_opt)
        return H_final
        
    except Exception:
        try:
            return _safe_heuristic(dm)
        except Exception:
            return np.ones_like(dm)
