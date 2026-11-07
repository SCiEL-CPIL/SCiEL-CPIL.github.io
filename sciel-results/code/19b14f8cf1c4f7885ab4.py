import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0

def _run_aco_fast(D, eta, n_ants=20, n_iter=30, seed=42):
    """Run ACO with given heuristic eta, return best tour length."""
    n = D.shape[0]
    rng = np.random.RandomState(seed)
    tau = np.ones((n, n))
    best_len = float('inf')
    
    for it in range(n_iter):
        all_paths = []
        all_lens = []
        
        for ant in range(n_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path = [0]
            tour_len = 0.0
            
            for step in range(1, n):
                available = np.where(~visited)[0]
                probs = tau[current, available] * eta[current, available]
                total = probs.sum()
                
                if total <= 0:
                    next_node = available[0]
                else:
                    probs = probs / total
                    r = rng.random()
                    cumsum = np.cumsum(probs)
                    next_node = available[np.searchsorted(cumsum, r)]
                
                tour_len += D[current, next_node]
                current = next_node
                visited[current] = True
                path.append(current)
            
            tour_len += D[current, 0]
            path.append(0)
            
            if tour_len < best_len:
                best_len = tour_len
            all_paths.append(path)
            all_lens.append(tour_len)
        
        tau *= 0.9
        for ant_idx in range(n_ants):
            l = all_lens[ant_idx]
            if l <= 0:
                continue
            reward = 1.0 / l
            path = all_paths[ant_idx]
            for k in range(len(path) - 1):
                i = path[k]
                j = path[k+1]
                tau[i, j] += reward
    
    return best_len

def _make_eta(D, p):
    n = D.shape[0]
    eta = np.ones((n, n))
    mask = D > 0
    eta[mask] = 1.0 / np.power(D[mask], p)
    eta[~mask] = 1e-9
    return eta

def heuristics(distance_matrix):
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    
    if time.time() > _DEADLINE:
        return _make_eta(D, 1.0)
    
    search_deadline = time.time() + 85.0
    
    candidates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    
    best_p = 1.0
    best_score = float('inf')
    
    for p in candidates:
        if time.time() > search_deadline:
            break
        try:
            eta = _make_eta(D, p)
            score = _run_aco_fast(D, eta, seed=42)
            if score < best_score:
                best_score = score
                best_p = p
        except Exception:
            continue
    
    # Refine
    if time.time() < search_deadline - 15:
        step = 0.25
        current_p = best_p
        for _ in range(20):
            if time.time() > search_deadline:
                break
            improved = False
            for delta_p in [-step, step]:
                new_p = current_p + delta_p
                if new_p <= 0.1:
                    continue
                try:
                    eta = _make_eta(D, new_p)
                    score = _run_aco_fast(D, eta, seed=42)
                    if score < best_score:
                        best_score = score
                        best_p = new_p
                        current_p = new_p
                        improved = True
                        break
                except Exception:
                    continue
            if not improved:
                step *= 0.7
            if step < 0.01:
                break
    
    if time.time() < search_deadline - 5:
        step = 0.05
        current_p = best_p
        for _ in range(10):
            if time.time() > search_deadline:
                break
            improved = False
            for delta_p in [-step, step]:
                new_p = current_p + delta_p
                if new_p <= 0.05:
                    continue
                try:
                    eta = _make_eta(D, new_p)
                    score = _run_aco_fast(D, eta, seed=42)
                    if score < best_score:
                        best_score = score
                        best_p = new_p
                        current_p = new_p
                        improved = True
                        break
                except Exception:
                    continue
            if not improved:
                break
    
    return _make_eta(D, best_p)
