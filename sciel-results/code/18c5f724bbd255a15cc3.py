import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    """
    Calculate the raw score R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    We normalize ||h||_1 = 1, so R(h) = ||h||_2^2 / ||h*h||_inf.
    The autoconvolution is h*h = conv(h, h).
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    
    # Autoconvolution
    # The continuous autoconvolution of the step function f(t) = sum h_i * 1_{[x_i, x_i+DELTA]}(t)
    # is (f*f)(t) = sum_{i,j} h_i h_j * (measure of intersection of [x_i, x_i+DELTA] and [t-x_j, t-x_j+DELTA]).
    # The peak of this continuous function is DELTA * max_k (sum_{i+j=k} h_i h_j).
    # The ratio is:
    # ||h||_2^2 = DELTA * sum(h_i^2)
    # ||h||_1 = DELTA * sum(h_i)
    # ||h*h||_inf = DELTA * max_k (sum_{i+j=k} h_i h_j)
    # Ratio = (DELTA * sum(h_i^2)) / (DELTA * sum(h_i) * DELTA * max_k (sum_{i+j=k} h_i h_j))
    #       = (1/DELTA) * (sum(h_i^2) / (sum(h_i) * max_k (sum_{i+j=k} h_i h_j)))
    # Since we normalize sum(h_i)=1, the ratio is (1/DELTA) * (sum(h_i^2) / max_k (sum_{i+j=k} h_i h_j)).
    # The factor 1/DELTA is constant, so we maximize the discrete part.
    
    h_conv = np.convolve(h_norm, h_norm, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-15:
        return 0.0
        
    return l2_sq / max_conv

def _evaluate_candidate(h):
    # Ensure non-negativity and normalization
    h = np.maximum(h, 0)
    if np.sum(h) < 1e-15:
        return 0.0
    h /= np.sum(h)
    return _score_raw(h)

def _generate_sidon_like_support(K, seed=0):
    """
    Generate a support set with minimal collisions in autoconvolution.
    We want to minimize the number of pairs (i, j) such that i+j = k for any k,
    or more precisely, we want the autoconvolution to be as flat as possible.
    A Sidon set (B2 sequence) ensures all pairwise sums are distinct.
    """
    rng = np.random.default_rng(seed)
    support = set()
    
    # Start with a random element
    support.add(rng.integers(0, N))
    
    while len(support) < K:
        # Try adding a new element
        added = False
        # Randomly shuffle candidates to break ties
        candidates = [i for i in range(N) if i not in support]
        rng.shuffle(candidates)
        
        for cand in candidates:
            new_support = support | {cand}
            # Check if all pairwise sums are distinct (Sidon property)
            sums = set()
            is_sidon = True
            for i in new_support:
                for j in new_support:
                    if i <= j:
                        s = i + j
                        if s in sums:
                            is_sidon = False
                            break
                        sums.add(s)
                if not is_sidon:
                    break
            
            if is_sidon:
                support = new_support
                added = True
                break
        
        if not added:
            # If no Sidon element found, just pick a random one that minimizes peak increase
            best_cand = None
            best_peak = np.inf
            for cand in candidates:
                new_support = support | {cand}
                h_test = np.zeros(N)
                for idx in new_support:
                    h_test[idx] = 1.0
                conv = np.convolve(h_test, h_test, mode='full')
                peak = np.max(conv)
                if peak < best_peak:
                    best_peak = peak
                    best_cand = cand
            
            if best_cand is not None:
                support.add(best_cand)
            else:
                break

    return sorted(list(support))

def _optimize_weights(support, h0=None):
    """
    Optimize weights for a fixed support to maximize score.
    Uses SLSQP on the non-smooth objective, with multiple restarts.
    """
    K = len(support)
    if K == 0:
        return np.zeros(N), 0.0
    
    if K == 1:
        h = np.zeros(N)
        h[support[0]] = 1.0
        return h, _score_raw(h)
    
    def objective(w):
        h = np.zeros(N)
        h[support] = w
        return -_score_raw(h)
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Multiple restarts
    for trial in range(10):
        if trial == 0:
            if h0 is not None:
                w0 = h0[support]
            else:
                w0 = np.ones(K) / K
        else:
            w0 = np.random.dirichlet(np.ones(K))
        
        bounds = [(0, 1)] * K
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        try:
            res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                w_opt = res.x
                h_opt = np.zeros(N)
                h_opt[support] = w_opt
                h_opt = np.maximum(h_opt, 0)
                if np.sum(h_opt) > 0:
                    h_opt /= np.sum(h_opt)
                score_opt = _score_raw(h_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # 1. Explore different sparsity levels K
    # The optimal K is likely small (Sidon sets are sparse).
    # We try K from 2 to 10.
    
    for K in range(2, 11):
        if time.time() > deadline - 5:
            break
        
        # Number of trials for each K
        num_trials = 20 if K < 6 else 5
        if time.time() > deadline - 20:
            num_trials = 2
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Generate a Sidon-like support
            support = _generate_sidon_like_support(K, seed=trial * 100 + K)
            
            # Optimize weights
            h_candidate, score_candidate = _optimize_weights(support)
            
            if score_candidate > best_score:
                best_score = score_candidate
                best_h = h_candidate.copy()
                
            # Also try a random support for diversity
            random_support = np.random.choice(N, K, replace=False)
            h_rand, score_rand = _optimize_weights(random_support)
            if score_rand > best_score:
                best_score = score_rand
                best_h = h_rand.copy()

    # 2. Local Refinement
    # Try adding/removing elements from the best support
    support_best = np.where(best_h > 1e-8)[0]
    
    # Try adding one more element
    if len(support_best) < N - 1:
        for pos in range(N):
            if pos in support_best:
                continue
            new_support = list(support_best) + [pos]
            h_new, score_new = _optimize_weights(new_support, h0=best_h)
            if score_new > best_score:
                best_score = score_new
                best_h = h_new.copy()
                support_best = np.where(best_h > 1e-8)[0]
                break # Restart search with new support
                
    # Try removing one element
    if len(support_best) > 2:
        for pos in support_best:
            new_support = [p for p in support_best if p != pos]
            h_new, score_new = _optimize_weights(new_support, h0=best_h)
            if score_new > best_score:
                best_score = score_new
                best_h = h_new.copy()
                support_best = np.where(best_h > 1e-8)[0]
                break

    # 3. Global Optimization with Differential Evolution on the full vector
    # This is a fallback for if the local search gets stuck.
    # We use a low population size to fit within time limits.
    if time.time() < deadline - 10:
        def de_objective(h_flat):
            return -_evaluate_candidate(h_flat)
        
        # Bounds: [0, 1] for each element
        bounds = [(0, 1)] * N
        # Initial population: mix of random and best_h
        # DE requires a population, so we let it generate, but we can seed it.
        # To be safe, we just run a short DE.
        try:
            res = differential_evolution(de_objective, bounds, maxiter=50, popsize=15, tol=1e-5, seed=42, polish=True, init='sobol')
            if res.success:
                h_de = res.x
                h_de = np.maximum(h_de, 0)
                if np.sum(h_de) > 0:
                    h_de /= np.sum(h_de)
                score_de = _score_raw(h_de)
                if score_de > best_score:
                    best_score = score_de
                    best_h = h_de
        except Exception:
            pass

    # Final Polish
    if time.time() < deadline - 2:
        support_final = np.where(best_h > 1e-8)[0]
        if len(support_final) > 0:
            h_final, score_final = _optimize_weights(support_final, h0=best_h)
            if score_final > best_score:
                best_score = score_final
                best_h = h_final

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(f"Score: {_score_raw(h)}")
