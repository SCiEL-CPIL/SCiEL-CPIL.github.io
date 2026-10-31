import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.signal import fftconvolve

N = 50
DEADLINE = 145.0  # safety margin before 150s

def construct():
    start = time.time()
    
    def elapsed():
        return time.time() - start
    
    def compute_ratio(h):
        """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
        h = np.asarray(h, dtype=np.float64)
        h1 = np.sum(h)
        if h1 < 1e-300:
            return 0.0
        h2 = np.sum(h * h)
        hh = fftconvolve(h, h, mode='full')
        hh_max = np.max(hh)
        if hh_max < 1e-300:
            return 0.0
        return h2 / (h1 * hh_max)
    
    def neg_ratio(h):
        """Negative ratio for minimization."""
        return -compute_ratio(h)
    
    def neg_ratio_constrained(params):
        """Negative ratio with L1 normalization. params are raw weights."""
        h = np.asarray(params, dtype=np.float64)
        h1 = np.sum(h)
        if h1 < 1e-300:
            return 0.0
        h_norm = h / h1
        return -compute_ratio(h_norm)
    
    best_h = np.ones(N) / N
    best_val = compute_ratio(best_h)
    
    # Strategy 1: Try sparse Sidon-like sets
    def find_sidon_set_greedy(n):
        """Find a Sidon set greedily."""
        used = set()
        set_elements = []
        for i in range(n):
            found = False
            for candidate in range(n):
                if candidate in used:
                    continue
                # Check if adding candidate creates a duplicate sum
                valid = True
                for j in range(len(set_elements)):
                    for k in range(len(set_elements)):
                        if j == k:
                            continue
                        if candidate + set_elements[j] == set_elements[k] + set_elements[j]:
                            pass # This check is for B2 sets, let's simplify
                # Simplified check: ensure all pair sums are unique
                new_sums = set()
                for a in set_elements + [candidate]:
                    for b in set_elements + [candidate]:
                        s = a + b
                        if s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                if valid:
                    set_elements.append(candidate)
                    used.add(candidate)
                    found = True
                    break
            if not found:
                break
        return set_elements
    
    # Try a few sparse structures
    # 1. Single spike
    h_single = np.zeros(N)
    h_single[N//2] = 1.0
    val = compute_ratio(h_single)
    if val > best_val:
        best_val = val
        best_h = h_single.copy()
    
    # 2. Sidon set
    sidon_set = find_sidon_set_greedy(10)
    if len(sidon_set) > 0:
        h_sidon = np.zeros(N)
        h_sidon[sidon_set] = 1.0 / len(sidon_set)
        val = compute_ratio(h_sidon)
        if val > best_val:
            best_val = val
            best_h = h_sidon.copy()
    
    # Strategy 2: Local search from best sparse structure
    def local_search_swap(h, max_iters=500):
        """Local search by swapping mass between indices."""
        h = h.copy()
        h1 = np.sum(h)
        h = h / h1
        current_val = compute_ratio(h)
        for _ in range(max_iters):
            if elapsed() > DEADLINE:
                break
            improved = False
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    # Try moving some mass from i to j
                    # Simple swap: h[i] -> h[j], h[j] -> h[i]
                    h_new = h.copy()
                    h_new[i], h_new[j] = h_new[j], h_new[i]
                    val_new = compute_ratio(h_new)
                    if val_new > current_val:
                        h = h_new
                        current_val = val_new
                        improved = True
            if not improved:
                break
        return h, current_val
    
    if elapsed() < DEADLINE - 10:
        h_opt, val_opt = local_search_swap(best_h)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Strategy 3: Optimize weights of a basis
    # Use a small set of basis vectors (e.g., peaks at various locations)
    # h = sum c_i * b_i, c_i >= 0
    # Let's try optimizing a sparse set of active indices with continuous weights
    
    # Identify top K indices from best_h
    K = 10
    top_indices = np.argsort(best_h)[-K:]
    
    def neg_ratio_basis(weights):
        """Negative ratio for basis weights."""
        h = np.zeros(N)
        h[top_indices] = weights
        return neg_ratio_constrained(h)
    
    if elapsed() < DEADLINE - 5:
        # Optimize weights for the top K indices
        x0 = np.full(K, 1.0 / K)
        bounds = [(0, 1)] * K
        try:
            res = minimize(neg_ratio_basis, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12})
            h_opt = np.zeros(N)
            h_opt[top_indices] = res.x
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Strategy 4: Differential Evolution for global search
    if elapsed() < DEADLINE - 20:
        try:
            # Use DE to find a good sparse structure
            # Represent h as log-weights to ensure non-negativity
            def neg_log_ratio(params):
                h = np.exp(params)
                h = h / np.sum(h)
                return -compute_ratio(h)
            
            bounds = [(-2, 2)] * N
            # DE can be slow, so use a small population and few generations
            res = differential_evolution(neg_log_ratio, bounds, 
                                         maxiter=20, popsize=10, 
                                         tol=1e-4, seed=42)
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish with L-BFGS-B
    if elapsed() < DEADLINE - 5:
        try:
            x0 = np.log(best_h + 1e-10)
            def neg_log_ratio(x):
                h = np.exp(x)
                h = h / np.sum(h)
                return -compute_ratio(h)
            
            res = minimize(neg_log_ratio, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-15})
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return list(best_h)

# EVOLVE-BLOCK-END
