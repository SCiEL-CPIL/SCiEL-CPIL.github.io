import numpy as np
import scipy.optimize as opt
import time
from itertools import combinations

# Global cache for precomputed graph data and operator data
_graph_cache = {}
_operator_cache = {}

def _get_graph_data(n, density, seed):
    key = (n, density, seed)
    if key in _graph_cache:
        return _graph_cache[key]
    
    # Generate graph
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < density:
                edges.append((i, j))
    
    # Precompute cuts for all 2^n states
    num_states = 1 << n
    cuts = np.zeros(num_states, dtype=np.float64)
    for k in range(num_states):
        cut_count = 0
        for (i, j) in edges:
            if ((k >> i) & 1) != ((k >> j) & 1):
                cut_count += 1
        cuts[k] = cut_count
        
    # Find optimal cut (brute force)
    opt_cut = np.max(cuts)
    
    data = {
        'n': n,
        'edges': edges,
        'cuts': cuts,
        'opt_cut': opt_cut
    }
    _graph_cache[key] = data
    return data

def _get_operator_data(n):
    if n in _operator_cache:
        return _operator_cache[n]
    
    num_states = 1 << n
    # B = sum X_i
    # Construct B matrix
    B = np.zeros((num_states, num_states), dtype=complex)
    for i in range(n):
        for k in range(num_states):
            # Flip bit i
            k_flipped = k ^ (1 << i)
            B[k_flipped, k] += 1.0
            
    # Diagonalize B
    # B is Hermitian
    w, v = np.linalg.eigh(B)
    
    data = {
        'n': n,
        'w': w,
        'v': v
    }
    _operator_cache[n] = data
    return data

def _simulate_qaoa(n, params, graph_data, op_data):
    gamma1, beta1, gamma2, beta2 = params
    num_states = 1 << n
    # Initial state |+>^n
    state = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    
    # Step 1: exp(-i gamma1 C)
    phase1 = np.exp(-1j * gamma1 * graph_data['cuts'])
    state = state * phase1
    
    # Step 2: exp(-i beta1 B)
    v = op_data['v']
    w = op_data['w']
    
    c = v.conj().T @ state
    c = c * np.exp(-1j * beta1 * w)
    state = v @ c
    
    # Step 3: exp(-i gamma2 C)
    phase2 = np.exp(-1j * gamma2 * graph_data['cuts'])
    state = state * phase2
    
    # Step 4: exp(-i beta2 B)
    c = v.conj().T @ state
    c = c * np.exp(-1j * beta2 * w)
    state = v @ c
    
    # Expected Cut
    probs = np.abs(state)**2
    exp_cut = np.sum(probs * graph_data['cuts'])
    
    return exp_cut

def angles(n, edges, p):
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Precompute data for this specific graph
    num_states = 1 << n
    cuts = np.zeros(num_states, dtype=np.float64)
    for k in range(num_states):
        cut_count = 0
        for (i, j) in edges:
            if ((k >> i) & 1) != ((k >> j) & 1):
                cut_count += 1
        cuts[k] = cut_count
        
    opt_cut = np.max(cuts)
    if opt_cut == 0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
        
    graph_data = {
        'n': n,
        'edges': edges,
        'cuts': cuts,
        'opt_cut': opt_cut
    }
    
    op_data = _get_operator_data(n)
    
    def objective(params):
        if time.time() - start_time > deadline:
            raise TimeoutError
        g1, b1, g2, b2 = params
        exp_cut = _simulate_qaoa(n, [g1, b1, g2, b2], graph_data, op_data)
        ratio = exp_cut / opt_cut
        return -ratio # Minimize negative ratio
    
    # Strategy: Multi-start Local Optimization
    # 1. Generate a set of candidate initial points
    # 2. Evaluate them quickly
    # 3. Run Nelder-Mead from the best few
    
    # Standard good starting points for p=2 QAOA
    candidates = []
    candidates.append([0.35, 0.25, 0.35, 0.25])
    candidates.append([0.5, 0.5, 0.5, 0.5])
    candidates.append([0.2, 0.1, 0.2, 0.1])
    candidates.append([0.4, 0.3, 0.4, 0.3])
    
    best_x = None
    best_val = -np.inf
    
    # First pass: Evaluate static candidates
    for x0 in candidates:
        if time.time() - start_time > deadline:
            break
        try:
            val = -objective(x0)
            if val > best_val:
                best_val = val
                best_x = x0
        except TimeoutError:
            break
        except Exception:
            pass
            
    # Second pass: Random restarts to find better basin
    rng = np.random.default_rng(42)
    random_candidates = []
    num_random = 50 # Increased from 20
    
    for _ in range(num_random):
        if time.time() - start_time > deadline - 10:
            break
        # Random angles in [0, pi]
        x0 = rng.random(4) * np.pi
        try:
            val = -objective(x0)
            if val > best_val:
                best_val = val
                best_x = x0
            random_candidates.append((val, x0))
        except TimeoutError:
            break
        except Exception:
            pass
            
    # Sort random candidates by value
    random_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Polish the best random candidate and the best overall candidate
    points_to_optimize = []
    if best_x is not None:
        points_to_optimize.append(best_x)
        
    # Add top 3 random candidates if they are distinct
    for val, x0 in random_candidates[:3]:
        if np.linalg.norm(x0 - best_x) > 0.1:
            points_to_optimize.append(x0)
            
    # Add one more heuristic if distinct
    for x0 in candidates:
        if np.linalg.norm(x0 - best_x) > 0.1:
            points_to_optimize.append(x0)
            break

    # Run Nelder-Mead on these points
    for x0 in points_to_optimize:
        if time.time() - start_time > deadline - 5:
            break
        try:
            res = opt.minimize(objective, x0, method='Nelder-Mead', 
                               options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 500})
            if -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x
        except TimeoutError:
            pass
        except Exception:
            pass
            
    # If we still have time, do a very tight polish on the best solution
    if best_x is not None and time.time() - start_time < deadline - 2:
        try:
            res = opt.minimize(objective, best_x, method='Nelder-Mead', 
                               options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 1000})
            if -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x
        except TimeoutError:
            pass
        except Exception:
            pass

    if best_x is None:
        best_x = candidates[0]

    result = np.array([
        [best_x[0], best_x[1]],
        [best_x[2], best_x[3]]
    ])
    
    return result
