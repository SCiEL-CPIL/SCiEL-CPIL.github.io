"""Root finding: Ill-conditioned - Adaptive Ensemble Solver."""
import math
import time
import heapq

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses an adaptive ensemble approach: maintains multiple candidate trajectories
    with different methods, selecting the next evaluation based on a heuristic score.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    
    def eval_f(x):
        """Counted evaluation of f."""
        eval_count[0] += 1
        return f(x)
    
    def time_ok():
        return time.time() - start_time < MAX_WALL

    def get_best():
        """Return the candidate with the smallest |f(x)|."""
        best_x = None
        best_fx = float('inf')
        for c in candidates:
            if c['valid'] and c['fx'] is not None:
                if abs(c['fx']) < best_fx:
                    best_fx = abs(c['fx'])
                    best_x = c['x']
        if best_x is None:
            return x0, fx0
        return best_x, best_fx

    # Candidate structure
    # Each candidate is a dict:
    # {
    #   'id': int,
    #   'method': str ('newton', 'secant', 'bisection', 'illinois'),
    #   'x': float, 'fx': float,
    #   'x2': float, 'fx2': float, # for secant/illinois
    #   'a': float, 'fa': float, 'b': float, 'fb': float, # for bisection/illinois
    #   'h': float, # step size for newton
    #   'last_fx': float, # previous |fx| for progress check
    #   'stall_count': int,
    #   'valid': bool,
    #   'score': float # heuristic score for priority
    # }
    
    x0 = float(x0)
    fx0 = eval_f(x0)
    
    if abs(fx0) <= TARGET_TOL:
        return x0
    
    # Initialize candidates
    candidates = []
    candidate_id = 0
    
    def add_candidate(method, **kwargs):
        nonlocal candidate_id
        c = {
            'id': candidate_id,
            'method': method,
            'x': None, 'fx': None,
            'x2': None, 'fx2': None,
            'a': None, 'fa': None, 'b': None, 'fb': None,
            'h': 1e-5 * max(1.0, abs(x0)),
            'last_fx': abs(fx0) if 'fx' not in kwargs else abs(kwargs.get('fx', fx0)),
            'stall_count': 0,
            'valid': True,
            'score': 0.0
        }
        c.update(kwargs)
        candidates.append(c)
        candidate_id += 1
        return c

    # 1. Newton from x0
    add_candidate('newton', x=x0, fx=fx0)
    
    # 2. Secant from x0 and x0+h
    h_sec = 1e-4 * max(1.0, abs(x0))
    x2 = x0 + h_sec
    if x2 != x0:
        fx2 = eval_f(x2)
        if abs(fx2) <= TARGET_TOL:
            return x2
        add_candidate('secant', x=x0, fx=fx0, x2=x2, fx2=fx2)
    
    # 3. Try to find a bracket for Bisection/Illinois
    # Expand from x0 to find sign change
    base_h = max(1e-6, abs(x0) * 1e-6)
    bracket_found = False
    for scale in [1, 10, 100, 1000, 10000, 100000]:
        if not time_ok() or eval_count[0] >= MAX_EVALS - 10:
            break
        h = base_h * scale
        if h < 1e-15:
            continue
        for sign in [1, -1]:
            if not time_ok() or eval_count[0] >= MAX_EVALS - 10:
                break
            x_c = x0 + sign * h
            if x_c == x0:
                continue
            fx_c = eval_f(x_c)
            if abs(fx_c) <= TARGET_TOL:
                return x_c
            if fx0 * fx_c < 0:
                a, b = (x0, x_c) if x0 < x_c else (x_c, x0)
                fa, fb = (fx0, fx_c) if x0 < x_c else (fx_c, fx0)
                add_candidate('bisection', a=a, fa=fa, b=b, fb=fb)
                add_candidate('illinois', a=a, fa=fa, b=b, fb=fb)
                bracket_found = True
                break
        if bracket_found:
            break

    # If no bracket found, add a few more Newton candidates with different h
    if not bracket_found:
        for h_mult in [1e-3, 1e-2, 1e-6]:
            h_new = h_mult * max(1.0, abs(x0))
            add_candidate('newton', x=x0, fx=fx0, h=h_new)

    # Main Loop: Adaptive Ensemble
    # At each step, pick the candidate with the best "promise"
    # Promise is based on:
    # 1. Current |fx| (lower is better)
    # 2. Progress in recent steps (ratio of |fx| to last_fx)
    # 3. Method-specific heuristics (e.g., bisection is safe, newton is fast but risky)
    
    while time_ok() and eval_count[0] < MAX_EVALS:
        # Calculate scores for all valid candidates
        # Score = -log(|fx| + eps) + progress_bonus - stall_penalty
        # We want to minimize score, so we use a max-heap with negative scores or min-heap with scores
        # Let's use a priority queue where lower score is higher priority.
        
        # Instead of a heap, since the number of candidates is small (< 10), we can just sort.
        # This avoids complex heap updates.
        
        active_candidates = [c for c in candidates if c['valid']]
        if not active_candidates:
            break
        
        # Update scores
        for c in active_candidates:
            if c['fx'] is None:
                c['score'] = float('inf')
                continue
            
            current_fx = abs(c['fx'])
            if current_fx <= TARGET_TOL:
                return c['x']
            
            # Progress: how much did |fx| decrease?
            if c['last_fx'] > 0:
                progress_ratio = current_fx / c['last_fx']
                if progress_ratio < 0.9:
                    progress_bonus = 1.0 - progress_ratio # 0.1 if 10% reduction
                else:
                    progress_bonus = 0.0
                    c['stall_count'] += 1
            else:
                progress_bonus = 0.0
            
            # Base score: log scale of |fx|
            base_score = math.log(current_fx + 1e-12)
            
            # Method penalty: Newton is risky, Bisection is safe but slow
            method_penalty = 0.0
            if c['method'] == 'newton':
                method_penalty = 0.5 * c['stall_count']
            elif c['method'] == 'secant':
                method_penalty = 0.2 * c['stall_count']
            elif c['method'] in ['bisection', 'illinois']:
                method_penalty = 0.0
            
            c['score'] = base_score - progress_bonus + method_penalty
        
        # Select candidate with lowest score
        best_c = min(active_candidates, key=lambda c: c['score'])
        
        # Perform one step for best_c
        method = best_c['method']
        
        if method == 'newton':
            x = best_c['x']
            fx = best_c['fx']
            h = best_c['h']
            
            if eval_count[0] >= MAX_EVALS - 1:
                break
            
            # Finite difference derivative
            fx_plus = eval_f(x + h)
            d = (fx_plus - fx) / h
            
            if d == 0 or not math.isfinite(d):
                # Try larger h
                h2 = h * 10
                if eval_count[0] >= MAX_EVALS - 1:
                    best_c['valid'] = False
                    break
                fx_plus2 = eval_f(x + h2)
                d = (fx_plus2 - fx) / h2
                if d == 0 or not math.isfinite(d):
                    best_c['valid'] = False
                    continue
            
            step = fx / d
            if not math.isfinite(step):
                best_c['valid'] = False
                continue
            
            # Damp step
            if abs(step) > 10 * max(1.0, abs(x)):
                step = math.copysign(10 * max(1.0, abs(x)), step)
            
            x_new = x - step
            if not math.isfinite(x_new):
                best_c['valid'] = False
                continue
            
            if eval_count[0] >= MAX_EVALS:
                break
            fx_new = eval_f(x_new)
            
            best_c['x'] = x_new
            best_c['fx'] = fx_new
            best_c['last_fx'] = abs(fx)
            best_c['h'] = h * (0.5 if abs(fx_new) > abs(fx) else 1.5)
            if best_c['h'] < 1e-15:
                best_c['h'] = 1e-15
            if best_c['h'] > 1e-2 * max(1.0, abs(x_new)):
                best_c['h'] = 1e-2 * max(1.0, abs(x_new))

        elif method == 'secant':
            x1, fx1 = best_c['x'], best_c['fx']
            x2, fx2 = best_c['x2'], best_c['fx2']
            
            if x1 == x2 or fx1 == fx2:
                best_c['valid'] = False
                continue
            
            d = (fx2 - fx1) / (x2 - x1)
            if d == 0 or not math.isfinite(d):
                best_c['valid'] = False
                continue
            
            x_new = x1 - fx1 / d
            if not math.isfinite(x_new):
                best_c['valid'] = False
                continue
            
            if eval_count[0] >= MAX_EVALS:
                break
            fx_new = eval_f(x_new)
            
            best_c['x'] = x2
            best_c['fx'] = fx2
            best_c['x2'] = x_new
            best_c['fx2'] = fx_new
            best_c['last_fx'] = abs(min(abs(fx1), abs(fx2)))

        elif method in ['bisection', 'illinois']:
            a, fa = best_c['a'], best_c['fa']
            b, fb = best_c['b'], best_c['fb']
            
            if fa * fb > 0:
                best_c['valid'] = False
                continue
            
            if method == 'bisection':
                mid = (a + b) / 2
            else: # illinois
                denom = fb - fa
                if denom == 0:
                    mid = (a + b) / 2
                else:
                    mid = a - fa * (b - a) / denom
                if mid <= a or mid >= b:
                    mid = (a + b) / 2
            
            if mid == a or mid == b:
                best_c['valid'] = False
                continue
            
            if eval_count[0] >= MAX_EVALS:
                break
            fm = eval_f(mid)
            
            if fa * fm <= 0:
                best_c['b'] = mid
                best_c['fb'] = fm
                if method == 'illinois':
                    best_c['fa'] = 0.5 * fa
            else:
                best_c['a'] = mid
                best_c['fa'] = fm
                if method == 'illinois':
                    best_c['fb'] = 0.5 * fb
            
            # Update x/fx for scoring (use the point with smaller |f|)
            if abs(fa) < abs(fb):
                best_c['x'] = a
                best_c['fx'] = fa
            else:
                best_c['x'] = b
                best_c['fx'] = fb
            
            best_c['last_fx'] = min(abs(fa), abs(fb))
            
            if abs(best_c['fx']) <= TARGET_TOL:
                return best_c['x']

        # Check if any candidate has converged
        for c in candidates:
            if c['valid'] and c['fx'] is not None and abs(c['fx']) <= TARGET_TOL:
                return c['x']
        
        # Prune stagnant candidates if we have too many
        if len([c for c in candidates if c['valid']]) > 6:
            # Remove the one with highest score and highest stall count
            active = [c for c in candidates if c['valid']]
            worst = max(active, key=lambda c: c['score'] + c['stall_count'])
            worst['valid'] = False

    # Return best found
    best_x, best_fx = get_best()
    return best_x

# EVOLVE-BLOCK-END
