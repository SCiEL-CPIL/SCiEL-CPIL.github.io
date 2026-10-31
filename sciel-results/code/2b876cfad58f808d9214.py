import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def is_sidon_set(indices):
    """Check if a set of indices forms a B2 (Sidon) set: all pairwise sums are distinct."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def build_addition_graph(N):
    """
    Build the addition graph on {0, ..., N-1}.
    Two indices a, b are adjacent if there exists another pair (c, d) with c+d = a+b,
    where (c,d) != (a,b) and (c,d) != (b,a). This includes self-sums: a+a collides
    with c+d if c+d=2a and {c,d} != {a,a}.
    An independent set in this graph is a Sidon set.
    """
    # For each sum s, collect all unordered pairs (i,j) with i<=j and i+j=s
    sum_to_pairs = {}
    for i in range(N):
        for j in range(i, N):
            s = i + j
            if s not in sum_to_pairs:
                sum_to_pairs[s] = []
            sum_to_pairs[s].append((i, j))
    
    # Build adjacency: a and b conflict if there's a sum s where both (a,b) and some other pair exist
    # Actually, we need: a and b are "incompatible" if adding b to a set containing a creates a collision.
    # More precisely: in a Sidon set, all pairwise sums (i,j) with i<=j must be distinct.
    # Two elements a, b (a<b) conflict if there exists c,d in the set such that a+b = c+d with {c,d} != {a,b}.
    # But for graph-based MIS, we want: an edge between a and b means they CANNOT both be in a Sidon set.
    # a and b cannot both be in a Sidon set if:
    #   - There exist c != a, c != b such that a + c = b + c' for some c'... 
    #   Actually simpler: a and b conflict if 2a = a + b (i.e., a=b, impossible for distinct) 
    #   or if there's a third element c such that a+c = b+c (impossible) or a+a = b+c or a+b = c+c etc.
    # 
    # The correct condition: a and b (a < b) are incompatible in a Sidon set if there exists c (c != a, c != b)
    # such that a + c = b + c (no), or a + a = b + c (c = 2a - b, need c != a, c != b, c >= 0, c < N),
    # or a + b = c + c (c = (a+b)/2, need c integer, c != a, c != b, c >= 0, c < N).
    #
    # Wait, that's not quite right either. In a Sidon set, the condition is that ALL pairwise sums are distinct.
    # If we have elements {a, b} plus potentially others, the constraint is that no two different pairs give the same sum.
    # For just {a, b}: sums are a+a, a+b, b+b. These are all distinct as long as a != b.
    # The conflict arises when we try to ADD a third element c. But for a graph-based approach,
    # we can define: a and b are "adjacent" (conflicting) if they CANNOT coexist in ANY Sidon set.
    # a and b cannot coexist if: there exists c such that {a,b,c} is not Sidon.
    # But that's not a pairwise property...
    #
    # Better approach: Use the standard result. A set is Sidon iff all pairwise sums i+j (i<=j) are distinct.
    # Two elements a, b (a<b) are "2-dependent" if 2a = a+b (impossible) or a+b = 2b (impossible) 
    # or 2a = 2b (impossible). So any two distinct elements can coexist in a Sidon set of size 2.
    # The constraint is higher-order. So a simple graph MIS won't directly work.
    #
    # Alternative: Use a greedy approach with backtracking, or use the addition table more directly.
    # Let me instead implement a greedy Sidon set construction with random restarts.
    return None

def generate_sidon_sets():
    """Generate several Sidon sets of various sizes within [0, N)."""
    sidon_sets = []
    base_sets = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 9, 15, 21],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 10],
        [0, 1, 4, 9],
        [0, 1, 4],
        [0, 1, 3],
    ]
    
    for s in base_sets:
        if all(0 <= x < N for x in s) and is_sidon_set(s):
            sidon_sets.append(list(s))
    
    np.random.seed(123)
    for _ in range(5000):
        k = np.random.randint(4, 11)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            found = False
            for existing in sidon_sets:
                if len(existing) == k and set(existing) == set(indices):
                    found = True
                    break
            if not found:
                sidon_sets.append(indices)
                if len(sidon_sets) > 50:
                    break
    
    return sidon_sets

def optimize_weights_for_support(support, beta=100.0, maxiter=1000, ftol=1e-12, n_restarts=3):
    """
    Optimize weights h_i for i in support to minimize max convolution coefficient.
    Uses smooth max surrogate: (1/beta) * log(sum(exp(beta * conv_k)))
    Parameterization: h = softmax(x) ensures positivity and sum=1.
    Multiple random restarts to escape local minima.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    def softmax(x):
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def objective_and_grad(x):
        h = softmax(x)
        # Full h vector
        h_full = np.zeros(N)
        h_full[support] = h
        
        conv = fftconvolve(h_full, h_full, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        # Objective: (1/beta) * log(sum(exp(beta * conv_k)))
        # = max_c + (1/beta) * log(sum(exp(beta * shifted)))
        obj_val = max_c + (1.0 / beta) * np.log(sum_exp)
        
        # Gradient computation
        # d obj / d h_j = sum_k weights_k * d conv_k / d h_j
        # d conv_k / d h_j = 2 * h_{k-j} (with h_m = 0 if m out of bounds)
        # So d obj / d h_j = 2 * sum_k weights_k * h_{k-j}
        
        grad_h = np.zeros(N)
        for m in support:
            hm = h_full[m]
            if hm > 1e-15:
                # grad_h[j] += 2 * hm * weights[m+j] for j in 0..N-1
                # weights has length 2N-1
                # m+j ranges from m to m+N-1
                grad_h[:N] += 2.0 * hm * weights[m : m + N]
        
        # Now chain rule: h = softmax(x)
        # h_i = exp(x_i) / sum(exp(x_j))
        # d h_i / d x_j = h_i * (delta_ij - h_j)
        # d obj / d x_j = sum_i (d obj / d h_i) * (d h_i / d x_j)
        # = sum_i grad_h_i * h_i * (delta_ij - h_j)
        # = grad_h_j * h_j - h_j * sum_i grad_h_i * h_i
        
        grad_h_support = grad_h[support]
        dot_prod = np.dot(grad_h_support, h)
        grad_x = h * (grad_h_support - dot_prod)
        
        return obj_val, grad_x
    
    # Multiple restarts with different initializations
    best_h_full = None
    best_sc = 0.0
    
    for restart in range(n_restarts):
        if restart == 0:
            x0 = np.zeros(k)
        else:
            # Random initialization in log-space
            x0 = np.random.randn(k) * 2.0
        
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                           jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-10})
            x_opt = res.x
            h_opt_local = softmax(x_opt)
            
            h_full = np.zeros(N)
            h_full[support] = h_opt_local
            sc = compute_ratio(h_full)
            
            if sc > best_sc:
                best_sc = sc
                best_h_full = h_full.copy()
        except Exception as e:
            pass
    
    if best_h_full is None:
        # Fallback to uniform
        h_full = np.zeros(N)
        h_full[support] = 1.0 / k
        best_sc = compute_ratio(h_full)
        best_h_full = h_full
    
    return best_h_full, best_sc

def local_search_swap(h, support, max_iters=100, deadline=None):
    """
    Local search: try adding, removing, or swapping elements in support to improve the ratio.
    Uses Sidon constraint to limit candidate additions.
    """
    k = len(support)
    best_h = h.copy()
    best_score = compute_ratio(best_h)
    best_support = sorted(support)
    
    def check_time():
        if deadline is not None:
            return time.time() < deadline
        return True
    
    for iteration in range(max_iters):
        if not check_time():
            break
        improved = False
        # Try adding a new element (only Sidon-compatible ones)
        candidates = []
        for new_idx in range(N):
            if new_idx in best_support:
                continue
            new_support = best_support + [new_idx]
            new_support.sort()
            if len(new_support) > 15:
                continue
            if is_sidon_set(new_support):
                candidates.append(new_idx)
        
        np.random.shuffle(candidates)
        for new_idx in candidates:
            if not check_time():
                break
            new_support = sorted(best_support + [new_idx])
            h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0, maxiter=500, n_restarts=2)
            if sc_new > best_score + 1e-6:
                best_h = h_new
                best_score = sc_new
                best_support = new_support
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing an element
        for remove_idx in list(best_support):
            if not check_time():
                break
            new_support = [x for x in best_support if x != remove_idx]
            if len(new_support) < 3:
                continue
            
            h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0, maxiter=500, n_restarts=2)
            if sc_new > best_score + 1e-6:
                best_h = h_new
                best_score = sc_new
                best_support = new_support
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping an element
        for old_idx in best_support:
            if not check_time():
                break
            swap_candidates = []
            for new_idx in range(N):
                if new_idx in best_support:
                    continue
                new_support = [x for x in best_support if x != old_idx]
                new_support.append(new_idx)
                new_support.sort()
                if is_sidon_set(new_support):
                    swap_candidates.append(new_idx)
            
            np.random.shuffle(swap_candidates)
            for new_idx in swap_candidates[:5]:  # Limit swaps per old element
                if not check_time():
                    break
                new_support = sorted([x for x in best_support if x != old_idx] + [new_idx])
                h_new, sc_new = optimize_weights_for_support(new_support, beta=50.0, maxiter=500, n_restarts=2)
                if sc_new > best_score + 1e-6:
                    best_h = h_new
                    best_score = sc_new
                    best_support = new_support
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_h, best_score, best_support

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    best_support = [0]
    
    sidon_sets = generate_sidon_sets()
    
    # Sort by size descending, then by score potential
    sidon_sets.sort(key=lambda x: (-len(x), sum(x)))
    
    # Phase 1: Optimize weights for each Sidon set
    for s in sidon_sets:
        if time.time() > deadline - 20.0:
            break
        
        # Use a range of betas, starting lower for faster convergence
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > deadline - 20.0:
                break
            
            h_opt, sc = optimize_weights_for_support(s, beta=beta, maxiter=800)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
                best_support = list(s)
    
    # Phase 2: Local search from the best solution
    if time.time() < deadline - 25.0:
        h_opt, sc_opt, support_opt = local_search_swap(best_h, best_support, max_iters=30, deadline=deadline - 15.0)
        if sc_opt > best_score:
            best_score = sc_opt
            best_h = h_opt.copy()
            best_support = support_opt
    
    # Phase 3: High-precision polish with multiple restarts
    if time.time() < deadline - 10.0:
        for beta in [150.0, 300.0, 600.0]:
            if time.time() > deadline - 10.0:
                break
            h_polish, sc_polish = optimize_weights_for_support(best_support, beta=beta, maxiter=2000, ftol=1e-14, n_restarts=5)
            if sc_polish > best_score:
                best_score = sc_polish
                best_h = h_polish.copy()
    
    # Phase 4: Final gradient polish on full simplex
    if time.time() < deadline - 5.0:
        try:
            def full_objective_and_grad(x):
                h = np.maximum(x, 0.0)
                h_sum = np.sum(h)
                if h_sum < 1e-15:
                    return 0.0, np.zeros(N)
                h = h / h_sum
                
                conv = fftconvolve(h, h, mode='full')
                max_c = np.max(conv)
                if max_c < 1e-15:
                    return 0.0, np.zeros(N)
                
                beta_final = 500.0
                shifted = conv - max_c
                exp_shifted = np.exp(beta_final * shifted)
                sum_exp = np.sum(exp_shifted)
                weights = exp_shifted / sum_exp
                
                obj_val = max_c + (1.0 / beta_final) * np.log(sum_exp)
                
                grad_h = np.zeros(N)
                for m in range(N):
                    hm = h[m]
                    if hm > 1e-15:
                        grad_h[:N] += 2.0 * hm * weights[m : m + N]
                
                # Project gradient to simplex
                grad_proj = grad_h - np.mean(grad_h)
                
                return obj_val, grad_proj
            
            x0 = best_h.copy()
            res = minimize(full_objective_and_grad, x0, method='L-BFGS-B', 
                          jac=True, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            h_final = np.maximum(res.x, 0.0)
            h_sum = np.sum(h_final)
            if h_sum > 1e-15:
                h_final = h_final / h_sum
            sc_final = compute_ratio(h_final)
            if sc_final > best_score:
                best_score = sc_final
                best_h = h_final.copy()
        except Exception:
            pass
    
    return list(best_h)
