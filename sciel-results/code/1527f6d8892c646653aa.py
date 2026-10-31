import numpy as np
import time
import random
import itertools
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    # Autoconvolution via FFT for efficiency
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    # Autoconvolution of length N with itself has length 2N-1
    # The FFT gives circular conv, but for zero-padded it's linear
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_peak)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # 1. Build the Addition Graph
    # Vertices: 0..N-1
    # Edge between i and j if the multiset of sums {i+j, 2i, 2j} (with multiplicity 
    # considerations for the convolution peak) creates a collision with 
    # sums from other pairs in the set.
    # Actually, for the "Sidon" property (all pair sums distinct), 
    # we want a set S such that all sums i+j (i<=j) are distinct.
    # This is equivalent to finding a Maximum Independent Set in a graph where 
    # vertices are indices, and an edge exists between i and j if 
    # adding j to a set containing i (and potentially others) creates a sum collision.
    # 
    # A simpler structural proxy: 
    # Two indices i, j are "incompatible" if 2i == 2j (impossible for i!=j in Z_N unless N even and i=j+N/2) 
    # or if they create a sum that is likely to collide.
    # 
    # Let's define a conflict graph G where an edge (i, j) exists if 
    # the set {i, j} already has a "bad" sum structure relative to the whole space? 
    # No, that's not local.
    #
    # Let's use the standard definition for Sidon sets in Z_N:
    # A set A is Sidon if all equations a1+a2=a3+a4 with a1<=a2, a3<=a4 have a1=a3, a2=a4.
    # This is hard to check locally for an independent set formulation directly on indices 
    # without knowing the rest of the set.
    #
    # Alternative Structural Strategy:
    # Instead of a dynamic graph, we use a **Greedy Randomized Independent Set** on a 
    # pre-defined "Conflict Graph" where edges represent direct sum collisions 
    # with *any* other pair.
    # 
    # Actually, a more effective "Structural Change" is to treat the support selection 
    # as a **Quadratic Unconstrained Binary Optimization (QUBO)** or simply 
    # a **Greedy Set Packing** problem where we maximize the number of unique sums.
    #
    # Let's implement a robust Greedy Sidon Set Constructor with Random Restarts, 
    # which is structurally different from the continuous local search of the seed.
    # The seed used a simple greedy with a fixed conflict threshold. 
    # We will use a more rigorous "Maximum Independent Set" approximation on 
    # the graph of "sum collisions".
    
    # Pre-compute all possible sums for pairs (i, j) with i <= j
    # Sum s = i + j mod N
    # We want to select a set of indices S such that the map (i, j) -> i+j is injective 
    # on S x S (with i<=j).
    
    # Conflict Graph Construction:
    # Vertices: indices 0..N-1
    # This is tricky because the conflict depends on the *set* S, not just pairs.
    # However, we can use a heuristic: 
    # Two indices i and j are "conflicting" if 2i == 2j (no) or if they are "close" 
    # in a way that creates small differences?
    
    # Let's stick to the most powerful structural change: 
    # **Parametric Family Search + Convex Optimization**
    # We generate candidate *supports* (binary vectors) using a diverse set of 
    # combinatorial heuristics (Sidon, Arithmetic Progressions, Random Sparse), 
    # and then for each support, we solve the convex problem:
    # Minimize ||h*h||_inf subject to sum(h)=1, h>=0, support(h) fixed.
    # This is a convex problem (minimize max of linear functions of h).
    # We can solve it efficiently using SLSQP or even simpler methods if the support is small.
    
    best_h = np.ones(N) / N
    best_score = compute_ratio(best_h)
    
    def optimize_weights_for_support(support_mask):
        """
        Given a binary support mask, optimize the weights of the non-zero elements
        to maximize the ratio.
        Ratio = sum(h^2) / (sum(h) * max(conv))
        Since sum(h) is fixed (scale invariance), we can set sum(h)=1.
        Maximize sum(h^2) / max(conv) is NOT convex.
        However, minimizing max(conv) subject to sum(h)=1, h>=0 IS convex.
        Minimizing max(conv) tends to flatten the peak, which increases the ratio 
        because the denominator decreases. The numerator sum(h^2) will also change.
        
        Actually, we want to MAXIMIZE the ratio.
        Let's use a direct optimization of the ratio using SLSQP, 
        but constrained to the support.
        """
        indices = np.where(support_mask > 0)[0]
        k = len(indices)
        if k == 0:
            return 0.0, np.zeros(N)
        
        def objective(w):
            # w are the weights for the non-zero indices
            h = np.zeros(N)
            h[indices] = w
            # Scale invariance: we can fix sum(h)=1 to reduce dimensionality, 
            # but SLSQP can handle it. Let's fix sum(w)=1 for stability.
            # Actually, let's just maximize the ratio directly.
            l1 = np.sum(h)
            if l1 < 1e-15:
                return 0.0
            l2_sq = np.sum(h*h)
            n_fft = 2 * N
            H = np.fft.fft(h, n=n_fft)
            conv = np.fft.ifft(H * H).real
            conv_peak = np.max(conv[:2*N-1])
            if conv_peak < 1e-15:
                return 0.0
            return - (l2_sq / (l1 * conv_peak)) # Minimize negative ratio
        
        def constraints(w):
            return np.sum(w) - 1.0 # Fix L1 norm to 1
        
        # Initial guess: uniform weights on support
        w0 = np.ones(k) / k
        
        # Bounds: w >= 0
        bounds = [(0, None)] * k
        
        # Constraints: sum(w) = 1
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        try:
            res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 100, 'ftol': 1e-6})
            if res.success:
                h_opt = np.zeros(N)
                h_opt[indices] = res.x
                score = -res.fun
                return score, h_opt
        except Exception:
            pass
        return 0.0, np.zeros(N)

    # Generate diverse structural candidates for the support
    candidates = []
    
    # A. Greedy Sidon Sets (Multiple Restarts)
    # The seed had a weak greedy. We implement a stronger one.
    for _ in range(100):
        if time.time() > deadline:
            break
        # Random shuffle of indices
        indices = list(range(N))
        random.shuffle(indices)
        selected = []
        used_sums = set()
        
        for idx in indices:
            # Check if adding idx creates sum collisions
            # Sums to check: idx+idx, idx+j for j in selected
            new_sums = [(2*idx) % N]
            for j in selected:
                new_sums.append((idx+j) % N)
            
            # If any of these sums are already in used_sums, it's a collision.
            # For a strict Sidon set, we require NO collisions.
            # But strict Sidon sets are very small (~sqrt(N)).
            # We might allow 1 collision for denser sets?
            # Let's try strict first, then relax if needed.
            
            if any(s in used_sums for s in new_sums):
                continue
                
            selected.append(idx)
            for s in new_sums:
                used_sums.add(s)
                
            if len(selected) >= 15: # Cap size for optimization speed
                break
                
        if len(selected) >= 3:
            mask = np.zeros(N)
            mask[selected] = 1.0
            candidates.append(mask)
            
    # B. Arithmetic Progressions (APs)
    # APs have structured sums.
    for start in range(N):
        if time.time() > deadline:
            break
        for length in range(3, 15):
            for step in range(1, N):
                mask = np.zeros(N)
                valid = True
                for i in range(length):
                    idx = (start + i * step) % N
                    if mask[idx] > 0:
                        valid = False
                        break
                    mask[idx] = 1.0
                if valid and np.sum(mask) > 0:
                    candidates.append(mask)
                    
    # C. Random Sparse Sets with Density Control
    for _ in range(100):
        if time.time() > deadline:
            break
        k = random.randint(5, 20)
        indices = np.random.choice(N, k, replace=False)
        mask = np.zeros(N)
        mask[indices] = 1.0
        candidates.append(mask)

    # Evaluate candidates
    for mask in candidates:
        if time.time() > deadline:
            break
        score, h_opt = optimize_weights_for_support(mask)
        if score > best_score:
            best_score = score
            best_h = h_opt

    # Final Polish: If we have time, try a few more random restarts of the optimization
    # on the best support found.
    if time.time() < deadline - 5:
        best_indices = np.where(best_h > 1e-6)[0]
        if len(best_indices) > 0:
            # Perturb the weights slightly and re-optimize
            mask = np.zeros(N)
            mask[best_indices] = 1.0
            score, h_opt = optimize_weights_for_support(mask)
            if score > best_score:
                best_score = score
                best_h = h_opt

    return best_h.tolist()

if __name__ == "__main__":
    # Simple test
    h = construct()
    print(f"Score: {compute_ratio(h)}")
