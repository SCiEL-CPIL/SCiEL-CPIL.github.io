"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 50

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    def compute_score(heights):
        """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
        h = np.array(heights)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        # Autoconvolution (linear, full mode)
        conv = np.convolve(h, h, mode='full')
        linf = np.max(conv)
        if linf < 1e-12:
            return 0.0
        return l2_sq / (l1 * linf)
    
    def is_sidon(lst):
        """Check if lst is a Sidon set (all pair sums distinct)."""
        s = set(lst)
        sums = set()
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sm = lst[i] + lst[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True
    
    def greedy_sidon_with_restarts(n, target_size, n_restarts=20):
        """Find a Sidon set using greedy with random restarts and randomized ordering."""
        best_set = []
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            # Random permutation of indices
            order = list(range(n))
            random.shuffle(order)
            sidon = []
            for i in order:
                trial = sidon + [i]
                if is_sidon(trial):
                    sidon.append(i)
                    if len(sidon) >= target_size:
                        break
            if len(sidon) > len(best_set):
                best_set = sidon
                if len(best_set) >= target_size:
                    break
        return best_set
    
    def addition_graph_max_independent_set(n, max_time=5.0):
        """
        Formulate as maximum independent set on addition graph.
        Two indices i, j are connected if their sum collides with another pair's sum.
        Use greedy with random restarts.
        """
        start_time = time.time()
        best_set = []
        
        for restart in range(50):
            if time.time() - start_time > max_time:
                break
            if time.time() > deadline - 10:
                break
            
            # Random ordering
            order = list(range(n))
            random.shuffle(order)
            
            selected = []
            used_sums = set()
            
            for i in order:
                # Check if adding i creates a sum collision
                valid = True
                for j in selected:
                    s = i + j
                    if s in used_sums:
                        valid = False
                        break
                if valid:
                    # Add all new sums involving i
                    for j in selected:
                        used_sums.add(i + j)
                    used_sums.add(2 * i)  # self-sum
                    selected.append(i)
            
            if len(selected) > len(best_set):
                best_set = selected
        
        return best_set
    
    def refine_heights(support, max_iters=200, tol=1e-8):
        """
        Given a support (indices with nonzero heights), optimize the actual height values.
        Use coordinate descent / gradient-free local search.
        """
        if len(support) == 0:
            return np.zeros(N)
        
        h = np.zeros(N)
        for i in support:
            h[i] = 1.0
        
        best_h = h.copy()
        best_score = compute_score(h)
        
        # Coordinate descent: perturb each height in support
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for idx in support:
                # Try small perturbations
                current_val = h[idx]
                for delta in [-0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 1.0, 2.0]:
                    trial_val = current_val + delta
                    if trial_val < 0:
                        continue
                    h[idx] = trial_val
                    s = compute_score(h)
                    if s > best_score + tol:
                        best_score = s
                        best_h = h.copy()
                        improved = True
                    else:
                        h[idx] = current_val
            
            if not improved:
                break
        
        return best_h
    
    # Initialize with single spike (theoretical optimum candidate)
    best_heights = np.zeros(N)
    best_heights[0] = 1.0
    best_score = compute_score(best_heights)
    
    # Also try uniform as fallback
    h_uniform = np.ones(N)
    s_uniform = compute_score(h_uniform)
    if s_uniform > best_score:
        best_score = s_uniform
        best_heights = h_uniform.copy()
    
    # Strategy 1: Use addition graph / Sidon set approach with various sizes
    for target_size in [3, 4, 5, 6, 7, 8, 10, 12]:
        if time.time() > deadline - 30:
            break
        
        # Greedy Sidon with restarts
        sidon = greedy_sidon_with_restarts(N, target_size, n_restarts=15)
        if len(sidon) >= 2:
            h = np.zeros(N)
            for i in sidon:
                h[i] = 1.0
            s = compute_score(h)
            if s > best_score:
                best_score = s
                best_heights = h.copy()
        
        # Try refining heights on this support
        if time.time() < deadline - 20:
            refined = refine_heights(sidon, max_iters=100)
            s = compute_score(refined)
            if s > best_score:
                best_score = s
                best_heights = refined.copy()
    
    # Strategy 2: Addition graph max independent set
    if time.time() < deadline - 25:
        add_graph_set = addition_graph_max_independent_set(N, max_time=8.0)
        if len(add_graph_set) >= 2:
            h = np.zeros(N)
            for i in add_graph_set:
                h[i] = 1.0
            s = compute_score(h)
            if s > best_score:
                best_score = s
                best_heights = h.copy()
            
            # Refine
            if time.time() < deadline - 15:
                refined = refine_heights(add_graph_set, max_iters=150)
                s = compute_score(refined)
                if s > best_score:
                    best_score = s
                    best_heights = refined.copy()
    
    # Strategy 3: Random sparse sets with refinement
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        # Random size between 2 and 10
        k = random.randint(2, 10)
        support = random.sample(range(N), k)
        h = np.zeros(N)
        for i in support:
            h[i] = 1.0
        s = compute_score(h)
        if s > best_score:
            best_score = s
            best_heights = h.copy()
        
        # Quick refinement
        if time.time() < deadline - 8:
            refined = refine_heights(support, max_iters=50)
            s = compute_score(refined)
            if s > best_score:
                best_score = s
                best_heights = refined.copy()
    
    # Final polish: refine the best support found
    if time.time() < deadline - 3:
        support = list(np.where(best_heights > 0)[0])
        if len(support) > 0:
            refined = refine_heights(support, max_iters=300, tol=1e-10)
            s = compute_score(refined)
            if s > best_score:
                best_score = s
                best_heights = refined.copy()
    
    # Ensure nonnegative and valid
    best_heights = np.maximum(best_heights, 0)
    if np.sum(best_heights) < 1e-12:
        return [1.0] * N
    
    return list(best_heights)
# EVOLVE-BLOCK-END
