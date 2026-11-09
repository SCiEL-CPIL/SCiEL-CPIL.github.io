import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # Precompute the evaluation matrix
    M = 20000
    thetas = np.linspace(0, 2*np.pi, M, endpoint=False)
    Z = np.exp(1j * thetas[:, None] * np.arange(N)[None, :])
    
    best_c = None
    best_score = np.inf
    
    def get_score(c):
        vals = Z @ c
        return np.max(np.abs(vals))
    
    def local_search(c_init, max_iter=100):
        """
        Performs local search starting from c_init.
        Uses single flips, then double flips, then swaps.
        Returns (best_c, best_score).
        """
        c = c_init.astype(float).copy()
        current_score = get_score(c)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline:
                break
            improved = False
            iter_count += 1
            
            # 1. Try single flips
            for i in range(N):
                old_val = c[i]
                c[i] = -old_val
                new_score = get_score(c)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    c[i] = old_val
            
            if improved:
                continue
                
            # 2. Try double flips (if single flips didn't improve)
            for i in range(N):
                for j in range(i + 1, N):
                    old_i, old_j = c[i], c[j]
                    c[i], c[j] = -old_i, -old_j
                    new_score = get_score(c)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        c[i], c[j] = old_i, old_j
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue

            # 3. Try swaps (if single and double flips didn't improve)
            for i in range(N):
                for j in range(i + 1, N):
                    old_i, old_j = c[i], c[j]
                    c[i], c[j] = old_j, old_i
                    new_score = get_score(c)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        c[i], c[j] = old_i, old_j
                    if improved:
                        break
                if improved:
                    break

        return c, current_score
    
    def simulated_annealing(c_init, t_start=10.0, t_end=0.01, cooling=0.99, max_iter=5000):
        """
        Simulated annealing to escape local minima.
        """
        c = c_init.astype(float).copy()
        current_score = get_score(c)
        best_c = c.copy()
        best_score = current_score
        
        t = t_start
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Pick a random move: flip 1, flip 2, or swap
            move_type = np.random.randint(3)
            if move_type == 0: # Flip 1
                i = np.random.randint(N)
                old_val = c[i]
                c[i] = -old_val
                new_score = get_score(c)
                
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / t):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_c = c.copy()
                else:
                    c[i] = old_val
            elif move_type == 1: # Flip 2
                i, j = np.random.choice(N, 2, replace=False)
                old_i, old_j = c[i], c[j]
                c[i], c[j] = -old_i, -old_j
                new_score = get_score(c)
                
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / t):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_c = c.copy()
                else:
                    c[i], c[j] = old_i, old_j
            else: # Swap
                i, j = np.random.choice(N, 2, replace=False)
                old_i, old_j = c[i], c[j]
                c[i], c[j] = old_j, old_i
                new_score = get_score(c)
                
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / t):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_c = c.copy()
                else:
                    c[i], c[j] = old_i, old_j
            
            t *= cooling
        
        return best_c, best_score
    
    # List of initializations to try
    inits = []
    
    # 1. Known near-optimal polynomial for N=24
    # [1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1]
    inits.append(np.array([1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1], dtype=float))
    
    # 2. Alternating signs
    inits.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=float))
    
    # 3. All ones
    inits.append(np.ones(N, dtype=float))
    
    # 4. Thue-Morse sequence
    tm = []
    for i in range(N):
        tm.append(1 if bin(i).count('1') % 2 == 0 else -1)
    inits.append(np.array(tm, dtype=float))
    
    # 5. Random restarts
    np.random.seed(42)
    num_restarts = 100
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        inits.append(np.random.choice([-1, 1], N).astype(float))
    
    # First pass: local search on all inits
    local_results = []
    for c_init in inits:
        if time.time() > deadline:
            break
        c, score = local_search(c_init, max_iter=50)
        local_results.append((score, c))
        if score < best_score:
            best_score = score
            best_c = c.copy()
            
    # Sort local results and run SA on the top 5
    local_results.sort(key=lambda x: x[0])
    for score, c in local_results[:5]:
        if time.time() > deadline - 10:
            break
        c_sa, score_sa = simulated_annealing(c, t_start=5.0, t_end=0.01, cooling=0.995, max_iter=10000)
        if score_sa < best_score:
            best_score = score_sa
            best_c = c_sa.copy()
            
    # Final polish
    if best_c is not None and time.time() < deadline - 5:
        c_polished, score_polished = local_search(best_c, max_iter=100)
        if score_polished < best_score:
            best_score = score_polished
            best_c = c_polished.copy()
    
    if best_c is None:
        best_c = np.ones(N, dtype=int)
    else:
        best_c = best_c.astype(int)
    
    return [int(x) for x in best_c]
