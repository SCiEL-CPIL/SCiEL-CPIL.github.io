import time
import random
import numpy as np

def construct():
    # Deadline: 55 seconds to be safe
    deadline = time.time() + 55.0
    
    # Target sums: 128 to 256 inclusive
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1
    
    # Precompute contribution masks: for each x in [0,128], which sums does x contribute to?
    # contrib[x] is a boolean array of length NUM_SUMS where contrib[x][i] = True if x+y contributes to sum MIN_SUM+i for some y in A
    # Actually, we need: for a fixed x, removing x decrements M at positions where x+a is in [128,256] for a in A, plus 2x if in range
    # Let's precompute for each x, the list of sum indices that x would affect when paired with each possible a
    
    # More useful: precompute for each x, the set of sums {x+a : a in [0,128]} that fall in [128,256]
    # But this depends on A. Instead, let's precompute the "partner" contribution.
    
    # Key insight: when we remove x from A, we decrement M[s] for each s = x+a where a in A (including a=x for 2x).
    # So the change is: for each a in A, decrement M[x+a - MIN_SUM] if in range.
    
    # Let's use a different approach: represent M as a numpy array and use vectorized updates.
    
    # Precompute: for each x in [0,128], the array of sum indices (offset by MIN_SUM) that x contributes to when paired with any element in [0,128]
    # Actually simpler: for each x, the possible sums are x+0, x+1, ..., x+128. Filter to [128,256].
    sum_indices_for_x = {}
    for x in range(129):
        indices = []
        for a in range(129):
            s = x + a
            if MIN_SUM <= s <= MAX_SUM:
                indices.append(s - MIN_SUM)
        sum_indices_for_x[x] = np.array(indices, dtype=np.int32)
    
    # Helper to calculate multiplicities for a given set A using vectorized approach
    def calc_mult(A):
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        A_list = sorted(A)
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            # a + a
            s = 2 * a
            if MIN_SUM <= s <= MAX_SUM:
                M[s - MIN_SUM] += 1
            # a + b for b > a
            for j in range(i + 1, n):
                b = A_list[j]
                s = a + b
                if MIN_SUM <= s <= MAX_SUM:
                    M[s - MIN_SUM] += 1
        return M
    
    # Helper: compute a smooth surrogate for the peak using log-sum-exp
    def smooth_peak(M, beta=10.0):
        """Compute a smooth approximation of max(M) using log-sum-exp."""
        M_float = M.astype(np.float64)
        max_M = np.max(M_float)
        if max_M == 0:
            return 0.0
        # log-sum-exp: max_M + log(sum(exp(M - max_M)))
        shifted = M_float - max_M
        log_sum = np.log(np.sum(np.exp(shifted)))
        return max_M + log_sum / beta
    
    # Helper: remove element x from A and update M incrementally
    def remove_from_A(x, A, M):
        """Remove x from A, update M. Returns (new_A, new_M) or (None, None) if invalid."""
        new_M = M.copy()
        # For each a in A: decrement M[x+a - MIN_SUM] if in range
        for a in A:
            s = x + a
            if MIN_SUM <= s <= MAX_SUM:
                new_M[s - MIN_SUM] -= 1
        # Check validity
        if np.min(new_M) < 1:
            return None, None
        new_A = A - {x}
        return new_A, new_M
    
    # Helper: add element x to A and update M incrementally
    def add_to_A(x, A, M):
        """Add x to A, update M. Returns (new_A, new_M)."""
        new_M = M.copy()
        for a in A:
            s = x + a
            if MIN_SUM <= s <= MAX_SUM:
                new_M[s - MIN_SUM] += 1
        s = 2 * x
        if MIN_SUM <= s <= MAX_SUM:
            new_M[s - MIN_SUM] += 1
        new_A = A | {x}
        return new_A, new_M
    
    # Better initial set: use a structured approach
    # Start with elements that cover the range well
    # Try multiple initial configurations
    initial_sets = [
        set(range(64, 129)),  # Original
        set(range(32, 97)),   # Shifted
        set(range(0, 129, 2)),  # Every other
        set(range(64, 129)) | set(range(0, 32, 2)),  # Mixed
        set(range(48, 113)),   # Another shift
        set(range(0, 129, 3)), # Every third
    ]
    
    best_A = None
    best_score = float('inf')
    best_M = None
    
    for init_set in initial_sets:
        M_init = calc_mult(init_set)
        if np.min(M_init) < 1:
            continue  # Skip invalid initial sets
        
        A = set(init_set)
        M = M_init
        current_score = int(np.max(M))
        
        if current_score < best_score:
            best_score = current_score
            best_A = set(A)
            best_M = M.copy()
    
    if best_A is None:
        # Fallback to original
        A = set(range(64, 129))
        M = calc_mult(A)
        best_A = set(A)
        best_M = M.copy()
        best_score = int(np.max(M))
    
    # Simulated Annealing with improved moves
    ALL_ELEMENTS = list(range(0, 129))
    ALL_ELEMENTS_SET = set(ALL_ELEMENTS)
    
    # We'll do multiple restarts to escape local optima
    max_restarts = 10
    restart_time = (deadline - time.time()) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
        
        # Initialize for this restart
        if restart == 0:
            A = set(best_A)
            M = best_M.copy()
            current_score = best_score
        else:
            # Start from a random perturbation of best_A
            A = set(best_A)
            M = best_M.copy()
            current_score = best_score
            
            # Perturb: randomly remove and add some elements
            num_perturb = random.randint(3, 10)
            for _ in range(num_perturb):
                if len(A) > 5:
                    x = random.choice(list(A))
                    new_A, new_M = remove_from_A(x, A, M)
                    if new_M is not None:
                        A = new_A
                        M = new_M
                        current_score = int(np.max(M))
                candidates = [y for y in ALL_ELEMENTS if y not in A]
                if candidates:
                    y = random.choice(candidates)
                    new_A, new_M = add_to_A(y, A, M)
                    A = new_A
                    M = new_M
                    current_score = int(np.max(M))
        
        T = 3.0
        cooling = 0.999
        beta = 10.0  # For smooth peak
        
        # Track smooth score for acceptance
        current_smooth = smooth_peak(M, beta)
        
        # Limit iterations per restart
        max_iters = 50000
        iters = 0
        
        while time.time() < deadline and iters < max_iters:
            iters += 1
            
            # Choose a move type with bias toward removal/swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Remove an element
                if len(A) <= 10:  # Don't make set too small
                    T *= cooling
                    continue
                
                # Prefer removing elements that contribute to high-multiplicity sums
                A_list = list(A)
                x = random.choice(A_list)
                
                new_A, new_M = remove_from_A(x, A, M)
                if new_M is not None:
                    score_new = int(np.max(new_M))
                    delta = score_new - current_score
                    
                    # Use smooth surrogate for acceptance
                    smooth_new = smooth_peak(new_M, beta)
                    delta_smooth = smooth_new - current_smooth
                    
                    if delta < 0 or random.random() < np.exp(-delta_smooth / max(T, 0.01)):
                        A = new_A
                        M = new_M
                        current_score = score_new
                        current_smooth = smooth_new
                        if score_new < best_score:
                            best_score = score_new
                            best_A = set(A)
                            best_M = M.copy()
            
            elif move_type < 0.8:
                # Swap: remove x, add y
                if len(A) == 0:
                    T *= cooling
                    continue
                
                x = random.choice(list(A))
                # Choose y not in A
                candidates = [y for y in ALL_ELEMENTS if y not in A]
                if not candidates:
                    T *= cooling
                    continue
                y = random.choice(candidates)
                
                # Simulate: remove x then add y
                temp_A, temp_M = remove_from_A(x, A, M)
                if temp_M is not None:
                    new_A, new_M = add_to_A(y, temp_A, temp_M)
                    score_new = int(np.max(new_M))
                    delta = score_new - current_score
                    
                    # Use smooth surrogate for acceptance
                    smooth_new = smooth_peak(new_M, beta)
                    delta_smooth = smooth_new - current_smooth
                    
                    if delta < 0 or random.random() < np.exp(-delta_smooth / max(T, 0.01)):
                        A = new_A
                        M = new_M
                        current_score = score_new
                        current_smooth = smooth_new
                        if score_new < best_score:
                            best_score = score_new
                            best_A = set(A)
                            best_M = M.copy()
            
            else:
                # Add an element (less common)
                candidates = [y for y in ALL_ELEMENTS if y not in A]
                if not candidates:
                    T *= cooling
                    continue
                y = random.choice(candidates)
                
                new_A, new_M = add_to_A(y, A, M)
                score_new = int(np.max(new_M))
                delta = score_new - current_score
                
                # Use smooth surrogate for acceptance
                smooth_new = smooth_peak(new_M, beta)
                delta_smooth = smooth_new - current_smooth
                
                # Adding usually increases max, so only accept with low probability
                if delta <= 0 or random.random() < np.exp(-delta_smooth / max(T, 0.01)) * 0.5:
                    A = new_A
                    M = new_M
                    current_score = score_new
                    current_smooth = smooth_new
            
            # Cool down
            T *= cooling
            if T < 0.05:
                T = 0.05
        
        # Final polish for this restart: try to remove more elements
        if time.time() < deadline:
            removed = True
            while removed and time.time() < deadline:
                removed = False
                A_list = list(A)
                random.shuffle(A_list)
                for x in A_list:
                    if x not in A:
                        continue
                    new_A, new_M = remove_from_A(x, A, M)
                    if new_M is not None:
                        score_new = int(np.max(new_M))
                        if score_new <= best_score:
                            A = new_A
                            M = new_M
                            current_score = score_new
                            if score_new < best_score:
                                best_score = score_new
                                best_A = set(A)
                                best_M = M.copy()
                            removed = True
    
    return list(best_A)

# EVOLVE-BLOCK-START
# The code above is the implementation. 
# Note: The task asks to keep the function names and signatures.
# I will place the code in the EVOLVE-BLOCK.
# EVOLVE-BLOCK-END
