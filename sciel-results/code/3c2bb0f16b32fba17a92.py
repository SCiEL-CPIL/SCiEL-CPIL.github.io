"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for evaluation overhead
    
    # Generate all APs (length > 1)
    ap_indices = []
    for start in range(n):
        for step in range(1, n):
            seq = []
            cur = start
            while cur < n:
                seq.append(cur)
                cur += step
            if len(seq) > 1:
                ap_indices.append(np.array(seq, dtype=np.int32))
    
    num_aps = len(ap_indices)
    
    # Build M for L2 gradient: M[i, j] = count of APs containing i and j
    M = np.zeros((n, n), dtype=np.float64)
    for idx_arr in ap_indices:
        M[np.ix_(idx_arr, idx_arr)] += 1.0
        
    # Precompute masks for local search
    # mask[i] is a boolean array of length num_aps, True if AP contains index i
    masks = []
    for i in range(n):
        mask = np.zeros(num_aps, dtype=bool)
        for k, idx_arr in enumerate(ap_indices):
            if i in idx_arr:
                mask[k] = True
        masks.append(mask)
        
    def evaluate_discrete(signs):
        """Calculate the maximum absolute AP sum for a discrete sign sequence."""
        # Compute all sums at once using matrix multiplication if we had a dense matrix, 
        # but we have a list of arrays. 
        # We can compute sums efficiently by iterating, but for local search we want speed.
        # For initial evaluation, we can just sum.
        sums = np.zeros(num_aps)
        for k, idx_arr in enumerate(ap_indices):
            sums[k] = np.sum(signs[idx_arr])
        return np.max(np.abs(sums))
    
    def local_search(signs, time_limit):
        """
        Intensive local search using precomputed masks.
        Returns improved signs and score.
        """
        # Initialize sums
        sums = np.zeros(num_aps)
        for k, idx_arr in enumerate(ap_indices):
            sums[k] = np.sum(signs[idx_arr])
        
        current_max = np.max(np.abs(sums))
        improved = True
        
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    break
                
                # Calculate new sums if we flip sign at i
                # delta = -2 * signs[i]
                delta = -2.0 * signs[i]
                
                # Update sums for APs containing i
                mask = masks[i]
                new_sums = sums.copy()
                new_sums[mask] += delta
                
                new_max = np.max(np.abs(new_sums))
                
                if new_max < current_max:
                    # Accept flip
                    signs[i] *= -1
                    sums = new_sums
                    current_max = new_max
                    improved = True
        
        return signs, current_max

    best_signs = None
    best_score = float('inf')
    
    # Try multiple random restarts
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 10.0: # Reserve 10s for local search
            break
            
        # Initialize x
        x = np.random.uniform(-1, 1, n)
        
        # SGD parameters
        lr = 0.05
        lam = 50.0 # Strong regularization
        momentum = np.zeros(n)
        beta = 0.9
        steps = 200
        
        for t in range(steps):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Gradient of L2 loss: 2 * M * x
            grad_L2 = 2.0 * M.dot(x)
            
            # Gradient of regularization: 4 * lam * x * (x^2 - 1)
            grad_reg = 4.0 * lam * x * (x**2 - 1.0)
            
            grad = grad_L2 + grad_reg
            
            # Momentum update
            momentum = beta * momentum + (1 - beta) * grad
            x -= lr * momentum
            
            # Project to [-1, 1]
            x = np.clip(x, -1.0, 1.0)
            
        # Discretize
        signs = np.sign(x)
        signs[signs == 0] = 1.0
        
        # Evaluate
        score = evaluate_discrete(signs)
        
        if score < best_score:
            best_score = score
            best_signs = signs.copy()
            
        # Run local search on the best signs found so far (or current if it's the best)
        # We run local search on the current candidate if it's good, or on the global best
        # To save time, we only run local search if we have time
        if time.time() - start_time < time_limit - 2.0:
            # Use the current signs for local search if it's the best, otherwise use best_signs
            # Actually, let's just run local search on the current candidate
            # And update global best if it improves
            ls_signs, ls_score = local_search(signs.copy(), time_limit - 1.0)
            
            if ls_score < best_score:
                best_score = ls_score
                best_signs = ls_signs.copy()
                
            # Also try local search on the global best if it's different?
            # Might be too slow. Let's stick to one local search per restart.
            # But if the current one wasn't the best, we might want to refine the best one.
            # Let's just refine the current one.
            
    # Final check
    if best_signs is None:
        # Fallback: random signs
        best_signs = np.random.choice([-1, 1], n)
    
    return [int(s) for s in best_signs]

# EVOLVE-BLOCK-END
