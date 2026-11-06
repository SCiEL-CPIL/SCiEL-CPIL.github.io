"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute all arithmetic progressions with length >= 2
    aps = []
    for d in range(1, N // 2 + 1):
        for start in range(N - d):
            if start + d < N:
                indices = list(range(start, N, d))
                if len(indices) >= 2:
                    aps.append(indices)
    
    num_aps = len(aps)
    
    # Build the AP matrix as a numpy array of shape (num_aps, N) with int8
    M = np.zeros((num_aps, N), dtype=np.int8)
    for i, ap in enumerate(aps):
        for idx in ap:
            M[i, idx] = 1
    
    # Convert to float32 for fast matvec
    Mf = M.astype(np.float32)
    
    # For incremental updates: for each position, which APs contain it?
    pos_to_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for idx in ap:
            pos_to_aps[idx].append(i)
    # Convert to numpy arrays for fast indexing
    pos_to_aps_arr = [np.array(lst, dtype=np.int32) for lst in pos_to_aps]
    
    def compute_max_abs_sum(signs):
        """Compute maximum absolute sum over all APs using matrix-vector product."""
        sums = Mf.dot(signs.astype(np.float32))
        return int(np.max(np.abs(sums)))
    
    def compute_soft_max_loss(x, temperature=1.0):
        """Compute smooth soft-max approximation of max |AP sum|.
        
        Uses log-sum-exp for both positive and negative sides.
        """
        sums = Mf.dot(x.astype(np.float32))
        abs_sums = np.abs(sums)
        # Soft max: temperature * log(sum(exp(sums/T)))
        max_val = np.max(abs_sums)
        log_sum_exp = max_val + temperature * np.log(np.sum(np.exp((abs_sums - max_val) / temperature)))
        return float(log_sum_exp)
    
    def compute_soft_max_grad(x, temperature=1.0):
        """Compute gradient of soft-max loss."""
        sums = Mf.dot(x.astype(np.float32))
        abs_sums = np.abs(sums)
        # Soft-max weights: exp(abs_sums / T) / sum(exp(abs_sums / T))
        max_val = np.max(abs_sums)
        shifted = (abs_sums - max_val) / temperature
        weights = np.exp(shifted)
        weights /= np.sum(weights)
        # Gradient of soft-max of |s| w.r.t. x: weights * sign(s) * M^T
        sign_sums = np.sign(sums).astype(np.float32)
        sign_sums[sign_sums == 0] = 1.0
        grad_ap = weights * sign_sums  # gradient w.r.t. each AP sum
        grad_x = Mf.T.dot(grad_ap)
        return grad_x
    
    rng = random.Random(42)
    
    # Phase 1: Gradient-based continuous optimization with soft-max loss
    best_signs = None
    best_cost = float('inf')
    
    # Try multiple initializations
    for init_seed in [42, 0, 1, 2, 3, 7, 13, 99]:
        if time.time() >= deadline - 8.0:
            break
        
        rng_init = random.Random(init_seed)
        x = np.array([rng_init.uniform(-1, 1) for _ in range(N)], dtype=np.float64)
        
        # Adam optimizer with adaptive learning rate
        lr = 0.05
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = np.zeros(N, dtype=np.float64)
        v = np.zeros(N, dtype=np.float64)
        
        # Adaptive temperature: start high, decrease
        T_start = 2.0
        T_end = 0.3
        num_steps = 800
        
        for step in range(num_steps):
            if time.time() >= deadline - 10.0:
                break
            
            # Anneal temperature
            progress = step / max(num_steps - 1, 1)
            T = T_start + (T_end - T_start) * progress
            
            grad = compute_soft_max_grad(x, T)
            
            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1 ** (step + 1))
            v_hat = v / (1 - beta2 ** (step + 1))
            x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Project to [-1, 1]
            x = np.clip(x, -1.0, 1.0)
        
        # Discretize to ±1
        signs = np.sign(x).astype(np.int8)
        signs[signs == 0] = 1
        
        cost = compute_max_abs_sum(signs)
        if cost < best_cost:
            best_cost = cost
            best_signs = signs.copy()
    
    # Phase 2: Incremental discrete refinement
    if best_signs is not None:
        signs = best_signs.copy()
        current_cost = best_cost
    else:
        signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        current_cost = compute_max_abs_sum(signs)
        best_signs = signs.copy()
        best_cost = current_cost
    
    # Maintain AP sums for incremental updates
    ap_sums = Mf.dot(signs.astype(np.float32)).astype(np.float64)
    max_abs = float(np.max(np.abs(ap_sums)))
    
    # Greedy local search with incremental updates
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for idx in range(N):
            if time.time() >= deadline - 3.0:
                break
            
            old_sign = signs[idx]
            new_sign = -old_sign
            
            # Update affected AP sums incrementally
            affected = pos_to_aps_arr[idx]
            if len(affected) == 0:
                continue
            
            old_ap_sums = ap_sums[affected].copy()
            ap_sums[affected] += 2.0 * new_sign  # flip changes sum by 2*new_sign
            
            new_max_abs = float(np.max(np.abs(ap_sums)))
            
            if new_max_abs < max_abs:
                # Accept the flip
                signs[idx] = new_sign
                max_abs = new_max_abs
                current_cost = int(max_abs)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_signs = signs.copy()
                improved = True
            else:
                # Revert
                ap_sums[affected] = old_ap_sums
    
    # Phase 3: Simulated annealing with incremental updates for escaping local optima
    T = 3.0
    cooling_rate = 0.9997
    
    while time.time() < deadline - 2.0:
        if T < 0.05:
            break
        
        idx = rng.randrange(N)
        old_sign = signs[idx]
        new_sign = -old_sign
        
        affected = pos_to_aps_arr[idx]
        if len(affected) == 0:
            T *= cooling_rate
            continue
        
        old_ap_sums = ap_sums[affected].copy()
        ap_sums[affected] += 2.0 * new_sign
        new_max_abs = float(np.max(np.abs(ap_sums)))
        
        delta = new_max_abs - max_abs
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            signs[idx] = new_sign
            max_abs = new_max_abs
            current_cost = int(max_abs)
            if current_cost < best_cost:
                best_cost = current_cost
                best_signs = signs.copy()
        else:
            ap_sums[affected] = old_ap_sums
        
        T *= cooling_rate
    
    # Final greedy pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(N):
            if time.time() >= deadline:
                break
            old_sign = signs[idx]
            new_sign = -old_sign
            affected = pos_to_aps_arr[idx]
            if len(affected) == 0:
                continue
            old_ap_sums = ap_sums[affected].copy()
            ap_sums[affected] += 2.0 * new_sign
            new_max_abs = float(np.max(np.abs(ap_sums)))
            if new_max_abs < max_abs:
                signs[idx] = new_sign
                max_abs = new_max_abs
                current_cost = int(max_abs)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_signs = signs.copy()
                improved = True
            else:
                ap_sums[affected] = old_ap_sums
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
