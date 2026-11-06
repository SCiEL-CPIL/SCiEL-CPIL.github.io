import random
import numpy as np
import time
import math

N = 256

def construct():
    deadline = time.time() + 55.0
    best_seq = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    # Precompute all APs
    ap_indices_list = []
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                indices = [a + i * d for i in range(l)]
                ap_indices_list.append(indices)
    
    num_aps = len(ap_indices_list)
    
    # Build incidence matrix A (num_aps x N) where A[ap_idx, pos] = 1 if pos in ap
    A = np.zeros((num_aps, N), dtype=np.float64)
    for ap_idx, indices in enumerate(ap_indices_list):
        for idx in indices:
            A[ap_idx, idx - 1] = 1.0
    
    # Map position to APs containing it (for fast updates)
    pos_to_ap = [[] for _ in range(N)]
    for ap_idx, indices in enumerate(ap_indices_list):
        for idx in indices:
            pos_to_ap[idx - 1].append(ap_idx)
    
    pos_to_ap_np = [np.array(lst, dtype=np.int32) if lst else np.empty(0, dtype=np.int32) for lst in pos_to_ap]
    
    def compute_initial_sums(s):
        # Use matrix-vector product for speed
        S = A @ s.astype(np.float64)
        return np.round(S).astype(np.int16)
    
    def build_hist(S):
        hist = np.zeros(N + 1, dtype=np.int32)
        abs_S = np.abs(S)
        np.add.at(hist, abs_S, 1)
        return hist, int(np.max(abs_S))
    
    def try_flip(j, s, S, hist, current_max, temp, rng):
        """Try flipping position j. Returns (accepted, new_max, delta_val)."""
        old_val = s[j]
        delta = -2 * old_val
        
        ap_idx_list = pos_to_ap_np[j]
        
        old_max = current_max
        max_new_abs_in_affected = 0
        
        for ap_idx in ap_idx_list:
            old_sum = S[ap_idx]
            new_sum = old_sum + delta
            
            old_abs = abs(old_sum)
            new_abs = abs(new_sum)
            
            hist[old_abs] -= 1
            hist[new_abs] += 1
            
            S[ap_idx] = new_sum
            
            if new_abs > max_new_abs_in_affected:
                max_new_abs_in_affected = new_abs
        
        if hist[old_max] == 0:
            if max_new_abs_in_affected > old_max:
                new_max = max_new_abs_in_affected
            else:
                scan_result = old_max - 1
                while scan_result > 0 and hist[scan_result] == 0:
                    scan_result -= 1
                new_max = max(max_new_abs_in_affected, scan_result)
        else:
            new_max = max(old_max, max_new_abs_in_affected)
        
        cost_diff = new_max - old_max
        
        if cost_diff <= 0:
            accept = True
        else:
            if temp > 0.01:
                p = math.exp(-cost_diff / temp)
                accept = (rng.random() < p)
            else:
                accept = False
        
        if accept:
            s[j] = -old_val
            return True, new_max, delta
        else:
            for ap_idx in ap_idx_list:
                new_sum = S[ap_idx]
                old_sum = new_sum - delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[new_abs] -= 1
                hist[old_abs] += 1
                
                S[ap_idx] = old_sum
            
            return False, current_max, delta
    
    def local_search(s, S, hist, current_max, deadline, rng, max_steps=50000):
        """Simulated annealing with acceptance criteria."""
        temp = 4.0
        cooling_rate = 0.998
        steps = 0
        
        while time.time() < deadline and steps < max_steps:
            j = rng.randint(0, N - 1)
            
            accepted, new_max, _ = try_flip(j, s, S, hist, current_max, temp, rng)
            
            if accepted:
                current_max = new_max
            
            temp *= cooling_rate
            steps += 1
            
            if temp < 0.001:
                break
        
        return s, S, hist, current_max
    
    def greedy_improve(s, S, hist, current_max, deadline, rng, max_rounds=10):
        """Greedy hill-climbing: repeatedly try all positions, accept any improvement."""
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            for j in range(N):
                old_max = current_max
                old_val = s[j]
                delta = -2 * old_val
                
                ap_idx_list = pos_to_ap_np[j]
                max_new_abs_in_affected = 0
                
                for ap_idx in ap_idx_list:
                    old_sum = S[ap_idx]
                    new_sum = old_sum + delta
                    new_abs = abs(new_sum)
                    if new_abs > max_new_abs_in_affected:
                        max_new_abs_in_affected = new_abs
                
                if hist[old_max] == 0:
                    if max_new_abs_in_affected > old_max:
                        new_max = max_new_abs_in_affected
                    else:
                        scan_result = old_max - 1
                        while scan_result > 0 and hist[scan_result] == 0:
                            scan_result -= 1
                        new_max = max(max_new_abs_in_affected, scan_result)
                else:
                    new_max = max(old_max, max_new_abs_in_affected)
                
                if new_max < current_max:
                    # Accept: actually apply the flip
                    for ap_idx in ap_idx_list:
                        old_sum = S[ap_idx]
                        new_sum = old_sum + delta
                        
                        old_abs = abs(old_sum)
                        new_abs = abs(new_sum)
                        
                        hist[old_abs] -= 1
                        hist[new_abs] += 1
                        
                        S[ap_idx] = new_sum
                    
                    s[j] = -old_val
                    current_max = new_max
                    improved = True
            
            if not improved:
                break
        
        return s, S, hist, current_max
    
    def perturb(s, S, hist, current_max, rng, num_flips=8):
        """Randomly flip some positions to escape local minima."""
        for _ in range(num_flips):
            j = rng.randint(0, N - 1)
            old_val = s[j]
            delta = -2 * old_val
            
            ap_idx_list = pos_to_ap_np[j]
            
            for ap_idx in ap_idx_list:
                old_sum = S[ap_idx]
                new_sum = old_sum + delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[old_abs] -= 1
                hist[new_abs] += 1
                
                S[ap_idx] = new_sum
            
            s[j] = -old_val
        
        # Recompute max
        abs_S = np.abs(S)
        current_max = int(np.max(abs_S))
        hist = np.zeros(N + 1, dtype=np.int32)
        np.add.at(hist, abs_S, 1)
        
        return s, S, hist, current_max
    
    def smooth_objective(s, beta=0.5):
        """Compute a smooth surrogate for the max absolute AP sum using log-sum-exp."""
        S = A @ s.astype(np.float64)
        abs_S = np.abs(S)
        # Log-sum-exp for numerical stability
        max_val = np.max(abs_S)
        if max_val == 0:
            return 0.0
        log_sum = max_val + np.log(np.sum(np.exp(abs_S - max_val)))
        return log_sum / beta
    
    def gradient_descent_continuous(continuous_s, deadline, rng, max_iters=2000):
        """
        Optimize a continuous vector using SGD with a smooth surrogate.
        The surrogate is the soft-max of |A @ s|, which is differentiable.
        We project the result to ±1 at the end.
        """
        s_cont = continuous_s.astype(np.float64).copy()
        lr = 0.1
        beta = 2.0  # temperature for soft-max
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute S = A @ s_cont
            S = A @ s_cont
            abs_S = np.abs(S)
            
            # Soft-max gradient: d/ds (soft_max(|S|)) 
            # soft_max(|S|) = (1/beta) * log(sum(exp(beta * |S|)))
            # d/dS_i (soft_max) = exp(beta * |S_i|) / sum(exp(beta * |S|))
            # d/ds = A^T @ (d/dS)
            
            # For numerical stability
            max_abs = np.max(abs_S)
            if max_abs == 0:
                break
            
            shifted = beta * (abs_S - max_abs)
            exp_shifted = np.exp(shifted)
            sum_exp = np.sum(exp_shifted)
            
            # Gradient w.r.t. S
            grad_S = exp_shifted / sum_exp
            
            # Gradient w.r.t. s: A^T @ grad_S * sign(S)
            # Note: d|S_i|/dS_i = sign(S_i)
            sign_S = np.sign(S)
            grad_S_full = grad_S * sign_S
            
            grad_s = A.T @ grad_S_full
            
            # Update
            s_cont -= lr * grad_s
            
            # Optionally, project towards ±1 softly
            # This encourages the solution to be close to ±1
            # We can use a tanh-like projection or just let it be
            # For now, keep it continuous but add a small regularization towards ±1
            # Actually, let's just do plain SGD for now
        
        # Discretize to ±1
        s_discrete = np.where(s_cont >= 0, 1, -1).astype(np.int8)
        return s_discrete
    
    # Multiple restarts with different strategies
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        
        # Initialize with random signs
        s = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        
        S = compute_initial_sums(s)
        hist, current_max = build_hist(S)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # First: greedy improvement to get a good starting point
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            s, S, hist, current_max = greedy_improve(s, S, hist, current_max, time.time() + remaining_time * 0.15, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Then: simulated annealing
        remaining_time = deadline - time.time()
        if remaining_time > 3.0:
            sa_deadline = time.time() + remaining_time * 0.55
            s, S, hist, current_max = local_search(s, S, hist, current_max, sa_deadline, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Try gradient-based approach on a continuous relaxation
        remaining_time = deadline - time.time()
        if remaining_time > 10.0:
            # Start from current discrete solution, convert to continuous
            s_cont = s.astype(np.float64) * 0.5  # Scale down
            
            gd_deadline = time.time() + remaining_time * 0.3
            s_gd = gradient_descent_continuous(s_cont, gd_deadline, rng, max_iters=500)
            
            S_gd = compute_initial_sums(s_gd)
            hist_gd, current_max_gd = build_hist(S_gd)
            
            if current_max_gd < best_score:
                best_score = current_max_gd
                best_seq = s_gd.copy()
            
            # Polish the GD result with greedy + SA
            if time.time() < deadline - 5.0:
                s_gd, S_gd, hist_gd, current_max_gd = greedy_improve(
                    s_gd, S_gd, hist_gd, current_max_gd, time.time() + (deadline - time.time()) * 0.2, rng
                )
                
                if current_max_gd < best_score:
                    best_score = current_max_gd
                    best_seq = s_gd.copy()
            
            if time.time() < deadline - 3.0:
                s_gd, S_gd, hist_gd, current_max_gd = local_search(
                    s_gd, S_gd, hist_gd, current_max_gd, time.time() + (deadline - time.time()) * 0.4, rng, max_steps=10000
                )
                
                if current_max_gd < best_score:
                    best_score = current_max_gd
                    best_seq = s_gd.copy()
        
        # Perturbation and re-optimization for escape from local minima
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            s, S, hist, current_max = perturb(s, S, hist, current_max, rng, num_flips=12)
            
            # Greedy + SA again
            if time.time() < deadline - 3.0:
                s, S, hist, current_max = greedy_improve(s, S, hist, current_max, time.time() + (deadline - time.time()) * 0.25, rng)
            
            if time.time() < deadline - 2.0:
                s, S, hist, current_max = local_search(s, S, hist, current_max, time.time() + (deadline - time.time()) * 0.65, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Try a few more restarts if time permits
        if time.time() > deadline - 3.0:
            break
    
    # Final polish on the best sequence found
    if best_seq is not None and time.time() < deadline - 2.0:
        s_final = best_seq.copy()
        S_final = compute_initial_sums(s_final)
        hist_final, current_max_final = build_hist(S_final)
        
        # Greedy polish
        s_final, S_final, hist_final, current_max_final = greedy_improve(
            s_final, S_final, hist_final, current_max_final, deadline - 1.0, rng, max_rounds=15
        )
        
        # Short SA polish
        if time.time() < deadline - 1.0:
            s_final, S_final, hist_final, current_max_final = local_search(
                s_final, S_final, hist_final, current_max_final, deadline - 0.5, rng, max_steps=10000
            )
        
        if current_max_final < best_score:
            best_score = current_max_final
            best_seq = s_final.copy()
    
    return [int(x) for x in best_seq]
