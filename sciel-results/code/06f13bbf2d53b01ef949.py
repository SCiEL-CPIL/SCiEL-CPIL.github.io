"""Bermudan put exercise-policy research: Binned Backward Induction with Linear Interpolation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit an exercise policy for a Bermudan put option using binned backward induction
    with linear interpolation of continuation values to reduce discretization bias.
    """
    start_time = time.time()
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    n_paths, n_cols = training_paths.shape
    
    # Determine exercise indices
    start_idx = n_cols - n_dates
    
    dt = T / n_dates
    disc = np.exp(-r * dt)
    
    # Binning setup with more bins for better resolution
    all_s = training_paths[:, start_idx:]
    s_min = np.min(all_s)
    s_max = np.max(all_s)
    
    n_bins = 200
    s_min = max(s_min, 1e-6)
    log_s_min = np.log(s_min)
    log_s_max = np.log(s_max)
    
    # Create log-spaced bin edges
    log_edges = np.linspace(log_s_min, log_s_max, n_bins + 1)
    s_edges = np.exp(log_edges)
    
    # Bin centers for interpolation
    s_centers = np.exp(0.5 * (log_edges[:-1] + log_edges[1:]))
    
    # Payoff at last date
    s_last = training_paths[:, n_cols - 1]
    payoff_last = np.maximum(K - s_last, 0.0)
    V = payoff_last.copy()
    
    # Store continuation values per date (as arrays at bin centers)
    cont_vals_per_date = []
    
    # Backward induction
    for t_rel in range(n_dates - 2, -1, -1):
        col_idx = start_idx + t_rel
        s_t = training_paths[:, col_idx]
        
        # Determine bin for each path using log-space for better distribution
        log_s_t = np.log(np.maximum(s_t, 1e-10))
        # Find which bin each point falls into
        bin_idx = np.digitize(log_s_t, log_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        # Calculate average V for each bin
        sum_v = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        
        np.add.at(sum_v, bin_idx, V)
        np.add.at(counts, bin_idx, 1)
        
        # Handle empty bins with interpolation
        mask = counts > 0
        cont_vals = np.zeros(n_bins)
        cont_vals[mask] = sum_v[mask] / counts[mask]
        
        # For empty bins, use linear interpolation from neighbors
        empty_mask = ~mask
        if np.any(empty_mask):
            non_empty_idx = np.where(mask)[0]
            if len(non_empty_idx) >= 2:
                empty_idx = np.where(empty_mask)[0]
                for ei in empty_idx:
                    # Find nearest non-empty bins
                    left_idx = np.where(non_empty_idx < ei)[0]
                    right_idx = np.where(non_empty_idx > ei)[0]
                    
                    if len(left_idx) > 0 and len(right_idx) > 0:
                        li = non_empty_idx[left_idx[-1]]
                        ri = non_empty_idx[right_idx[0]]
                        # Linear interpolation
                        weight = (ei - li) / (ri - li) if ri > li else 0.5
                        cont_vals[ei] = (1 - weight) * cont_vals[li] + weight * cont_vals[ri]
                    elif len(left_idx) > 0:
                        li = non_empty_idx[left_idx[-1]]
                        cont_vals[ei] = cont_vals[li]
                    elif len(right_idx) > 0:
                        ri = non_empty_idx[right_idx[0]]
                        cont_vals[ei] = cont_vals[ri]
        
        cont_vals_per_date.append(cont_vals)
        
        # Calculate new V using linear interpolation
        payoff_t = np.maximum(K - s_t, 0.0)
        
        # Linear interpolation of continuation values
        # Get fractional position within bin
        log_s_t_clipped = np.clip(log_s_t, log_edges[0], log_edges[-1])
        # Find bin and fractional position
        bin_idx_interp = np.digitize(log_s_t_clipped, log_edges) - 1
        bin_idx_interp = np.clip(bin_idx_interp, 0, n_bins - 1)
        
        # Calculate fractional position within bin
        bin_start = log_edges[bin_idx_interp]
        bin_end = log_edges[bin_idx_interp + 1]
        frac = (log_s_t_clipped - bin_start) / (bin_end - bin_start)
        frac = np.clip(frac, 0.0, 1.0)
        
        # Interpolate between current bin and next bin
        next_bin = np.clip(bin_idx_interp + 1, 0, n_bins - 1)
        cont_path = (1 - frac) * cont_vals[bin_idx_interp] + frac * cont_vals[next_bin]
        
        disc_cont = disc * cont_path
        V = np.maximum(payoff_t, disc_cont)
    
    # Reverse to match time order
    cont_vals_per_date.reverse()
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 130:
        # Return a simple policy if running out of time
        class SimplePolicy:
            def __init__(self, K):
                self.K = K
            def exercise(self, current_state, time):
                if time < 1:
                    return np.zeros(len(current_state), dtype=bool)
                return current_state < self.K
        return SimplePolicy(K)
    
    class ExercisePolicy:
        def __init__(self, K, disc, s_edges, log_edges, n_bins, cont_vals_per_date, n_dates):
            self.K = K
            self.disc = disc
            self.s_edges = s_edges
            self.log_edges = log_edges
            self.n_bins = n_bins
            self.cont_vals = cont_vals_per_date
            self.n_dates = n_dates
            
        def exercise(self, current_state, time):
            if time < 1:
                return np.zeros(len(current_state), dtype=bool)
            
            t_rel = time - 1
            
            if t_rel < 0 or t_rel >= self.n_dates:
                return np.zeros(len(current_state), dtype=bool)
            
            if t_rel == self.n_dates - 1:
                return current_state < self.K
            
            # Use log-space for better interpolation
            log_state = np.log(np.maximum(current_state, 1e-10))
            log_state_clipped = np.clip(log_state, self.log_edges[0], self.log_edges[-1])
            
            # Find bin and fractional position
            bin_idx = np.digitize(log_state_clipped, self.log_edges) - 1
            bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            
            bin_start = self.log_edges[bin_idx]
            bin_end = self.log_edges[bin_idx + 1]
            frac = (log_state_clipped - bin_start) / (bin_end - bin_start)
            frac = np.clip(frac, 0.0, 1.0)
            
            # Interpolate between current bin and next bin
            next_bin = np.clip(bin_idx + 1, 0, self.n_bins - 1)
            cont_vals = self.cont_vals[t_rel]
            cont_path = (1 - frac) * cont_vals[bin_idx] + frac * cont_vals[next_bin]
            
            disc_cont = self.disc * cont_path
            payoff = np.maximum(self.K - current_state, 0.0)
            
            return payoff >= disc_cont
    
    return ExercisePolicy(K, disc, s_edges, log_edges, n_bins, cont_vals_per_date, n_dates)

# EVOLVE-BLOCK-END
