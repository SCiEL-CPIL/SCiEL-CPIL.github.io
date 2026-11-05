"""Bermudan put exercise-policy research: Binned Backward Induction with Savitzky-Golay Smoothing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import savgol_filter

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
    
    # Savitzky-Golay parameters for smoothing
    # Window must be odd; use 11 with polynomial order 3
    sg_window = min(11, n_bins - 1) if n_bins > 2 else 0
    if sg_window % 2 == 0:
        sg_window -= 1
    sg_polyorder = 3
    
    def smooth_cont_values(cv, counts):
        """Apply Savitzky-Golay smoothing, preserving zeros for empty bins."""
        if sg_window < 3 or len(cv) <= sg_window:
            return cv
        # Only smooth where we have data
        mask = counts > 0
        if np.sum(mask) < sg_window:
            return cv
        smoothed = cv.copy()
        # Apply SG filter to the full array (empty bins are 0, will be fixed after)
        try:
            smoothed = savgol_filter(cv, sg_window, sg_polyorder)
        except Exception:
            smoothed = cv.copy()
        # Restore zeros for empty bins (they should remain 0 for interpolation purposes)
        smoothed[~mask] = 0.0
        return smoothed
    
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
        
        # Apply Savitzky-Golay smoothing to reduce MC noise
        cont_vals = smooth_cont_values(cont_vals, counts)
        
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
    
    # High-precision polish pass: re-run backward induction with finer bins
    # using the smoothed continuation values as a guide for bin initialization
    elapsed = time.time() - start_time
    if elapsed < 100:  # Safety margin: 150s limit, need ~50s for polish + policy creation
        n_bins_fine = 400
        log_edges_fine = np.linspace(log_s_min, log_s_max, n_bins_fine + 1)
        s_edges_fine = np.exp(log_edges_fine)
        
        # Payoff at last date
        s_last_fine = training_paths[:, n_cols - 1]
        payoff_last_fine = np.maximum(K - s_last_fine, 0.0)
        V_fine = payoff_last_fine.copy()
        
        cont_vals_fine_per_date = []
        
        for t_rel in range(n_dates - 2, -1, -1):
            col_idx = start_idx + t_rel
            s_t_fine = training_paths[:, col_idx]
            
            log_s_t_fine = np.log(np.maximum(s_t_fine, 1e-10))
            bin_idx_fine = np.digitize(log_s_t_fine, log_edges_fine) - 1
            bin_idx_fine = np.clip(bin_idx_fine, 0, n_bins_fine - 1)
            
            sum_v_fine = np.zeros(n_bins_fine)
            counts_fine = np.zeros(n_bins_fine)
            
            np.add.at(sum_v_fine, bin_idx_fine, V_fine)
            np.add.at(counts_fine, bin_idx_fine, 1)
            
            mask_fine = counts_fine > 0
            cont_vals_fine = np.zeros(n_bins_fine)
            cont_vals_fine[mask_fine] = sum_v_fine[mask_fine] / counts_fine[mask_fine]
            
            # Fill empty bins
            empty_mask_fine = ~mask_fine
            if np.any(empty_mask_fine):
                non_empty_idx_fine = np.where(mask_fine)[0]
                if len(non_empty_idx_fine) >= 2:
                    empty_idx_fine = np.where(empty_mask_fine)[0]
                    for ei in empty_idx_fine:
                        left_idx_fine = np.where(non_empty_idx_fine < ei)[0]
                        right_idx_fine = np.where(non_empty_idx_fine > ei)[0]
                        
                        if len(left_idx_fine) > 0 and len(right_idx_fine) > 0:
                            li = non_empty_idx_fine[left_idx_fine[-1]]
                            ri = non_empty_idx_fine[right_idx_fine[0]]
                            weight = (ei - li) / (ri - li) if ri > li else 0.5
                            cont_vals_fine[ei] = (1 - weight) * cont_vals_fine[li] + weight * cont_vals_fine[ri]
                        elif len(left_idx_fine) > 0:
                            li = non_empty_idx_fine[left_idx_fine[-1]]
                            cont_vals_fine[ei] = cont_vals_fine[li]
                        elif len(right_idx_fine) > 0:
                            ri = non_empty_idx_fine[right_idx_fine[0]]
                            cont_vals_fine[ei] = cont_vals_fine[ri]
            
            # Smooth the fine-grained continuation values
            sg_window_fine = min(11, n_bins_fine - 1) if n_bins_fine > 2 else 0
            if sg_window_fine % 2 == 0:
                sg_window_fine -= 1
            if sg_window_fine >= 3 and np.sum(mask_fine) >= sg_window_fine:
                try:
                    cont_vals_fine = savgol_filter(cont_vals_fine, sg_window_fine, sg_polyorder)
                    cont_vals_fine[~mask_fine] = 0.0
                except Exception:
                    pass
            
            cont_vals_fine_per_date.append(cont_vals_fine)
            
            # Calculate new V using linear interpolation on fine grid
            payoff_t_fine = np.maximum(K - s_t_fine, 0.0)
            
            log_s_t_fine_clipped = np.clip(log_s_t_fine, log_edges_fine[0], log_edges_fine[-1])
            bin_idx_interp_fine = np.digitize(log_s_t_fine_clipped, log_edges_fine) - 1
            bin_idx_interp_fine = np.clip(bin_idx_interp_fine, 0, n_bins_fine - 1)
            
            bin_start_fine = log_edges_fine[bin_idx_interp_fine]
            bin_end_fine = log_edges_fine[bin_idx_interp_fine + 1]
            frac_fine = (log_s_t_fine_clipped - bin_start_fine) / (bin_end_fine - bin_start_fine)
            frac_fine = np.clip(frac_fine, 0.0, 1.0)
            
            next_bin_fine = np.clip(bin_idx_interp_fine + 1, 0, n_bins_fine - 1)
            cont_path_fine = (1 - frac_fine) * cont_vals_fine[bin_idx_interp_fine] + frac_fine * cont_vals_fine[next_bin_fine]
            
            disc_cont_fine = disc * cont_path_fine
            V_fine = np.maximum(payoff_t_fine, disc_cont_fine)
        
        cont_vals_fine_per_date.reverse()
        
        # Use the fine-grained results if they completed in time
        elapsed_fine = time.time() - start_time
        if elapsed_fine < 140:
            cont_vals_per_date = cont_vals_fine_per_date
            log_edges = log_edges_fine
            s_edges = s_edges_fine
            n_bins = n_bins_fine
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 140:
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
