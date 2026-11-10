import numpy as np
import time
import random

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i
    in the same bin. Larger is more attractive.
    
    Strategy:
    1. Perform a fast local search to find a high-quality packing.
    2. Analyze the bins in the best packing found.
    3. Construct the heuristic matrix H[i, j] based on co-occurrence in the high-quality solution,
       weighted by squared inverse waste to strongly prioritize tight fits.
    4. This teaches the ACO the "patterns" of good packing found by the local search.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    TIME_LIMIT = 300.0
    MARGIN = 15.0  # Stop 15s before limit for safety
    
    # Initialize heuristic to 1e-9 (neutral/invalid)
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # 1. Fast Local Search to find a good packing
    # Sort indices by demand descending for initial greedy
    indices = np.argsort(d)[::-1]
    
    best_bins = None
    best_score = float('inf')
    
    # Lower bound for early stopping
    lb = int(np.ceil(np.sum(d) / capacity))
    
    # Multiple restarts to escape local optima
    num_restarts = 80
    
    for r in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - MARGIN:
            break
            
        # Initialize order: sorted for first, perturbed for subsequent
        if r == 0:
            order = indices.copy()
        else:
            # Perturb the sorted order with random swaps
            order = indices.copy()
            num_swaps = min(15, n // 8)
            for _ in range(num_swaps):
                i, j = random.sample(range(n), 2)
                order[i], order[j] = order[j], order[i]
        
        # Greedy Bin Packing with Best Fit
        bins = []  # list of dicts: {'items': [], 'sum': 0}
        item_to_bin = np.full(n, -1, dtype=int)
        
        for idx in order:
            size = d[idx]
            best_bin_idx = -1
            best_remaining = float('inf')
            
            for b_idx in range(len(bins)):
                rem = capacity - bins[b_idx]['sum']
                if size <= rem:
                    if rem < best_remaining:
                        best_remaining = rem
                        best_bin_idx = b_idx
            
            if best_bin_idx != -1:
                bins[best_bin_idx]['items'].append(idx)
                bins[best_bin_idx]['sum'] += size
                item_to_bin[idx] = best_bin_idx
            else:
                bins.append({'items': [idx], 'sum': size})
                item_to_bin[idx] = len(bins) - 1
        
        # Local Improvement: move items to reduce bin count or tighten packing
        # Use a marking approach to avoid index shift issues
        improved = True
        max_improve_iters = 8
        iter_count = 0
        while improved and iter_count < max_improve_iters:
            improved = False
            iter_count += 1
            if time.time() - start_time > TIME_LIMIT - MARGIN:
                break
                
            # Try to move each item to a different bin if it fits
            for idx in range(n):
                if time.time() - start_time > TIME_LIMIT - MARGIN:
                    break
                current_bin = item_to_bin[idx]
                if current_bin < 0:
                    continue
                size = d[idx]
                
                # Try moving to any other bin where it fits
                for b_idx in range(len(bins)):
                    if b_idx == current_bin:
                        continue
                    rem = capacity - bins[b_idx]['sum']
                    if size <= rem:
                        # Only move if it improves: empties current bin or tightens
                        if len(bins[current_bin]['items']) == 1:
                            # Moving empties the current bin - always beneficial
                            bins[b_idx]['items'].append(idx)
                            bins[b_idx]['sum'] += size
                            bins[current_bin]['items'].remove(idx)
                            bins[current_bin]['sum'] -= size
                            item_to_bin[idx] = b_idx
                            improved = True
                            break
                        else:
                            # Check if moving tightens: current bin has more waste after removal
                            # and target bin has less waste after addition
                            # Simple heuristic: move if target bin's remaining after placement
                            # is less than current bin's remaining before removal
                            current_rem_before = capacity - bins[current_bin]['sum']
                            target_rem_after = capacity - (bins[b_idx]['sum'] + size)
                            if target_rem_after < current_rem_before - size:
                                bins[b_idx]['items'].append(idx)
                                bins[b_idx]['sum'] += size
                                bins[current_bin]['items'].remove(idx)
                                bins[current_bin]['sum'] -= size
                                item_to_bin[idx] = b_idx
                                improved = True
                                break
        
        # Clean up empty bins and rebuild item_to_bin
        non_empty_bins = [b for b in bins if len(b['items']) > 0]
        num_bins = len(non_empty_bins)
        
        # Rebuild item_to_bin for valid bins
        new_item_to_bin = np.full(n, -1, dtype=int)
        for b_idx, b in enumerate(non_empty_bins):
            for item in b['items']:
                new_item_to_bin[item] = b_idx
        
        if num_bins < best_score:
            best_score = num_bins
            best_bins = [b['items'][:] for b in non_empty_bins]
            
            # Early stop if we reached the lower bound
            if best_score == lb:
                break
    
    # 2. Construct Heuristic from best_bins
    if best_bins is not None:
        eta = np.full((n, n), 1e-9, dtype=float)
        
        for bin_items in best_bins:
            bin_sum = sum(d[idx] for idx in bin_items)
            waste = capacity - bin_sum
            # Squared inverse waste: strongly prioritizes tight fits
            # Add small epsilon to avoid division by zero
            weight = 1.0 / (1.0 + waste) ** 2
            
            m = len(bin_items)
            for a in range(m):
                for b in range(m):
                    if a == b:
                        continue
                    i = bin_items[a]
                    j = bin_items[b]
                    eta[i, j] += weight
                    
        # Normalize eta
        max_eta = np.max(eta)
        if max_eta > 0:
            eta = eta / max_eta
            eta = eta * 10.0
            
        # Ensure validity: if d[i] + d[j] > capacity, eta[i,j] must be 1e-9
        pair_sums = d[:, None] + d[None, :]
        valid = pair_sums <= capacity
        eta = np.where(valid, eta, 1e-9)
        
        # Ensure no zero/negative/inf
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        
    else:
        # Fallback: squared inverse waste-based heuristic
        pair_sums = d[:, None] + d[None, :]
        valid = pair_sums <= capacity
        waste = np.where(valid, capacity - pair_sums, 0)
        eta = np.where(valid, 1.0 / (1.0 + waste) ** 2, 1e-9)
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9

    return eta
