import numpy as np
import time
import random

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i
    in the same bin. Larger is more attractive.
    
    Strategy:
    1. Perform a fast local search (Best-Fit Decreasing + improvement) to find a high-quality packing.
    2. Analyze the bins in the best packing found.
    3. Construct the heuristic matrix H[i, j] based on co-occurrence in the high-quality solution,
       weighted by bin quality (lower waste is better).
    4. Boost pairs that are "tight" (small residual space) as they represent critical packings.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    TIME_LIMIT = 300.0
    MARGIN = 10.0
    
    # Initialize heuristic
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Validity mask: items i and j can be in the same bin if d[i] + d[j] <= capacity
    pair_sums = d[:, None] + d[None, :]
    valid = pair_sums <= capacity
    
    # 1. Fast Local Search to find a good packing
    # Use Best-Fit Decreasing with multiple restarts
    
    indices = np.argsort(d)[::-1]
    
    best_bins = None
    best_score = float('inf')
    
    num_restarts = 100
    time_per_restart = (TIME_LIMIT - MARGIN - (time.time() - start_time)) / num_restarts
    time_per_restart = max(time_per_restart, 0.05)
    
    for r in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - MARGIN:
            break
            
        # Initialize order
        if r == 0:
            order = indices.copy()
        elif r % 5 == 0:
            # Sorted order
            order = indices.copy()
        else:
            # Perturbed sorted order
            order = indices.copy()
            for _ in range(5 + r % 10):
                i, j = random.sample(range(n), 2)
                order[i], order[j] = order[j], order[i]
        
        # Greedy Bin Packing with Best Fit
        bins = []  # List of [sum, [item_indices]]
        item_to_bin = [-1] * n
        
        for idx in order:
            size = d[idx]
            placed = False
            
            # Find best fit bin (least remaining space that still fits)
            best_bin_idx = -1
            best_remaining = float('inf')
            
            for b_idx in range(len(bins)):
                rem = capacity - bins[b_idx][0]
                if size <= rem:
                    if rem < best_remaining:
                        best_remaining = rem
                        best_bin_idx = b_idx
            
            if best_bin_idx != -1:
                bins[best_bin_idx][0] += size
                bins[best_bin_idx][1].append(idx)
                item_to_bin[idx] = best_bin_idx
                placed = True
            
            if not placed:
                bins.append([size, [idx]])
                item_to_bin[idx] = len(bins) - 1
        
        # Local Improvement: Try to move items to reduce bins
        # Only move an item if it empties its current bin
        improved = True
        max_improve_iters = 3
        iter_count = 0
        while improved and iter_count < max_improve_iters:
            improved = False
            iter_count += 1
            if time.time() - start_time > TIME_LIMIT - MARGIN:
                break
            
            # Rebuild item_to_bin
            item_to_bin = [-1] * n
            for b_idx, bin_data in enumerate(bins):
                for idx in bin_data[1]:
                    item_to_bin[idx] = b_idx
            
            for idx in range(n):
                if time.time() - start_time > TIME_LIMIT - MARGIN:
                    break
                current_bin = item_to_bin[idx]
                if current_bin == -1:
                    continue
                size = d[idx]
                
                # Only try to move if current bin has exactly 1 item (will be emptied)
                if len(bins[current_bin][1]) > 1:
                    continue
                
                # Try moving to any other bin
                for b_idx in range(len(bins)):
                    if b_idx == current_bin:
                        continue
                    rem = capacity - bins[b_idx][0]
                    if size <= rem:
                        # Move item
                        bins[b_idx][0] += size
                        bins[b_idx][1].append(idx)
                        bins[current_bin][0] -= size
                        bins[current_bin][1].remove(idx)
                        item_to_bin[idx] = b_idx
                        
                        # Remove empty bin
                        if len(bins[current_bin][1]) == 0:
                            bins.pop(current_bin)
                            # Update item_to_bin for remaining items
                            new_item_to_bin = [-1] * n
                            for new_b_idx, bin_data in enumerate(bins):
                                for item_idx in bin_data[1]:
                                    new_item_to_bin[item_idx] = new_b_idx
                            item_to_bin = new_item_to_bin
                        improved = True
                        break
        
        # Calculate score: number of non-empty bins
        num_bins = len(bins)
        
        if num_bins < best_score:
            best_score = num_bins
            best_bins = [bin_data[1] for bin_data in bins if len(bin_data[1]) > 0]
            
            # Early stop if we reached lower bound
            lb = int(np.ceil(np.sum(d) / capacity))
            if best_score == lb:
                break
    
    # 2. Construct Heuristic from best_bins
    if best_bins is not None:
        eta = np.full((n, n), 1e-9, dtype=float)
        
        for bin_items in best_bins:
            bin_sum = sum(d[idx] for idx in bin_items)
            waste = capacity - bin_sum
            # Weight: higher weight for tighter bins
            weight = 1.0 / (1.0 + waste)
            
            m = len(bin_items)
            for a in range(m):
                for b in range(m):
                    if a == b:
                        continue
                    i = bin_items[a]
                    j = bin_items[b]
                    eta[i, j] += weight
        
        # Normalize
        max_eta = np.max(eta)
        if max_eta > 0:
            eta = eta / max_eta
            eta = eta * 10.0
        
        # Apply validity mask
        eta = np.where(valid, eta, 1e-9)
        
        # Ensure no zero/negative/inf
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
    
    else:
        # Fallback: simple waste-based heuristic
        waste = np.where(valid, capacity - pair_sums, 0)
        eta = np.where(valid, 1.0 / (1.0 + waste), 1e-9)
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9

    return eta
