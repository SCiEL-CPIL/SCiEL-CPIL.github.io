import numpy as np
import time
import random

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i
    in the same bin. Larger is more attractive.
    
    Strategy:
    1. Perform multiple greedy + local search restarts to find high-quality packings.
    2. Accumulate co-occurrence statistics across multiple good solutions, weighted by solution quality.
    3. Construct the heuristic matrix H[i, j] based on weighted co-occurrence.
    4. Apply a final polish pass to refine the best solution.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    TIME_LIMIT = 300.0
    MARGIN = 5.0  # Stop 5s before limit for safety
    
    def time_remaining():
        return TIME_LIMIT - MARGIN - (time.time() - start_time)
    
    # Initialize co-occurrence accumulator
    co_occurrence = np.zeros((n, n), dtype=float)
    
    # Track best solutions for later use
    best_bins = None
    best_score = float('inf')
    good_solutions = []  # Store top solutions for weighted accumulation
    
    # Sort indices by demand descending for initial greedy
    indices = np.argsort(d)[::-1]
    
    # Lower bound for early stopping
    lb = int(np.ceil(np.sum(d) / capacity))
    
    num_restarts = 0
    max_restarts = 200  # Upper bound; actual count depends on time
    
    while time_remaining() > 2.0:
        num_restarts += 1
        if num_restarts > max_restarts:
            break
            
        # Initialize order: first restart is sorted, subsequent ones are perturbed
        if num_restarts == 1:
            order = indices.copy()
        else:
            # Perturb the sorted order with increasing randomness
            order = indices.copy()
            # Number of swaps increases slightly with restart count
            num_swaps = min(5 + num_restarts // 10, 20)
            for _ in range(num_swaps):
                i, j = random.sample(range(n), 2)
                order[i], order[j] = order[j], order[i]
        
        # Greedy Bin Packing with Best Fit
        bins = []  # List of dicts: {'items': [], 'sum': 0}
        item_to_bin = {}
        
        for idx in order:
            size = d[idx]
            best_bin_idx = -1
            best_remaining = float('inf')
            
            for b_idx, b in enumerate(bins):
                rem = capacity - b['sum']
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
        
        # Local Improvement: Move items to reduce bin count
        # Use a safe approach: don't pop bins, just mark them as empty
        improved = True
        max_improve_iters = 10
        iter_count = 0
        while improved and iter_count < max_improve_iters:
            improved = False
            iter_count += 1
            if time_remaining() < 1.0:
                break
                
            for idx in range(n):
                if time_remaining() < 1.0:
                    break
                current_bin = item_to_bin[idx]
                size = d[idx]
                
                # Try moving to any other non-empty bin
                for b_idx in range(len(bins)):
                    if b_idx == current_bin or len(bins[b_idx]['items']) == 0:
                        continue
                    rem = capacity - bins[b_idx]['sum']
                    if size <= rem:
                        # Check if moving improves: current bin becomes empty
                        if len(bins[current_bin]['items']) == 1:
                            # Move it
                            bins[b_idx]['items'].append(idx)
                            bins[b_idx]['sum'] += size
                            bins[current_bin]['items'].remove(idx)
                            bins[current_bin]['sum'] -= size
                            item_to_bin[idx] = b_idx
                            improved = True
                            break
        
        # Calculate score: number of non-empty bins
        non_empty_bins = [b for b in bins if len(b['items']) > 0]
        num_bins = len(non_empty_bins)
        
        # Update best solution
        if num_bins < best_score:
            best_score = num_bins
            best_bins = [b['items'] for b in non_empty_bins]
            
            # Early stop if we reached lower bound
            if best_score == lb:
                break
        
        # Accumulate co-occurrence for good solutions
        # Weight by solution quality: better solutions get higher weight
        if num_bins <= best_score + 1:  # Accept solutions within 1 of best
            # Weight: exponential decay based on gap from best
            gap = num_bins - best_score
            weight = np.exp(-0.5 * gap)
            
            for b in non_empty_bins:
                bin_items = b['items']
                bin_sum = b['sum']
                waste = capacity - bin_sum
                # Additional weight for tighter bins
                bin_weight = weight / (1.0 + waste * 0.1)
                
                m = len(bin_items)
                for a in range(m):
                    for b_idx in range(m):
                        if a == b_idx:
                            continue
                        i = bin_items[a]
                        j = bin_items[b_idx]
                        co_occurrence[i, j] += bin_weight
        
        # Store solution for potential later use
        if num_bins <= best_score + 1 and len(good_solutions) < 50:
            good_solutions.append({'bins': [b['items'] for b in non_empty_bins], 'score': num_bins})

    # Final polish: try 2-opt style swaps on the best solution
    if best_bins is not None and time_remaining() > 3.0:
        # Work on a copy
        current_bins = [list(b) for b in best_bins]
        current_score = len(current_bins)
        
        # Try to improve by swapping items between bins
        for _ in range(50):  # Limited polish iterations
            if time_remaining() < 2.0:
                break
            
            improved = False
            # Try all pairs of bins
            for b1 in range(len(current_bins)):
                if time_remaining() < 2.0:
                    break
                for b2 in range(b1 + 1, len(current_bins)):
                    if time_remaining() < 2.0:
                        break
                    
                    # Try swapping one item from b1 with one from b2
                    for i1 in range(len(current_bins[b1])):
                        for i2 in range(len(current_bins[b2])):
                            item1 = current_bins[b1][i1]
                            item2 = current_bins[b2][i2]
                            
                            # Check if swap is valid
                            new_bin1_sum = sum(d[x] for x in current_bins[b1]) - d[item1] + d[item2]
                            new_bin2_sum = sum(d[x] for x in current_bins[b2]) - d[item2] + d[item1]
                            
                            if new_bin1_sum <= capacity and new_bin2_sum <= capacity:
                                # Perform swap
                                current_bins[b1][i1] = item2
                                current_bins[b2][i2] = item1
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    # Construct heuristic from accumulated co-occurrence
    eta = np.full((n, n), 1e-9, dtype=float)
    
    if np.any(co_occurrence > 0):
        # Use the accumulated co-occurrence
        eta = co_occurrence.copy()
        
        # Normalize to reasonable range
        max_eta = np.max(eta)
        if max_eta > 0:
            eta = eta / max_eta
            eta = eta * 10.0
    elif best_bins is not None:
        # Fallback: use only the best solution
        for bin_items in best_bins:
            bin_sum = sum(d[idx] for idx in bin_items)
            waste = capacity - bin_sum
            weight = 1.0 / (1.0 + waste)
            
            m = len(bin_items)
            for a in range(m):
                for b in range(m):
                    if a == b:
                        continue
                    i = bin_items[a]
                    j = bin_items[b]
                    eta[i, j] += weight
        
        max_eta = np.max(eta)
        if max_eta > 0:
            eta = eta / max_eta
            eta = eta * 10.0
    else:
        # Final fallback: simple waste-based heuristic
        pair_sums = d[:, None] + d[None, :]
        valid = pair_sums <= capacity
        waste = np.where(valid, capacity - pair_sums, 0)
        eta = np.where(valid, 1.0 / (1.0 + waste), 1e-9)
    
    # Ensure validity: if d[i] + d[j] > capacity, eta[i,j] must be 1e-9
    pair_sums = d[:, None] + d[None, :]
    valid = pair_sums <= capacity
    eta = np.where(valid, eta, 1e-9)
    
    # Ensure no zero/negative/inf
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
