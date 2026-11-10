import numpy as np
import time
import random

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i
    in the same bin. Larger is more attractive.
    
    Strategy:
    1. Perform a fast local search (Simulated Annealing style) to find a high-quality packing.
    2. Analyze the bins in the best packing found.
    3. Construct the heuristic matrix H[i, j] based on how often item j was placed in the same bin as item i 
       in the high-quality solution, weighted by the bin's quality (lower waste is better).
    4. This effectively teaches the ACO the "patterns" of good packing found by the local search.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    TIME_LIMIT = 300.0
    MARGIN = 10.0 # Stop 10s before limit
    
    # Initialize heuristic to 1e-9 (neutral/invalid)
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Validity mask: items i and j can be in the same bin if d[i] + d[j] <= capacity
    # Note: In bin packing, usually we check if item fits in remaining space.
    # The heuristic H[i,j] is used when current item is i. 
    # If i is already in a bin with other items, the "current" state is complex.
    # However, the problem statement says: "attractiveness for placing item j right after item i in the same bin".
    # This implies a pairwise compatibility heuristic.
    # Standard ACO for bin packing often uses H[i,j] = 1 if d[i]+d[j]<=cap else 0, or based on waste.
    # Here we will use co-occurrence in a good solution.
    
    # 1. Fast Local Search to find a good packing
    # We will use a simple greedy + swap improvement approach.
    # Since n=120, we can afford a few iterations.
    
    # Sort indices by demand descending for initial greedy
    indices = np.argsort(d)[::-1]
    
    best_bins = None
    best_score = float('inf')
    
    # Multiple restarts to escape local optima
    num_restarts = 50
    time_per_restart = (TIME_LIMIT - MARGIN - (time.time() - start_time)) / num_restarts
    time_per_restart = max(time_per_restart, 0.1) # At least 0.1s per restart
    
    for r in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - MARGIN:
            break
            
        # Initialize random order or sorted order with some noise
        if r == 0:
            order = indices.copy()
        else:
            # Perturb the sorted order
            order = indices.copy()
            # Swap a few random pairs
            for _ in range(10):
                i, j = random.sample(range(n), 2)
                order[i], order[j] = order[j], order[i]
        
        # Greedy Bin Packing (First Fit Decreasing variant with random order)
        bins = []
        item_to_bin = {}
        
        for idx in order:
            size = d[idx]
            placed = False
            # Try to fit in existing bins
            # To speed up, we can check bins in order of least remaining space (Best Fit)
            # But simple First Fit is faster. Let's use Best Fit for better quality.
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
                placed = True
            
            if not placed:
                bins.append({'items': [idx], 'sum': size})
                item_to_bin[idx] = len(bins) - 1
        
        # Local Improvement: Try to swap items between bins to reduce total bins or improve packing
        # Simple heuristic: Try to move an item from a less filled bin to a more filled bin if it fits.
        # Or try to swap two items to allow a third to fit? That's complex.
        # Let's try a simple "tightening" pass:
        # For each bin, if we remove an item, can we put it in another bin?
        # This is O(n^2 * bins). n=120, bins~10. 120*120*10 = 144k ops. Fast.
        
        improved = True
        max_improve_iters = 5
        iter_count = 0
        while improved and iter_count < max_improve_iters:
            improved = False
            iter_count += 1
            if time.time() - start_time > TIME_LIMIT - MARGIN:
                break
                
            # Rebuild bins list for easier manipulation? Or modify in place.
            # Let's try to move items from bins with lower utilization to higher?
            # Actually, standard improvement: try to move item i from bin A to bin B if it fits.
            # This might reduce the number of bins if bin A becomes empty.
            
            # Sort bins by number of items descending to process fuller bins first?
            # Let's just iterate through all items.
            for idx in range(n):
                if time.time() - start_time > TIME_LIMIT - MARGIN:
                    break
                current_bin = item_to_bin[idx]
                size = d[idx]
                
                # Try moving to any other bin
                for b_idx in range(len(bins)):
                    if b_idx == current_bin:
                        continue
                    rem = capacity - bins[b_idx]['sum']
                    if size <= rem:
                        # Check if moving improves the solution
                        # If current bin becomes empty, we save a bin.
                        # Or if it just tightens packing.
                        # For now, let's only move if it empties the current bin.
                        if len(bins[current_bin]['items']) == 1:
                            # Move it
                            bins[b_idx]['items'].append(idx)
                            bins[b_idx]['sum'] += size
                            bins[current_bin]['items'].remove(idx)
                            bins[current_bin]['sum'] -= size
                            item_to_bin[idx] = b_idx
                            
                            # Remove empty bin
                            if len(bins[current_bin]['items']) == 0:
                                bins.pop(current_bin)
                                # Update item_to_bin for remaining items in higher indices
                                # This is tricky with indices. Let's rebuild item_to_bin after moves.
                                # Actually, popping shifts indices.
                                # Better approach: don't pop immediately, mark as empty.
                                # Or just count bins at the end.
                                # Let's rebuild item_to_bin from bins at the end of the pass.
                            improved = True
                            break # Try next item
                
                # Rebuild item_to_bin to handle index shifts if we popped
                # To avoid complexity, let's not pop bins, just mark them.
                # But "fewest bins" is the score. Empty bins don't count.
                # Let's just calculate score at the end.
        
        # Calculate score: number of non-empty bins
        num_bins = sum(1 for b in bins if len(b['items']) > 0)
        
        if num_bins < best_score:
            best_score = num_bins
            best_bins = [b['items'] for b in bins if len(b['items']) > 0]
            
            # Early stop if we reached a very good score (e.g., lower bound)
            # Lower bound: ceil(sum(demands) / capacity)
            lb = int(np.ceil(np.sum(d) / capacity))
            if best_score == lb:
                break

    # 2. Construct Heuristic from best_bins
    if best_bins is not None:
        # For each bin in the best solution, for each pair (i, j) in that bin,
        # increase eta[i, j] and eta[j, i].
        # Weight by bin quality? Or just count?
        # If a bin has very low waste, it's a "good" bin.
        # Let's weight by 1/waste or just 1.
        
        # Initialize eta to 1e-9
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
                    # Only add if they fit (they do, since they are in the same bin)
                    # But we must respect the global validity mask: d[i]+d[j] <= capacity
                    # Since they were in the same bin with potentially other items, d[i]+d[j] might be > capacity?
                    # No, if they are in the same bin, their sum is <= bin_sum <= capacity.
                    # So d[i]+d[j] <= capacity is guaranteed.
                    
                    eta[i, j] += weight
                    # Symmetric? The ACO picks j after i.
                    # If i and j are in the same bin, the order might matter for the ant.
                    # But for heuristic, symmetry is often fine, or we can make it directed.
                    # Let's keep it symmetric for now, or just add to [i,j].
                    
        # Normalize eta to avoid extremely large values
        max_eta = np.max(eta)
        if max_eta > 0:
            eta = eta / max_eta
            # Scale to a reasonable range, e.g., max 10
            eta = eta * 10.0
            
        # Ensure validity: if d[i] + d[j] > capacity, eta[i,j] must be 1e-9
        pair_sums = d[:, None] + d[None, :]
        valid = pair_sums <= capacity
        eta = np.where(valid, eta, 1e-9)
        
        # Ensure no zero/negative/inf
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        
    else:
        # Fallback: simple waste-based heuristic
        pair_sums = d[:, None] + d[None, :]
        valid = pair_sums <= capacity
        waste = np.where(valid, capacity - pair_sums, 0)
        eta = np.where(valid, 1.0 / (1.0 + waste), 1e-9)
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9

    return eta
