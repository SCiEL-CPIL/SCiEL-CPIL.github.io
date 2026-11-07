import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    Uses a time-bounded search to optimize the heuristic parameters
    against a simulated stochastic greedy packing process.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0 # 10s margin before 300s limit
    
    # Pre-compute pairwise data
    pair_sums = demands[:, None] + demands[None, :]
    fits = pair_sums <= capacity
    waste = capacity - pair_sums
    
    # Item frequencies (for rarity feature)
    # Demands are integers 20-99 passed as floats
    int_demands = demands.astype(int)
    # Use a dictionary or bincount for speed
    counts = np.bincount(int_demands, minlength=100)
    item_freq = counts[int_demands]
    inv_freq = 1.0 / (item_freq + 1e-9)
    
    # Base feature matrices
    # 1. Tightness: exp(-waste/sigma)
    # 2. Size: pair_sums
    # 3. Rarity: inv_freq[i] + inv_freq[j]
    
    rarity_matrix = inv_freq[:, None] + inv_freq[None, :]
    
    def run_simulator(eta):
        """
        Simulates the packing process driven by eta.
        Returns the minimum number of bins found in 'num_ants' runs.
        """
        num_ants = 10
        best_bins = float('inf')
        
        # Pre-sort indices for faster random choice? 
        # Or just use list(remaining)
        
        for _ in range(num_ants):
            bins_used = 0
            current_bin = []
            remaining = set(range(n))
            
            # Start with a random item
            if remaining:
                first = list(remaining)[np.random.randint(len(remaining))]
                current_bin.append(first)
                remaining.remove(first)
            
            while remaining:
                if not current_bin:
                    # Start new bin with random item
                    idx = list(remaining)[np.random.randint(len(remaining))]
                    current_bin.append(idx)
                    remaining.remove(idx)
                    continue
                
                last_i = current_bin[-1]
                
                # Find candidates
                candidates = []
                weights = []
                
                # Iterate over remaining items
                # Optimization: If n is small, this is fine.
                for j in remaining:
                    if demands[last_i] + demands[j] <= capacity:
                        candidates.append(j)
                        weights.append(eta[last_i, j])
                
                if not candidates:
                    # Close current bin, open new one
                    bins_used += 1
                    current_bin = []
                    # Start new bin with random item
                    idx = list(remaining)[np.random.randint(len(remaining))]
                    current_bin.append(idx)
                    remaining.remove(idx)
                    continue
                
                # Normalize weights
                weights = np.array(weights, dtype=float)
                # Ensure positive
                weights[weights <= 0] = 1e-9
                weights /= weights.sum()
                
                # Pick
                idx = candidates[np.random.choice(len(candidates), p=weights)]
                current_bin.append(idx)
                remaining.remove(idx)
            
            if current_bin:
                bins_used += 1
            
            if bins_used < best_bins:
                best_bins = bins_used
                
        return best_bins

    def evaluate_params(sigma, w1, w2, w3):
        # Construct eta
        decay = np.exp(-waste / sigma)
        eta = w1 * (pair_sums * decay) + w2 * pair_sums + w3 * rarity_matrix
        
        # Mask non-fitting
        eta[~fits] = 1e-9
        # Ensure positive
        eta[eta <= 0] = 1e-9
        
        return run_simulator(eta)

    best_score = float('inf')
    best_params = (15.0, 1.0, 1.0, 1.0) # Default
    
    # Search loop
    while time.time() < deadline:
        # Sample parameters
        sigma = np.random.uniform(5, 50)
        w1 = np.random.uniform(0.1, 5.0)
        w2 = np.random.uniform(0.1, 5.0)
        w3 = np.random.uniform(0.1, 5.0)
        
        score = evaluate_params(sigma, w1, w2, w3)
        
        if score < best_score:
            best_score = score
            best_params = (sigma, w1, w2, w3)

    # Construct final matrix with best params
    sigma, w1, w2, w3 = best_params
    decay = np.exp(-waste / sigma)
    eta = w1 * (pair_sums * decay) + w2 * pair_sums + w3 * rarity_matrix
    eta[~fits] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
