import numpy as np
import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 129
    TARGET_LOW = 128
    TARGET_HIGH = 256
    TARGET_SLICE = slice(TARGET_LOW, TARGET_HIGH + 1)
    
    def get_counts_from_mask(A_mask):
        vals = np.where(A_mask)[0]
        if len(vals) == 0:
            return np.zeros(257, dtype=int)
        sums = vals[:, None] + vals[None, :]
        i, j = np.triu_indices(len(vals))
        upper_sums = sums[i, j]
        valid_sums = upper_sums[(upper_sums >= TARGET_LOW) & (upper_sums <= TARGET_HIGH)]
        counts = np.bincount(valid_sums, minlength=257)
        return counts
    
    def evaluate(counts):
        if np.min(counts[TARGET_SLICE]) == 0:
            return float('inf')
        return np.max(counts[TARGET_SLICE])
    
    # Greedy seed generation
    def make_greedy_seed():
        A_mask = np.zeros(N, dtype=bool)
        counts = np.zeros(257, dtype=int)
        
        # Start with a reasonable size
        target_size = 40
        
        for _ in range(target_size):
            best_k = None
            best_score = float('inf')
            
            # Try adding each candidate
            for k in range(N):
                if A_mask[k]:
                    continue
                
                # Calculate what adding k would do
                vals = np.where(A_mask)[0]
                new_pairs = np.append(k + vals, 2*k)
                valid_new = new_pairs[(new_pairs >= TARGET_LOW) & (new_pairs <= TARGET_HIGH)]
                
                new_counts = counts.copy()
                if len(valid_new) > 0:
                    new_counts[valid_new] += 1
                
                # Evaluate: we want to minimize the max count while keeping coverage
                if np.min(new_counts[TARGET_SLICE]) == 0:
                    score = float('inf')
                else:
                    score = np.max(new_counts[TARGET_SLICE])
                
                # Tie-breaker: prefer elements that cover more uncovered targets
                if score < best_score or (score == best_score and best_k is None):
                    best_score = score
                    best_k = k
            
            if best_k is not None:
                A_mask[best_k] = True
                vals = np.where(A_mask)[0]
                new_pairs = np.append(best_k + vals[:-1], 2*best_k)  # vals[:-1] excludes the just-added one for pairs with others
                valid_new = new_pairs[(new_pairs >= TARGET_LOW) & (new_pairs <= TARGET_HIGH)]
                if len(valid_new) > 0:
                    counts[valid_new] += 1
        
        return A_mask, counts
    
    # Min-cost max-flow for load balancing
    def solve_min_cost_flow(A_mask, max_iter=50):
        """
        Try to reduce peak multiplicity by finding alternative representations.
        This is a simplified approach: we'll use local search guided by flow-like reasoning.
        """
        vals = np.where(A_mask)[0]
        if len(vals) == 0:
            return A_mask
        
        # Build pair-to-sum mapping
        pairs = []
        for i in range(len(vals)):
            for j in range(i, len(vals)):
                s = vals[i] + vals[j]
                if TARGET_LOW <= s <= TARGET_HIGH:
                    pairs.append((i, j, s))
        
        # Count current multiplicity per sum
        counts = np.zeros(257, dtype=int)
        for i, j, s in pairs:
            counts[s] += 1
        
        if np.min(counts[TARGET_SLICE]) == 0:
            return A_mask
        
        current_peak = np.max(counts[TARGET_SLICE])
        
        # Try to reduce peak by swapping elements
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Find sums with peak multiplicity
            peak_sums = np.where(counts[TARGET_SLICE] == current_peak)[0] + TARGET_LOW
            
            for s in peak_sums:
                # Find all pairs that sum to s
                pairs_for_s = [(i, j) for i, j, ss in pairs if ss == s]
                
                # Try removing one pair and see if we can replace it
                for i, j in pairs_for_s:
                    # Try to find an alternative representation
                    # This is expensive, so we'll do a simple heuristic
                    pass
            
            # Simple local search: try swapping elements
            improved = False
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                # Pick a random element to potentially remove
                if len(vals) <= 20:
                    break
                k_idx = random.randint(0, len(vals) - 1)
                k = vals[k_idx]
                
                # Try removing k
                new_mask = A_mask.copy()
                new_mask[k] = False
                new_vals = np.where(new_mask)[0]
                
                # Check if coverage is maintained
                if len(new_vals) < 2:
                    continue
                
                new_sums = new_vals[:, None] + new_vals[None, :]
                i_idx, j_idx = np.triu_indices(len(new_vals))
                new_upper = new_sums[i_idx, j_idx]
                new_valid = new_upper[(new_upper >= TARGET_LOW) & (new_upper <= TARGET_HIGH)]
                new_counts = np.bincount(new_valid, minlength=257)
                
                if np.min(new_counts[TARGET_SLICE]) == 0:
                    continue
                
                new_peak = np.max(new_counts[TARGET_SLICE])
                
                if new_peak < current_peak:
                    A_mask = new_mask
                    vals = new_vals
                    counts = new_counts
                    current_peak = new_peak
                    improved = True
                    break
            
            if not improved:
                # Try adding elements to help balance
                for k in range(N):
                    if A_mask[k]:
                        continue
                    if time.time() > deadline:
                        break
                    
                    new_mask = A_mask.copy()
                    new_mask[k] = True
                    new_vals = np.where(new_mask)[0]
                    
                    new_sums = new_vals[:, None] + new_vals[None, :]
                    i_idx, j_idx = np.triu_indices(len(new_vals))
                    new_upper = new_sums[i_idx, j_idx]
                    new_valid = new_upper[(new_upper >= TARGET_LOW) & (new_upper <= TARGET_HIGH)]
                    new_counts = np.bincount(new_valid, minlength=257)
                    
                    if np.min(new_counts[TARGET_SLICE]) == 0:
                        continue
                    
                    new_peak = np.max(new_counts[TARGET_SLICE])
                    
                    if new_peak < current_peak:
                        A_mask = new_mask
                        vals = new_vals
                        counts = new_counts
                        current_peak = new_peak
                        break
        
        return A_mask
    
    # Local search with delta evaluation
    def local_search(A_mask, counts, max_steps=5000):
        vals = np.where(A_mask)[0]
        current_score = evaluate(counts)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add an element
                candidates = np.where(~A_mask)[0]
                if len(candidates) == 0:
                    continue
                k = candidates[random.randint(0, len(candidates) - 1)]
                
                # Calculate delta
                new_pairs = np.append(k + vals, 2 * k)
                valid_new = new_pairs[(new_pairs >= TARGET_LOW) & (new_pairs <= TARGET_HIGH)]
                
                new_counts = counts.copy()
                if len(valid_new) > 0:
                    new_counts[valid_new] += 1
                
                new_score = evaluate(new_counts)
                
                if new_score < current_score:
                    counts = new_counts
                    A_mask[k] = True
                    vals = np.append(vals, k)
                    current_score = new_score
            elif move_type < 0.8:
                # Remove an element
                if len(vals) <= 15:
                    continue
                k_idx = random.randint(0, len(vals) - 1)
                k = vals[k_idx]
                
                # Calculate delta
                old_pairs = k + vals
                valid_old = old_pairs[(old_pairs >= TARGET_LOW) & (old_pairs <= TARGET_HIGH)]
                
                new_counts = counts.copy()
                if len(valid_old) > 0:
                    new_counts[valid_old] -= 1
                
                new_score = evaluate(new_counts)
                
                if new_score < current_score:
                    counts = new_counts
                    A_mask[k] = False
                    vals = np.delete(vals, k_idx)
                    current_score = new_score
            else:
                # Swap: remove one, add another
                if len(vals) <= 15:
                    continue
                k_idx = random.randint(0, len(vals) - 1)
                k = vals[k_idx]
                
                candidates = np.where(~A_mask)[0]
                if len(candidates) == 0:
                    continue
                new_k = candidates[random.randint(0, len(candidates) - 1)]
                
                # Remove k, add new_k
                old_pairs = k + vals
                valid_old = old_pairs[(old_pairs >= TARGET_LOW) & (valid_old := old_pairs) <= TARGET_HIGH]
                
                new_vals = np.delete(vals, k_idx)
                new_pairs = np.append(new_k + new_vals, 2 * new_k)
                valid_new = new_pairs[(new_pairs >= TARGET_LOW) & (new_pairs <= TARGET_HIGH)]
                
                new_counts = counts.copy()
                if len(valid_old) > 0:
                    new_counts[valid_old] -= 1
                if len(valid_new) > 0:
                    new_counts[valid_new] += 1
                
                new_score = evaluate(new_counts)
                
                if new_score < current_score:
                    counts = new_counts
                    A_mask[k] = False
                    A_mask[new_k] = True
                    vals = np.append(new_vals, new_k)
                    current_score = new_score
        
        return A_mask, counts, current_score
    
    # Main loop
    best_A = None
    best_score = float('inf')
    
    # Generate initial seed
    A_mask, counts = make_greedy_seed()
    current_score = evaluate(counts)
    
    if current_score < best_score:
        best_score = current_score
        best_A = A_mask.copy()
    
    # Run local search
    A_mask, counts, current_score = local_search(A_mask, counts, max_steps=20000)
    
    if current_score < best_score:
        best_score = current_score
        best_A = A_mask.copy()
    
    # Try multiple restarts if time permits
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        
        # Generate a new seed with some randomness
        A_mask = np.zeros(N, dtype=bool)
        size = random.randint(35, 50)
        indices = random.sample(range(N), size)
        for i in indices:
            A_mask[i] = True
        
        counts = get_counts_from_mask(A_mask)
        current_score = evaluate(counts)
        
        if current_score < best_score:
            best_score = current_score
            best_A = A_mask.copy()
        
        # Run local search
        A_mask, counts, current_score = local_search(A_mask, counts, max_steps=5000)
        
        if current_score < best_score:
            best_score = current_score
            best_A = A_mask.copy()
        
        # Try min-cost flow refinement
        A_mask = solve_min_cost_flow(A_mask, max_iter=20)
        counts = get_counts_from_mask(A_mask)
        current_score = evaluate(counts)
        
        if current_score < best_score:
            best_score = current_score
            best_A = A_mask.copy()
    
    if best_A is None:
        # Fallback
        return list(range(64, 129))
    
    return np.where(best_A)[0].tolist()
