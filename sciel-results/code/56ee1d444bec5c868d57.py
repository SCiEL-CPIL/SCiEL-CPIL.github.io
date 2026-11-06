"""
Strategy: Spectral Subgradient Descent + Fast Local Search with Histogram Tracking.

1. Precompute all Arithmetic Progressions (APs) of length >= 2 in [1, 256].
2. Phase 1: Spectral Subgradient Descent.
   - Minimize the smooth surrogate f(x) = sum(sums^2) where sums are AP signed sums.
   - This helps find a good continuous basin in the hypercube neighborhood.
   - Project the continuous vector to the nearest ±1 signs.
3. Phase 2: Fast Local Search.
   - Use precomputed incidence lists (position -> APs).
   - Maintain current AP sums and a histogram of absolute sums for O(1) max retrieval.
   - Iteratively flip signs that reduce the maximum absolute sum.
   - This is significantly faster than recomputing the max over all APs for each flip.
"""

import time
import numpy as np
import random
import sys

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # 1. Precompute APs
    # We need all APs of length >= 2.
    # AP is defined by start, step, length.
    # Indices are 0-based for numpy, but problem is 1-based.
    # Let's store indices as 0-based.
    
    ap_indices = []
    # For each step d
    for d in range(1, n // 2 + 1):
        # For each start
        # Max start such that start + 2*d <= n (1-based) -> start <= n - 2*d
        # In 0-based: start_idx + 2*d < n -> start_idx < n - 2*d
        # Wait, length >= 2 means at least 2 terms.
        # Term 0: s, Term 1: s+d.
        # s+d < n (0-based) => s < n-d.
        # So s ranges from 0 to n-d-1.
        for s in range(0, n - d):
            terms = []
            cur = s
            while cur < n:
                terms.append(cur)
                cur += d
            # We only care about APs of length >= 2
            if len(terms) >= 2:
                ap_indices.append(terms)
                
    num_aps = len(ap_indices)
    
    # Build incidence list: pos_to_aps[pos] = list of ap_ids containing pos
    pos_to_aps = [[] for _ in range(n)]
    for i, idx in enumerate(ap_indices):
        for j in idx:
            pos_to_aps[j].append(i)
            
    # Build AP lengths array for quick access if needed, though sums are tracked directly.
    ap_lengths = np.array([len(idx) for idx in ap_indices], dtype=np.int32)

    # Helper to compute max discrepancy from sums
    def get_max_disc(sums):
        # sums is a numpy array of int
        return np.max(np.abs(sums))

    # Phase 1: Spectral Subgradient Descent
    # We want to minimize sum(sums^2).
    # Let x be a vector of length n.
    # sums = A @ x, where A is the incidence matrix (0/1).
    # We don't build A explicitly for memory, but we can compute sums efficiently?
    # Actually, for n=256 and ~50k APs, A is sparse but 50k*256 is ~12M entries.
    # Building A might take some time but is feasible.
    # Alternatively, we can just use the local search from a random or Thue-Morse start.
    # The "Spectral" part in the prompt suggests using the continuous relaxation.
    # Let's try to do a few steps of gradient descent on the continuous variable x.
    # grad_x = A^T @ (2 * sums) = 2 * A^T @ sums.
    # A^T @ sums: for each position j, sum of sums[ap_id] for all ap_id containing j.
    # This can be computed using the incidence list.
    
    # Initialize x with Thue-Morse or Random
    # Thue-Morse is a good start.
    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs

    # Let's start with Thue-Morse signs as the initial discrete solution.
    # Then run local search.
    # The prompt asks for "Spectral approximation ... via subgradient descent ... Project ... then local search".
    # Let's implement that to be faithful to the directive, even if TM is strong.
    
    # Initialize continuous x
    x = np.zeros(n)
    # Start with TM signs as initial x? Or random?
    # TM is deterministic and good. Let's use it.
    x = thue_morse_signs(n).astype(np.float64)
    
    # Precompute A^T @ v function
    def compute_grad(x):
        # 1. Compute sums = A @ x
        sums = np.zeros(num_aps, dtype=np.float64)
        for i, idx in enumerate(ap_indices):
            sums[i] = np.sum(x[idx])
        
        # 2. Compute grad = A^T @ (2 * sums)
        grad = np.zeros(n, dtype=np.float64)
        # For each ap, add 2*sums[ap] to all positions in ap
        # This is O(num_aps * avg_len)
        for i in range(num_aps):
            s = 2.0 * sums[i]
            if s != 0:
                for j in ap_indices[i]:
                    grad[j] += s
        return grad, sums

    # Subgradient Descent
    # Step size decay
    lr = 0.1
    for step in range(200):
        if time.time() > deadline - 30.0:
            break
            
        grad, sums = compute_grad(x)
        x -= lr * grad
        lr *= 0.95
        
        # Optional: Project to [-1, 1] to keep values bounded?
        # The objective is convex? No, sum of squares of linear forms is convex.
        # So gradient descent should work.
        
    # Project to nearest ±1
    signs = np.sign(x)
    # Handle zeros (should be rare)
    signs[signs == 0] = 1
    signs = signs.astype(np.int8)
    
    # Phase 2: Fast Local Search with Histogram
    # Initialize sums
    current_sums = np.zeros(num_aps, dtype=np.int32)
    for i, idx in enumerate(ap_indices):
        current_sums[i] = np.sum(signs[idx])
        
    # Histogram of absolute sums
    # Max possible sum is n=256. So abs sum in [0, 256].
    max_possible_sum = n
    hist = np.zeros(max_possible_sum + 1, dtype=np.int32)
    
    def update_hist(val, delta):
        # delta is +1 or -1 to the histogram count
        # val is the absolute sum
        hist[val] += delta
        if hist[val] < 0:
             # Should not happen, but for safety
             pass

    # Build initial histogram
    for s in current_sums:
        hist[abs(s)] += 1
        
    # Find current max
    current_max = 0
    for val in range(max_possible_sum, -1, -1):
        if hist[val] > 0:
            current_max = val
            break

    best_signs = signs.copy()
    best_max = current_max
    
    # Local Search Loop
    # We will try to flip signs to reduce current_max.
    # To speed up, we can focus on positions involved in APs with |sum| == current_max.
    # Or just iterate through all positions and check if flip improves max.
    
    # Optimization: Precompute for each position the list of APs it belongs to.
    # We already have pos_to_aps.
    
    # To make it even faster, we can maintain a list of "hot" APs (those with max sum).
    # But flipping one sign affects many APs.
    
    # Let's do a greedy pass.
    # Iterate through positions.
    # For each position, calculate the new max if we flip.
    # If new max < current_max, flip.
    # If new max == current_max, maybe flip if it reduces the count of max APs?
    # For simplicity, only flip if it strictly reduces max.
    # Then, if stuck, try random restarts or simulated annealing?
    # The prompt says "short, strict local search".
    
    # To improve performance, we can use the histogram to quickly check if a flip reduces the max.
    # When flipping sign at pos p:
    # For each ap in pos_to_aps[p]:
    #   old_sum = current_sums[ap]
    #   new_sum = old_sum - 2 * signs[p]
    #   Update hist: hist[abs(old_sum)] -= 1, hist[abs(new_sum)] += 1
    #   Update current_sums[ap] = new_sum
    #   Track if new_max < current_max.
    
    # Since we need to revert if not improving, we can do a "dry run" or store changes.
    # Given n=256 and ~200 APs per position on average (50k/256), a dry run is ~200 ops.
    # 256 positions * 200 ops = 50k ops per pass.
    # 50k ops is very fast in Python/Numpy mix.
    
    improved = True
    passes = 0
    max_passes = 1000
    
    while improved and passes < max_passes and time.time() < deadline - 5.0:
        improved = False
        passes += 1
        
        # Shuffle positions to avoid bias?
        # Random order might help escape local minima slightly.
        order = np.random.permutation(n)
        
        for p in order:
            if time.time() > deadline - 5.0:
                break
                
            old_sign = signs[p]
            # Flip
            new_sign = -old_sign
            
            # List of affected APs
            affected_aps = pos_to_aps[p]
            
            # Store old sums for reversion
            old_sums = []
            old_abs_sums = []
            
            # Apply flip and update histogram
            for ap_id in affected_aps:
                old_sum = current_sums[ap_id]
                new_sum = old_sum - 2 * old_sign
                old_sums.append(old_sum)
                old_abs_sums.append(abs(old_sum))
                
                # Update hist
                hist[abs(old_sum)] -= 1
                hist[abs(new_sum)] += 1
                
                current_sums[ap_id] = new_sum
                
            # Find new max
            # We can scan down from current_max
            new_max = current_max
            # It's possible new_max is lower, or same, or higher?
            # Flipping a sign changes sums by +/-2.
            # If we reduce a max sum, new_max might drop.
            # If we increase a non-max sum to become new max, new_max might rise.
            # We want to minimize max, so we only accept if new_max < current_max.
            # Or if new_max == current_max but count of max APs decreases?
            # Let's stick to strict reduction of max for now.
            
            # To find new_max efficiently:
            # Check if hist[current_max] is 0. If so, scan down.
            # If hist[current_max] > 0, new_max is at least current_max.
            # Actually, we need to check if any sum became larger than current_max.
            # The max possible new sum is current_max + 2 (if we flipped a sign that increased a sum close to max).
            # So new_max <= current_max + 2.
            
            # Let's just scan from current_max + 2 down to 0 to find the first non-zero hist entry.
            # This is O(max_possible_sum) = 256 ops.
            # 256 * 256 positions * 1000 passes = 65M ops. Might be slow in Python loop.
            
            # Optimization:
            # We only care if new_max < current_max.
            # This requires that hist[current_max] becomes 0 AND no new sum > current_max is created.
            # New sums are old_sum +/- 2.
            # If old_sum was current_max, new_sum is current_max - 2 or current_max + 2.
            # If old_sum was current_max - 1, new_sum is current_max - 3 or current_max + 1.
            # If old_sum was current_max - 2, new_sum is current_max - 4 or current_max.
            
            # Simpler approach:
            # 1. Check if any new_sum > current_max.
            # 2. If not, check if hist[current_max] == 0.
            # 3. If both true, find new_max by scanning down from current_max - 1.
            
            has_larger = False
            for ap_id, new_sum in zip(affected_aps, current_sums[affected_aps]):
                if abs(new_sum) > current_max:
                    has_larger = True
                    break
            
            if has_larger:
                # Revert
                for ap_id, old_sum, old_abs in zip(affected_aps, old_sums, old_abs_sums):
                    hist[abs(current_sums[ap_id])] -= 1
                    hist[old_abs] += 1
                    current_sums[ap_id] = old_sum
                continue
            
            if hist[current_max] > 0:
                # Max is still present.
                # Revert
                for ap_id, old_sum, old_abs in zip(affected_aps, old_sums, old_abs_sums):
                    hist[abs(current_sums[ap_id])] -= 1
                    hist[old_abs] += 1
                    current_sums[ap_id] = old_sum
                continue
            
            # Max is gone. Find new max.
            new_max = current_max - 1
            while new_max >= 0 and hist[new_max] == 0:
                new_max -= 1
                
            if new_max < current_max:
                # Accept flip
                signs[p] = new_sign
                current_max = new_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert (should not happen if logic is correct, but safe)
                for ap_id, old_sum, old_abs in zip(affected_aps, old_sums, old_abs_sums):
                    hist[abs(current_sums[ap_id])] -= 1
                    hist[old_abs] += 1
                    current_sums[ap_id] = old_sum

    # Final verification
    final_sums = np.zeros(num_aps, dtype=np.int32)
    for i, idx in enumerate(ap_indices):
        final_sums[i] = np.sum(best_signs[idx])
    final_max = np.max(np.abs(final_sums))
    
    if final_max < best_max:
        best_max = final_max

    return [int(s) for s in best_signs]

# EVOLVE-BLOCK-END
