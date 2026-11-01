"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Uses Simulated Annealing on a soft penalty function to minimize the number of codewords
    while ensuring coverage.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        return _generic_greedy(q, n, radius)

    total_words = 3 ** n
    all_words = list(itertools.product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (distance <= 1)
    # neighbors[i] is a list of indices j such that dist(word_i, word_j) <= 1
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i) # self
        for pos in range(n):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                neighbors[i].append(j)
    
    # Convert to numpy arrays for faster intersection/union if needed, 
    # but sets might be faster for small N. 
    # Let's use bitmasks or just lists. 
    # For 729 words, a boolean array of length 729 is efficient.
    
    # Initial state: Tetracode construction (81 words)
    # Or random? Tetracode is a good start.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_mask = np.zeros(total_words, dtype=bool)
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            initial_mask[word_to_idx[w]] = True
            
    best_mask = initial_mask.copy()
    best_size = np.sum(best_mask)
    
    # Helper to compute coverage and size
    def evaluate(mask):
        size = np.sum(mask)
        covered = np.zeros(total_words, dtype=bool)
        # Efficiently update coverage?
        # For SA, we need fast evaluation.
        # Let's maintain a 'coverage_count' array.
        # If we do SA, we should update incrementally.
        # But for simplicity in the first pass, let's just compute from scratch if N is small?
        # 729 * 729 ops is ~500k. In Python loop, that's slow.
        # Vectorized: 
        # covered = np.zeros(total_words, dtype=bool)
        # for i in range(total_words):
        #    if mask[i]:
        #       covered[neighbors[i]] = True
        # This is O(N * avg_neighbors). 729 * 13 ~ 9k ops. Very fast.
        
        covered = np.zeros(total_words, dtype=bool)
        active_indices = np.where(mask)[0]
        if len(active_indices) > 0:
            # Flatten neighbors of active indices
            # This creates a large array, but it's manageable.
            # Or just loop.
            for i in active_indices:
                covered[neighbors[i]] = True
        uncovered = total_words - np.sum(covered)
        return size, uncovered

    # Simulated Annealing
    start_time = time.time()
    timeout = 110.0 # Leave 10s for safety
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    current_mask = initial_mask.copy()
    current_size, current_uncovered = evaluate(current_mask)
    current_cost = current_size + 100.0 * current_uncovered # Heavy penalty on uncovered initially
    
    best_size_sa = current_size if current_uncovered == 0 else float('inf')
    best_mask_sa = current_mask.copy() if current_uncovered == 0 else None

    while T > T_min and (time.time() - start_time) < timeout:
        # Generate a move
        move_type = random.choice(['remove', 'add', 'swap'])
        new_mask = current_mask.copy()
        
        if move_type == 'remove':
            active = np.where(current_mask)[0]
            if len(active) == 0: continue
            idx = random.choice(active)
            new_mask[idx] = False
        elif move_type == 'add':
            inactive = np.where(~current_mask)[0]
            if len(inactive) == 0: continue
            idx = random.choice(inactive)
            new_mask[idx] = True
        else: # swap
            active = np.where(current_mask)[0]
            inactive = np.where(~current_mask)[0]
            if len(active) == 0 or len(inactive) == 0: continue
            idx_rem = random.choice(active)
            idx_add = random.choice(inactive)
            new_mask[idx_rem] = False
            new_mask[idx_add] = True
            
        new_size, new_uncovered = evaluate(new_mask)
        new_cost = new_size + 100.0 * new_uncovered
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
            current_mask = new_mask
            current_size = new_size
            current_uncovered = new_uncovered
            current_cost = new_cost
            
            if new_uncovered == 0:
                if new_size < best_size_sa:
                    best_size_sa = new_size
                    best_mask_sa = new_mask.copy()
        
        T *= cooling
        
    # Final check: if we have a valid mask, use it.
    # If best_mask_sa is None, we didn't find a valid one during SA?
    # That's unlikely if we start with a valid one (81 words).
    # But if we got stuck in a local minimum with high size, we might have lost validity?
    # No, we only accept moves. If we start valid, and accept a move that makes it invalid,
    # the penalty is high, so we might accept it to escape, but we might end up invalid.
    # We must return a VALID code.
    
    if best_mask_sa is not None:
        final_mask = best_mask_sa
    else:
        # Fallback: Repair the current mask if it's invalid, or use initial if current is invalid
        # Actually, let's just use the initial valid mask if SA failed to find a better valid one
        # Or repair the best_mask_sa?
        # Let's just use the initial 81-word code if we don't have a better valid one.
        # Or better: Run a greedy repair on the best_mask_sa found (even if invalid) to make it valid.
        final_mask = current_mask.copy()
        
        # Greedy repair
        size, unc = evaluate(final_mask)
        if unc > 0:
            covered = np.zeros(total_words, dtype=bool)
            for i in np.where(final_mask)[0]:
                covered[neighbors[i]] = True
            uncovered_indices = np.where(~covered)[0]
            
            # Sort candidates by potential coverage
            # Just pick best greedily
            while unc > 0:
                best_idx = -1
                best_gain = -1
                # Optimization: Only check inactive nodes
                candidates = np.where(~final_mask)[0]
                for idx in candidates:
                    gain = np.sum(~covered[neighbors[idx]])
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                if best_idx == -1:
                    break
                final_mask[best_idx] = True
                covered[neighbors[best_idx]] = True
                unc -= best_gain
                
        # Check if this repaired code is better than best_mask_sa?
        # If best_mask_sa was None, this is our best.
        # If best_mask_sa existed, it was valid and smaller.
        # So if best_mask_sa is None, use repaired current/best.
        # Actually, let's just track the best VALID mask seen at any point.
        
    # Re-evaluate to be sure
    # The logic above: best_mask_sa is updated only if new_uncovered == 0.
    # Since we start with a valid mask (81 words), best_mask_sa is at least the 81-word code.
    # So best_mask_sa is never None.
    
    final_mask = best_mask_sa
    result_indices = np.where(final_mask)[0]
    result = [list(all_words[i]) for i in result_indices]
    
    return result

def _generic_greedy(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = {}
    for w in all_words:
        nb = set()
        for v in all_words:
            dist = sum(1 for i in range(n) if w[i] != v[i])
            if dist <= radius:
                nb.add(v)
        neighbors[w] = nb
    
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        best_word = None
        max_covered = -1
        for w in all_words:
            if w in code:
                continue
            covered_count = len(neighbors[w] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w
        
        if best_word is None:
            break
            
        code.append(best_word)
        uncovered -= neighbors[best_word]
        
    return [list(c) for c in code]

# EVOLVE-BLOCK-END
