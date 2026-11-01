import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)
    
    # Precompute coverage masks
    # masks[i] is a boolean array of length total_words, True if word j is within radius of word i
    masks = np.zeros((total_words, total_words), dtype=bool)
    
    # Optimization for larger spaces (though this task is small)
    if total_words > 5000:
        return _simple_greedy(q, n, radius)
        
    # Compute masks efficiently
    # Convert all_words to a numpy array for vectorized distance calculation
    words_arr = np.array(all_words)
    # Compute Hamming distance matrix
    # dist[i][j] = sum(words_arr[i] != words_arr[j])
    # This creates a (729, 729, 6) array, which is ~3M elements. Fine.
    diff = words_arr[:, np.newaxis, :] != words_arr[np.newaxis, :, :]
    dist = np.sum(diff, axis=2)
    masks = dist <= radius

    def get_cost(counts):
        """Cost is size + penalty for uncovered"""
        uncovered = np.sum(counts == 0)
        return len(counts) * 0 # placeholder, size is passed separately
        # Actually, cost = size + M * uncovered
    
    def greedy_init():
        """Standard greedy set cover"""
        counts = np.zeros(total_words, dtype=int)
        codewords = []
        while True:
            if np.all(counts > 0):
                break
            uncovered_mask = (counts == 0)
            if not np.any(uncovered_mask):
                break
            scores = np.sum(masks & uncovered_mask, axis=1)
            max_score = np.max(scores)
            if max_score == 0:
                break
            candidates = np.where(scores == max_score)[0]
            # Random tie-breaking
            chosen_idx = candidates[np.random.randint(len(candidates))]
            codewords.append(chosen_idx)
            counts[masks[chosen_idx]] += 1
        return codewords, counts

    def greedy_delete(codewords, counts):
        """Remove all redundant codewords"""
        # Sort by index to be deterministic-ish, or random order
        order = list(range(len(codewords)))
        np.random.shuffle(order)
        changed = True
        while changed:
            changed = False
            for pos in order:
                if pos >= len(codewords):
                    continue
                idx = codewords[pos]
                # Check if removing idx leaves any word uncovered
                # counts[masks[idx]] - 1
                # If any of these becomes 0, it was uniquely covered by idx (or tied, but now 0)
                # Actually, if counts[j] > 1 for all j in masks[idx], then removal is safe.
                # If counts[j] == 1 for any j in masks[idx], removal is unsafe.
                # We can check: min(counts[masks[idx]]) > 1
                if np.min(counts[masks[idx]]) > 1:
                    # Safe to remove
                    counts[masks[idx]] -= 1
                    codewords.pop(pos)
                    changed = True
                    # Reset order? Or just continue?
                    # If we pop, indices shift. Better to rebuild order or use a set.
                    # For simplicity and speed in Python, let's just break and restart the while loop
                    # or update order.
                    # Given N is small (~75), restarting is fine.
                    break
        return codewords, counts

    def simulated_annealing(codewords, counts, start_temp=1.0, end_temp=0.01, steps=2000):
        """
        SA to minimize size.
        Cost = len(codewords) + 100 * uncovered_count
        """
        current_size = len(codewords)
        current_uncovered = np.sum(counts == 0)
        current_cost = current_size + 100 * current_uncovered
        
        best_codewords = codewords.copy()
        best_counts = counts.copy()
        best_cost = current_cost
        
        # Cooling schedule
        temp = start_temp
        
        for step in range(steps):
            # Generate neighbor
            new_codewords = codewords.copy()
            new_counts = counts.copy()
            
            r = np.random.random()
            if r < 0.4 and len(new_codewords) > 0:
                # Remove a random codeword
                pos = np.random.randint(len(new_codewords))
                idx = new_codewords[pos]
                new_counts[masks[idx]] -= 1
                new_codewords.pop(pos)
            elif r < 0.7 and len(new_codewords) > 0:
                # Swap a random codeword with a random new one
                pos = np.random.randint(len(new_codewords))
                old_idx = new_codewords[pos]
                new_idx = np.random.randint(total_words)
                if new_idx != old_idx:
                    new_counts[masks[old_idx]] -= 1
                    new_counts[masks[new_idx]] += 1
                    new_codewords[pos] = new_idx
            else:
                # Add a random codeword (only if we are feasible or close to it? 
                # Actually, adding increases size, so only useful if it allows removal later?
                # In SA, we allow worsening moves. Adding increases size by 1.
                # It might reduce uncovered count by many.
                # Let's allow it.
                new_idx = np.random.randint(total_words)
                if new_idx not in new_codewords: # Avoid duplicates
                    new_counts[masks[new_idx]] += 1
                    new_codewords.append(new_idx)
                
            new_size = len(new_codewords)
            new_uncovered = np.sum(new_counts == 0)
            new_cost = new_size + 100 * new_uncovered
            
            delta = new_cost - current_cost
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                # Accept
                codewords = new_codewords
                counts = new_counts
                current_size = new_size
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_codewords = codewords.copy()
                    best_counts = counts.copy()
                    # Optional: run greedy delete on best to refine
                    # But that might be expensive inside the loop.
            
            # Cool down
            temp *= 0.999 # Slow cooling
            
        # Final greedy delete on the best found
        best_codewords, best_counts = greedy_delete(best_codewords, best_counts)
        
        return best_codewords, best_counts

    def _simple_greedy(q, n, radius):
        # Fallback
        all_words = list(itertools.product(range(q), repeat=n))
        uncovered = set(all_words)
        codewords = []
        def get_coverage(word):
            covered = set()
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val == orig: continue
                    new_word = list(word)
                    new_word[pos] = val
                    covered.add(tuple(new_word))
            covered.add(tuple(word))
            return covered
        coverages = {w: get_coverage(w) for w in all_words}
        while uncovered:
            best_word = None
            best_count = -1
            for w in all_words:
                count = len(coverages[w] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = w
            if best_count == 0: break
            codewords.append(list(best_word))
            uncovered -= coverages[best_word]
        return codewords

    best_codewords = None
    best_size = float('inf')
    
    start_time = time.time()
    timeout = 110
    
    np.random.seed(42) # Global seed for reproducibility in this run
    
    # Run multiple SA instances
    for run in range(10):
        # 1. Greedy Init
        cw, counts = greedy_init()
        
        # 2. Greedy Delete (initial cleanup)
        cw, counts = greedy_delete(cw, counts)
        
        # 3. Simulated Annealing
        cw, counts = simulated_annealing(cw, counts, start_temp=2.0, end_temp=0.01, steps=3000)
        
        # 4. Final Greedy Delete
        cw, counts = greedy_delete(cw, counts)
        
        if len(cw) < best_size:
            best_size = len(cw)
            best_codewords = cw
            
        if time.time() - start_time > timeout:
            break
            
    if best_codewords is None:
        # Fallback to simple greedy if SA failed or timed out immediately
        cw, counts = greedy_init()
        cw, counts = greedy_delete(cw, counts)
        best_codewords = cw
        
    return [all_words[i] for i in best_codewords]
