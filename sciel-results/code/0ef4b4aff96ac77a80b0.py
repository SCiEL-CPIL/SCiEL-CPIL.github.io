# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct(q, n, radius):
    """
    Multi-start greedy set-cover algorithm to find a small covering code.
    Uses bitmask representation and multiple random tie-breaking orders
    to escape local optima of the standard greedy heuristic.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    if total_words <= 2**20:
        return _construct_greedy_multi_start(q, n, radius)
    else:
        # Fallback for large spaces (not expected for this task, but safe)
        return _construct_greedy_simple(q, n, radius)


def _construct_greedy_multi_start(q, n, radius):
    """
    Optimized greedy set-cover with multiple restarts for small spaces.
    """
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute ball masks as Python integers (bitmasks)
    ball_masks = [0] * total_words
    
    # Generate neighbors for each center
    for idx in range(total_words):
        center = idx_to_word(idx)
        mask = 1 << idx
        
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center.copy()
                        neighbor[pos] = val
                        nidx = word_to_idx(neighbor)
                        mask |= 1 << nidx
        
        if radius >= 2:
            # Add distance 2 neighbors
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for val1 in range(q):
                        if val1 == center[pos1]: continue
                        for val2 in range(q):
                            if val2 == center[pos2]: continue
                            neighbor = center.copy()
                            neighbor[pos1] = val1
                            neighbor[pos2] = val2
                            nidx = word_to_idx(neighbor)
                            mask |= 1 << nidx
                            
        ball_masks[idx] = mask
    
    all_mask = (1 << total_words) - 1
    best_codewords = None
    best_len = float('inf')
    
    # Increase restarts to improve probability of finding optimal solution
    # 200 restarts is fast enough for N=729
    num_restarts = 200
    
    for seed in range(num_restarts):
        # Generate a random permutation of indices for tie-breaking
        perm = np.random.RandomState(seed).permutation(total_words)
        
        uncovered = all_mask
        codewords = []
        
        while uncovered != 0:
            # Find the index in 'perm' that maximizes coverage of 'uncovered'
            # We iterate through the random permutation and keep the first one with max score.
            # This effectively breaks ties randomly according to the permutation.
            max_score = 0
            best_idx = -1
            
            for idx in perm:
                score = (ball_masks[idx] & uncovered).bit_count()
                if score > max_score:
                    max_score = score
                    best_idx = idx
            
            if max_score == 0:
                break # Should not happen if code is valid cover, but safety break
                
            codewords.append(idx_to_word(best_idx))
            uncovered &= ~ball_masks[best_idx]
            
        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = codewords
            
            # Early termination if we hit a very good bound (e.g., known optimal)
            # For K_3(6,1), optimal is 73.
            if best_len == 73:
                break
                
    return best_codewords

def _construct_greedy_simple(q, n, radius):
    """Fallback simple greedy for larger spaces."""
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    ball_masks = [0] * total_words
    for idx in range(total_words):
        center = idx_to_word(idx)
        mask = 1 << idx
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if val != center[pos]:
                        neighbor = center.copy()
                        neighbor[pos] = val
                        nidx = word_to_idx(neighbor)
                        mask |= 1 << nidx
        ball_masks[idx] = mask
        
    uncovered = (1 << total_words) - 1
    codewords = []
    while uncovered != 0:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            covered = ball_masks[idx] & uncovered
            count = covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1:
            break
        codewords.append(idx_to_word(best_idx))
        uncovered &= ~ball_masks[best_idx]
    return codewords
# EVOLVE-BLOCK-END
