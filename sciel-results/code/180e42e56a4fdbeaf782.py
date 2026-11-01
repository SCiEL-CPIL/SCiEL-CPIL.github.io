import numpy as np
import itertools
import random

def _all_words(q, n):
    total = q ** n
    words = np.zeros((total, n), dtype=np.uint8)
    for i in range(total):
        v = i
        for j in range(n):
            words[i, j] = v % q
            v //= q
    return words

def _build_coverage_matrix(q, n, radius):
    total = q ** n
    words = _all_words(q, n)
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # C[i, j] is True if word j is covered by codeword i
    C = np.zeros((total, total), dtype=bool)
    
    for i in range(total):
        w = words[i]
        # Self
        C[i, i] = True
        
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_w = w.copy()
                    new_w[pos] = val
                    idx = word_to_idx.get(tuple(new_w))
                    if idx is not None:
                        C[i, idx] = True
                        
        # If radius > 1, we would need to expand further, but task is radius 1
        # The task says "at most one position", so radius is effectively 1.
        # The interface takes radius, but for this specific task it's 1.
        # We should handle general radius if possible, but for 6 matches q=3, radius=1 is the constraint.
        # If radius > 1, the logic above is insufficient.
        # However, the task description says "differ ... in at most one position".
        # So radius is fixed to 1 for this specific problem instance.
        # I will assume radius=1 for the optimization.
        
    return C, words

def _greedy_set_cover(C, words, restarts=10, rng_seed=42):
    total = len(words)
    best_codewords = None
    best_len = total
    
    rng = np.random.default_rng(rng_seed)
    
    for _ in range(restarts):
        uncovered = np.ones(total, dtype=bool)
        codewords = []
        
        while uncovered.any():
            # Calculate coverage count for each candidate
            # C[:, uncovered] gives a boolean matrix of shape (total, num_uncovered)
            # Summing along axis 1 gives the count of uncovered words covered by each candidate
            counts = C[:, uncovered].sum(axis=1)
            
            max_count = counts.max()
            if max_count == 0:
                break
            
            # Find all candidates with max_count
            best_candidates = np.where(counts == max_count)[0]
            
            # Randomly pick one from best_candidates to break symmetry
            if len(best_candidates) > 1:
                # Use random choice to explore different paths
                best_c = best_candidates[rng.integers(len(best_candidates))]
            else:
                best_c = best_candidates[0]
            
            codewords.append(best_c)
            uncovered[C[best_c]] = False
        
        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = codewords.copy()
            
    return [words[i].tolist() for i in best_codewords]

def construct(q, n, radius):
    # For the specific task: q=3, n=6, radius=1
    # If radius is not 1, we might need a different coverage builder, 
    # but the problem statement specifies "at most one position".
    
    total = q ** n
    
    if radius == 1:
        C, words = _build_coverage_matrix(q, n, radius)
        # Run greedy with restarts
        # 100 restarts should be fast enough (each restart is ~70 steps * 729*729 ops ~ 3.7M ops, * 100 = 370M ops. NumPy can handle this in < 120s)
        # Actually, C[:, uncovered].sum(axis=1) is O(total * num_uncovered).
        # Initially num_uncovered=729. Later it shrinks.
        # Total operations per restart is roughly sum of num_uncovered over steps.
        # This is much less than total^2.
        
        # Let's do 50 restarts to be safe and fast.
        result = _greedy_set_cover(C, words, restarts=50, rng_seed=123)
        return result
    else:
        # Fallback for other radii (not expected for this task)
        # Simple greedy without optimization
        words = _all_words(q, n)
        word_to_idx = {tuple(w): i for i, w in enumerate(words)}
        
        def get_neighbors(w):
            neighbors = {tuple(w)}
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig: continue
                    new_w = w.copy()
                    new_w[pos] = val
                    neighbors.add(tuple(new_w))
            # If radius > 1, this is incomplete. 
            # Given the task constraints, we assume radius=1.
            return neighbors
        
        uncovered = set(range(total))
        codewords = []
        while uncovered:
            best_c = -1
            best_count = -1
            for c in range(total):
                nbs = get_neighbors(words[c])
                count = len(uncovered.intersection(nbs))
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c == -1: break
            codewords.append(words[best_c].tolist())
            uncovered -= get_neighbors(words[best_c])
        return codewords
