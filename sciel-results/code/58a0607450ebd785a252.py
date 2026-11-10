import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    total_words = q ** n
    
    # Precompute all words as a 2D numpy array (total_words, n)
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Precompute the "ball" of each word: indices of all words within Hamming distance <= radius
    # For radius=1, ball size = 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)
    
    # Build coverage matrix: coverage[i, j] = 1 if word i is within radius of word j
    # This is a (total_words, total_words) matrix - might be large but for 729x729 it's fine
    # Actually, let's use a different approach: for each word index, store the list of words it covers
    
    # For efficiency, precompute for each word index, the set of word indices it covers
    # We'll use a numpy approach: compute Hamming distances efficiently
    
    # Actually for 729 words, we can precompute the full coverage matrix
    # coverage_matrix[i] = set of indices covered by word i
    # Let's use a boolean matrix of shape (total_words, total_words)
    
    # Compute Hamming distance matrix
    # diff[i,j] = number of positions where all_words[i] != all_words[j]
    # For 729x729x6 this is manageable
    
    # More efficient: for each word, compute which words are at distance <= radius
    # Using broadcasting
    # diff = (all_words[:, None, :] != all_words[None, :, :]).sum(axis=2)  # (729, 729)
    # This is 729*729*6 comparisons = ~3.2M, fine
    
    diff = (all_words[:, None, :].astype(np.int16) != all_words[None, :, :].astype(np.int16)).sum(axis=2)
    coverage_matrix = (diff <= radius).astype(np.uint8)  # (total_words, total_words)
    
    # Each row i tells us which words are covered by codeword i
    # We want to find a minimum set of rows such that every column is covered by at least one selected row
    
    # Greedy set cover
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    
    while not covered.all():
        # Find the word that covers the most uncovered words
        # uncovered_mask = ~covered
        # scores = coverage_matrix[:, ~covered].sum(axis=1)
        # But this is expensive each iteration. Let's track uncovered count per candidate.
        
        remaining = ~covered
        if remaining.sum() == 0:
            break
        
        # For each candidate word i, count how many uncovered words it covers
        # scores[i] = coverage_matrix[i] & remaining -> sum
        # Use numpy: (coverage_matrix * remaining[None, :]).sum(axis=1)
        scores = (coverage_matrix * remaining[None, :]).sum(axis=1)
        
        # Pick the best
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
        
        code_indices.append(best_idx)
        covered |= coverage_matrix[best_idx, :]
        
        # Deadline check
        if time.time() > deadline - 10:
            break
    
    # Convert to list of codewords
    best_code = [all_words[i].tolist() for i in code_indices]
    
    # Local search: try to improve
    # 1. Try removing each codeword and see if coverage is still maintained
    # 2. Try replacing a codeword with a different one
    
    def is_valid(code_idx_list):
        if not code_idx_list:
            return total_words == 0
        # Check if all words are covered
        cov = np.zeros(total_words, dtype=bool)
        for ci in code_idx_list:
            cov |= coverage_matrix[ci, :]
        return cov.all()
    
    # Remove redundant codewords
    changed = True
    while changed:
        changed = False
        for i in range(len(code_indices)):
            if len(code_indices) <= 1:
                break
            # Try removing code_indices[i]
            remaining = [c for j, c in enumerate(code_indices) if j != i]
            if is_valid(remaining):
                code_indices = remaining
                changed = True
                break
        
        if time.time() > deadline - 5:
            break
    
    best_code = [all_words[i].tolist() for i in code_indices]
    best_size = len(best_code)
    
    # Local search: try to replace a codeword with another to enable removal of another
    # 2-opt: try swapping two codewords, or replacing one with a new one
    
    # Try: for each pair (i, j) in code_indices, try removing i and replacing with some k not in code
    # This is expensive. Let's try a simpler approach:
    # For each codeword i, try replacing it with a word k that covers more uncovered words
    
    # Actually, let's try: for each codeword in the code, try removing it and adding a different word
    # to see if we can maintain coverage with same or fewer words
    
    # More targeted: try to find if any codeword can be replaced by another that allows
    # removing a different codeword
    
    # Simple 1-opt: try replacing each codeword with every other word, check if we can then remove one more
    # This is O(k * 729) where k is code size, might be too slow
    
    # Let's try a randomized local search with deadline
    import random
    
    rng = random.Random(42)
    
    # Try greedy restarts with different orderings
    # Actually, let's try: start from current code, randomly remove a word, then greedily add back
    # to restore coverage, see if we end up with fewer words
    
    for restart in range(200):
        if time.time() > deadline - 2:
            break
        
        # Random perturbation: remove 1-2 random codewords
        current = list(code_indices)
        if len(current) < 2:
            break
        
        # Remove 1 random codeword
        remove_idx = rng.randint(0, len(current) - 1)
        perturbed = [c for j, c in enumerate(current) if j != remove_idx]
        
        # Greedily add back to restore coverage
        covered = np.zeros(total_words, dtype=bool)
        for ci in perturbed:
            covered |= coverage_matrix[ci, :]
        
        new_code = list(perturbed)
        while not covered.all():
            remaining = ~covered
            scores = (coverage_matrix * remaining[None, :]).sum(axis=1)
            # Exclude words already in new_code
            for ci in new_code:
                scores[ci] = 0
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            new_code.append(best_idx)
            covered |= coverage_matrix[best_idx, :]
        
        # Remove redundant from new_code
        changed2 = True
        while changed2:
            changed2 = False
            for i in range(len(new_code)):
                if len(new_code) <= 1:
                    break
                remaining_code = [c for j, c in enumerate(new_code) if j != i]
                if is_valid(remaining_code):
                    new_code = remaining_code
                    changed2 = True
                    break
        
        if len(new_code) < best_size:
            best_size = len(new_code)
            code_indices = new_code
            best_code = [all_words[i].tolist() for i in code_indices]
        
        elif len(new_code) == best_size and rng.random() < 0.3:
            # Same size, keep current (or could explore)
            pass
    
    # Final: try to remove more redundancies
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for i in range(len(code_indices)):
            if len(code_indices) <= 1:
                break
            remaining = [c for j, c in enumerate(code_indices) if j != i]
            if is_valid(remaining):
                code_indices = remaining
                changed = True
                break
    
    best_code = [all_words[i].tolist() for i in code_indices]
    
    # Ensure all words are within radius of some codeword (final validation)
    if not is_valid(code_indices):
        # Fallback: add words until valid
        covered = np.zeros(total_words, dtype=bool)
        for ci in code_indices:
            covered |= coverage_matrix[ci, :]
        while not covered.all():
            remaining = ~covered
            scores = (coverage_matrix * remaining[None, :]).sum(axis=1)
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            code_indices.append(best_idx)
            covered |= coverage_matrix[best_idx, :]
        best_code = [all_words[i].tolist() for i in code_indices]
    
    return best_code
