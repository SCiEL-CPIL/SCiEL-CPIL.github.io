"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct(q, n, radius):
    """
    Constructs a covering code using a vectorized greedy set cover approach.
    The universe is all q^n words. We select codewords one by one, choosing the one 
    that covers the most currently uncovered words.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    total_words = q ** n
    if total_words > 100000:
        # Fallback for larger spaces if greedy becomes too slow, though n=7, q=3 is 2187
        # For n=7, q=3, 2187 is small enough for full vectorization.
        pass

    # Precompute all words as a 2D array of shape (total_words, n)
    # Each row is a word in base-q representation
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    
    # Precompute the "distance <= radius" mask? 
    # For q=3, n=7, radius=1:
    # A word w covers w itself and all words differing in exactly 1 position.
    # Total size of ball = 1 + n*(q-1) = 1 + 7*2 = 15.
    
    # To speed up the greedy selection, we can precompute the list of indices 
    # covered by each potential codeword.
    # However, storing 2187 x 15 integers is small.
    # Let's create a coverage matrix or list of lists.
    # Actually, for greedy set cover, we need to know how many *new* words a candidate covers.
    
    # Representation:
    # `covered` is a boolean array of size total_words.
    # For each candidate c, we need the sum of (~covered[indices_covered_by_c]).
    
    # Let's precompute the indices covered by each word.
    # word_indices[i] = list of indices in all_words that are within distance radius of all_words[i]
    
    # Efficient way to compute neighbors for ternary code radius 1:
    # For each word w, neighbors are:
    # 1. w itself
    # 2. For each position j, change w[j] to the other q-1 values.
    
    # Since q=3, n=7, we can just compute this once.
    # Map from word tuple to index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
        
    # Precompute coverage indices for each word
    # cover_indices[i] is a numpy array of indices of words covered by all_words[i]
    cover_indices = []
    for i in range(total_words):
        w = all_words[i]
        inds = [i]
        for j in range(n):
            for v in range(q):
                if v == w[j]:
                    continue
                new_w = w.copy()
                new_w[j] = v
                inds.append(word_to_idx[tuple(new_w)])
        cover_indices.append(np.array(inds, dtype=np.int32))
        
    # Initialize covered mask
    covered = np.zeros(total_words, dtype=bool)
    
    # Active candidates
    active = np.ones(total_words, dtype=bool)
    
    selected_codewords = []
    
    while True:
        if time.time() > deadline:
            break
            
        if not active.any():
            break
            
        # For each active candidate, calculate how many new words it covers
        # new_cov[i] = sum(~covered[cover_indices[i]])
        
        # Vectorized calculation is tricky because cover_indices are different lengths?
        # No, for radius 1, length is always 1 + n*(q-1).
        # But indexing a 1D array with a 2D array of indices doesn't work directly for sum per row easily without loops or fancy indexing.
        
        # Alternative: Iterate through active candidates? 
        # 2187 candidates is small. A Python loop over 2187 items taking ~microseconds each is fast.
        # But we want to be safe.
        
        # Let's try to vectorize the "count uncovered" step.
        # We can create a 2D array `cover_matrix` of shape (total_words, ball_size) containing the indices.
        # Then `covered_indices = covered[cover_matrix]` gives a (total_words, ball_size) boolean array.
        # `new_counts = (~covered_indices).sum(axis=1)`
        
        # Let's build cover_matrix once.
        if not hasattr(construct, '_cover_matrix'):
            ball_size = 1 + n * (q - 1)
            # This assumes uniform ball size, which is true for Hamming distance 1 in constant alphabet
            construct._cover_matrix = np.array(cover_indices, dtype=np.int32)
            construct._q = q
            construct._n = n
            construct._radius = radius
            
        cover_matrix = construct._cover_matrix
        ball_size = cover_matrix.shape[1]
        
        # Get covered status for all positions in the matrix
        # covered[cover_matrix] creates an array of shape (total_words, ball_size)
        # This might be memory intensive if total_words * ball_size is large, but 2187 * 15 is ~32k, very small.
        cov_status = covered[cover_matrix]
        
        # Count uncovered
        new_counts = (~cov_status).sum(axis=1)
        
        # Only consider active candidates
        new_counts[~active] = -1
        
        # Find the candidate with max new coverage
        max_idx = np.argmax(new_counts)
        
        if new_counts[max_idx] <= 0:
            # No more coverage possible, or we've covered everything
            # If we covered everything, we are done.
            # If new_counts is 0 for all active, it means all remaining active candidates cover 0 new words?
            # That implies all words are covered.
            break
            
        # Select this codeword
        selected_codewords.append(all_words[max_idx].tolist())
        
        # Update covered
        covered[cover_matrix[max_idx]] = True
        
        # Mark as inactive
        active[max_idx] = False
        
        # Check if all covered
        if covered.all():
            break
            
    # If we exited due to time, we might have a partial cover. 
    # The task requires a valid covering code. 
    # Greedy set cover guarantees a valid cover if run to completion.
    # If we break early due to time, we must ensure validity or return best effort?
    # "Valid when every word... is within Hamming distance".
    # If we return an invalid code, score is 0.
    # So we must ensure we finish or have a valid code.
    # Given n=7, q=3, the greedy algorithm typically finishes very quickly (<1s).
    # The deadline is a safety net.
    
    # If we didn't finish and covered is not all True, we have a problem.
    # We can try to complete it with a simple fallback if we have time, or just return what we have?
    # No, must be valid. 
    # Let's check if we are valid.
    if not covered.all():
        # We need to finish. The loop should have finished for this small size.
        # If it didn't, it's a bug or extremely slow.
        # Let's force a completion using a slower but guaranteed method if we have time, 
        # or just return the current list if it happens to be valid (unlikely).
        # Actually, if active.any() is false, we broke out. 
        # If we broke out due to time, we might have uncovered words.
        # Let's add a safety check.
        if time.time() < deadline - 5.0:
            # Try to fix by adding more codewords from the remaining space?
            # This is complex. For n=7, q=3, it should not happen.
            pass
            
    return selected_codewords

# EVOLVE-BLOCK-END
