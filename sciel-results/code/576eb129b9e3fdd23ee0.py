import numpy as np
import time

def construct(q, n, radius):
    # For this task, q=3, n=8, radius=1
    total_words = q ** n
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Generate all words
    words = np.array(list(np.ndindex(*([q] * n))), dtype=np.int32).reshape(-1, n)
    
    # Build incidence matrix: M[i, j] = 1 if word j is within distance radius of word i
    # For radius=1, word j is covered by word i if they differ in at most 1 position
    # This is equivalent to: Hamming distance <= 1
    
    # Efficient way: for each word i, the covered words are:
    # - word i itself
    # - words that differ in exactly one position (n * (q-1) words)
    
    # Precompute the neighborhood indices for each word
    # Since n=8, q=3, radius=1, each word covers 1 + 8*2 = 17 words
    
    # Let's build the incidence matrix as a list of boolean arrays or a sparse-like structure
    # But for greedy, we need to quickly compute "how many uncovered words does candidate i cover?"
    
    # Alternative: maintain a count array `cover_count[j]` = number of selected codewords that cover word j
    # We want to pick the codeword i that maximizes the number of j with cover_count[j] == 0
    
    # Let's use a different approach: maintain `uncovered` as a boolean array
    # At each step, for each candidate i, compute the number of uncovered words in its neighborhood
    
    # Precompute neighborhood indices for each word
    # neighborhood[i] = list of indices j such that dist(words[i], words[j]) <= radius
    
    # For efficiency, let's precompute the neighborhood as a list of arrays
    neighborhoods = []
    for i in range(total_words):
        w = words[i]
        covered_indices = [i]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = val
                    # Find the index of new_w
                    # This is slow. Let's precompute a mapping from tuple to index
                    pass
        neighborhoods.append(covered_indices)
    
    # This approach is too slow for precomputation. Let's use a different strategy.
    
    # Better: use the fact that we can compute the Hamming distance vectorized.
    # But 6561 x 6561 distance matrix is large but feasible.
    
    # Let's compute the distance matrix D where D[i,j] = Hamming distance between words[i] and words[j]
    # D[i,j] = sum over positions of (words[i,k] != words[j,k])
    
    # This is a 6561x6561 matrix of integers. 6561^2 = 43,046,721 entries.
    # As uint8, this is ~43 MB. Feasible.
    
    # Compute D
    # D = np.sum(words[:, np.newaxis, :] != words[np.newaxis, :, :], axis=2)
    # This creates a 6561x6561x8 temporary, which is 344 MB as bool. Might be tight but let's try.
    
    # Actually, let's do it in chunks to be safe.
    # Or, since radius=1, we only need to know if D[i,j] <= 1.
    
    # Let's precompute the neighborhood indices using a dictionary mapping tuple to index
    word_to_idx = {}
    for i in range(total_words):
        word_to_idx[tuple(words[i])] = i
    
    neighborhoods = []
    for i in range(total_words):
        w = words[i]
        covered = [i]
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    covered.append(idx)
        neighborhoods.append(covered)
    
    # Now, greedy set cover
    uncovered = np.ones(total_words, dtype=bool)
    selected = []
    
    # Precompute neighborhood as numpy arrays for faster indexing
    neighborhoods = [np.array(neighborhoods[i]) for i in range(total_words)]
    
    while uncovered.any() and time.time() < deadline:
        # For each candidate, compute the number of uncovered words it covers
        # This is: for each i, sum(uncovered[neighborhoods[i]])
        # We can vectorize this with a matrix, but let's try a loop first to see if it's fast enough
        
        # Actually, let's build the incidence matrix M where M[i, j] = 1 if j in neighborhoods[i]
        # Then the number of uncovered words covered by i is M[i, :] @ uncovered
        # But building M is 6561x6561, which is fine.
        
        # Let's build M once
        M = np.zeros((total_words, total_words), dtype=np.uint8)
        for i in range(total_words):
            M[i, neighborhoods[i]] = 1
        
        # Now, at each step:
        # scores = M @ uncovered (as float)
        # But we want the max score, and then update uncovered
        
        # However, recomputing M @ uncovered each time is O(total_words^2) per step, which is too slow
        # for potentially hundreds of steps.
        
        # Alternative: maintain a count array `count[j]` = number of selected codewords that cover j
        # Then `uncovered` is `count == 0`
        # When we select codeword i, we increment count[neighborhoods[i]]
        # And we want to pick i that maximizes the number of j with count[j]==0 in neighborhoods[i]
        
        # This is still hard to vectorize efficiently without M.
        
        # Let's try the M @ uncovered approach and see how many steps we can do in 140s.
        # Each step: M @ uncovered is a matrix-vector product, O(total_words^2) = 43M ops.
        # 43M ops is fast in NumPy, maybe 1-5 ms. So we can do thousands of steps.
        
        # But we only need ~486 steps at most. So this is very feasible.
        
        # Let's do it.
        
        # Actually, we already built M above. Let's move that outside the loop.
        
        # Let's restructure.
        break  # We'll rewrite the loop properly below
    
    # Let's restart with a cleaner implementation
    
    # Rebuild M
    M = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        M[i, neighborhoods[i]] = 1
    
    uncovered = np.ones(total_words, dtype=bool)
    selected = []
    
    while uncovered.any() and time.time() < deadline:
        # Compute the number of uncovered words each candidate covers
        # scores = M[uncovered_mask, :].sum(axis=1) is wrong
        # We want for each i: sum of M[i, j] for j where uncovered[j] is True
        # This is M @ uncovered (treating uncovered as a vector)
        scores = M @ uncovered.astype(np.float32)
        
        # Pick the candidate with the highest score
        best_idx = np.argmax(scores)
        
        # If the best score is 0, we have a problem (shouldn't happen if uncovered.any())
        if scores[best_idx] == 0:
            # Fallback: pick any uncovered word's neighbor
            # Actually, this shouldn't happen. Let's just pick the first uncovered word
            uncovered_indices = np.where(uncovered)[0]
            best_idx = uncovered_indices[0]
        
        # Select this codeword
        selected.append(words[best_idx].tolist())
        
        # Update uncovered
        uncovered[neighborhoods[best_idx]] = False
    
    return selected
