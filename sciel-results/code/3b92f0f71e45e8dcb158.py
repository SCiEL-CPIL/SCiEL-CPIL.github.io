import numpy as np

def construct(q, n, radius):
    total = q ** n
    # Generate all words
    # Using base-q conversion
    all_words = np.zeros((total, n), dtype=np.uint8)
    for i in range(total):
        v = i
        for j in range(n):
            all_words[i, j] = v % q
            v //= q
            
    # Vectorized distance calculation
    # w1: (total, 1, n), w2: (1, total, n)
    w1 = all_words[:, np.newaxis, :]
    w2 = all_words[np.newaxis, :, :]
    
    # Compute Hamming distances
    # diff is boolean array (total, total, n)
    diff = (w1 != w2)
    dist = diff.sum(axis=2)
    
    # Coverage matrix: cover_matrix[i, j] is True if word j is within radius of word i
    cover_matrix = dist <= radius
    
    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    
    while uncovered.any():
        # counts[i] = number of uncovered words covered by codeword i
        counts = cover_matrix @ uncovered
        
        best_idx = np.argmax(counts)
        
        if counts[best_idx] == 0:
            # Should not happen if cover_matrix is correct and uncovered is not empty,
            # unless there are no candidates? But every word covers itself.
            break
            
        codewords.append(all_words[best_idx].tolist())
        
        # Update uncovered
        uncovered[cover_matrix[best_idx]] = False
        
    return codewords
