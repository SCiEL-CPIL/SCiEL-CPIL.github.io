"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct(q, n, radius):
    """
    Greedy set-cover algorithm using NumPy for efficient coverage counting.
    """
    if n <= 0:
        return [[]]
    
    total_words = q ** n
    
    # Helper to convert index to word
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Helper to convert word to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute neighbor lists
    neighbors = []
    for idx in range(total_words):
        center = idx_to_word(idx)
        nb_list = [idx]
        
        if radius == 0:
            pass
        elif radius == 1:
            for pos in range(n):
                original_val = center[pos]
                for val in range(q):
                    if val != original_val:
                        new_word = center.copy()
                        new_word[pos] = val
                        nb_list.append(word_to_idx(new_word))
        else:
            # Generic radius handling
            def generate_neighbors(center, remaining_radius, pos=0):
                if pos == n:
                    yield list(center)
                    return
                if remaining_radius >= 0:
                    yield from generate_neighbors(center, remaining_radius, pos + 1)
                if remaining_radius > 0:
                    for val in range(q):
                        if val != center[pos]:
                            old_val = center[pos]
                            center[pos] = val
                            yield from generate_neighbors(center, remaining_radius - 1, pos + 1)
                            center[pos] = old_val

            for neighbor in generate_neighbors(center, radius):
                 nb_list.append(word_to_idx(neighbor))
        
        # Remove duplicates
        neighbors.append(list(set(nb_list)))

    # Create the coverage matrix M
    # M[i, j] = 1 if j is in neighbors[i]
    M = np.zeros((total_words, total_words), dtype=np.int8)
    for i in range(total_words):
        for j in neighbors[i]:
            M[i, j] = 1

    # Greedy selection
    uncovered = np.ones(total_words, dtype=np.int8)
    codewords = []
    
    while np.sum(uncovered) > 0:
        # counts[i] = number of currently uncovered words covered by codeword i
        counts = M @ uncovered
        
        # Find the codeword that covers the most uncovered words
        best_idx = np.argmax(counts)
        
        if counts[best_idx] == 0:
             break
        
        # Add this codeword to our list
        codewords.append(idx_to_word(best_idx))
        
        # Update uncovered status
        uncovered[neighbors[best_idx]] = 0

    return codewords
# EVOLVE-BLOCK-END
