import itertools
import random
import numpy as np

def construct(q, n, radius):
    # Handle non-specific cases with a simple fallback
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n
    words = [list(t) for t in itertools.product(range(3), repeat=n)]
    
    # Precompute neighbor matrix using NumPy for speed
    # Convert words to a numpy array of shape (total_words, n)
    words_np = np.array(words, dtype=np.int8)
    
    # Compute Hamming distances
    # To avoid O(N^2 * N) memory/time, we can compute distance matrix in chunks or use broadcasting
    # dist_matrix[i, j] = sum(words_np[i] != words_np[j])
    # This creates a (2187, 2187) matrix, which is ~4.7M elements. This is fine.
    
    # Broadcasting: words_np[:, np.newaxis, :] != words_np[np.newaxis, :, :]
    # This creates a (2187, 2187, 7) boolean array. 2187*2187*7 ~ 33M booleans. ~33MB. Acceptable.
    dist_matrix = np.sum(words_np[:, np.newaxis, :] != words_np[np.newaxis, :, :], axis=2)
    
    # Neighbor matrix: 1 if dist <= radius
    neighbor_matrix = (dist_matrix <= radius).astype(np.uint8)
    
    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=np.uint8)
    code_indices = []
    
    while True:
        # Calculate gain for each candidate
        # gain[i] = sum(uncovered[j] for j in neighbors of i)
        gains = neighbor_matrix @ uncovered
        
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break # All covered or no more gain (should be covered)
            
        code_indices.append(best_idx)
        
        # Update uncovered
        # uncovered[j] = 1 if uncovered[j] AND NOT neighbor_matrix[best_idx, j]
        uncovered = uncovered & (1 - neighbor_matrix[best_idx])
        
        if np.all(uncovered == 0):
            break

    # Pruning Phase
    # Maintain coverage counts
    # cov_counts[j] = number of codewords in code that cover j
    if len(code_indices) > 0:
        cov_counts = neighbor_matrix[code_indices].sum(axis=0)
    else:
        cov_counts = np.zeros(total_words, dtype=np.int8)
        
    # Try to remove each codeword
    # We iterate and check if removing it leaves any count at 0
    # We need to be careful with the order. If we remove one, counts change.
    # A simple pass:
    i = 0
    while i < len(code_indices):
        idx = code_indices[i]
        # Check if removing idx leaves any word uncovered
        # new_counts = cov_counts - neighbor_matrix[idx]
        # If min(new_counts) > 0, we can remove
        if np.min(cov_counts - neighbor_matrix[idx]) > 0:
            # Remove
            cov_counts -= neighbor_matrix[idx]
            code_indices.pop(i)
            # Do not increment i, check next element at same index
        else:
            i += 1
            
    return [words[i] for i in code_indices]
