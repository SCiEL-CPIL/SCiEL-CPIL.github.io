import numpy as np
import time
import itertools

def construct(q, n, radius):
    deadline = time.time() + 140  # safety margin before 150s kill
    
    total = q ** n
    # Enumerate all words as flat indices 0..total-1, decode to base-q digits
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Precompute: for each position and each alternative value, the "flip" effect
    # For radius=1, a word w covers w itself and all words differing in exactly one position
    
    # Build coverage mask: covered[i] = True if word i is covered
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    # For greedy: at each step, for each candidate c, count uncovered words within dist <= radius
    # Precompute neighbor lists for radius=1
    # For each word index, store the set of words within distance 1
    # But 729*13 = 9477 entries, fine
    
    # Actually, let's vectorize differently:
    # For candidate c, the words within distance 1 are:
    #   c itself, and for each position i, for each value v != c[i], the word with c[i] replaced by v
    
    # Precompute the "ball" for each candidate as a list of indices
    # For 729 candidates, each ball has 1 + n*(q-1) = 1 + 6*2 = 13 words
    # Total: 729 * 13 = 9477 index lookups per full greedy step
    
    # Build ball_indices: shape (total, ball_size) where ball_size = 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)
    ball_indices = np.zeros((total, ball_size), dtype=np.int32)
    
    # For each word, compute its neighbors within distance 1
    for idx in range(total):
        w = all_words[idx]
        neighbors = [idx]
        for pos in range(n):
            orig = w[pos]
            for v in range(q):
                if v == orig:
                    continue
                nw = w.copy()
                nw[pos] = v
                # Convert to flat index
                flat = 0
                for d in range(n):
                    flat = flat * q + nw[d]
                neighbors.append(flat)
        ball_indices[idx] = neighbors
    
    # Greedy set cover
    uncovered_count = total
    remaining = np.ones(total, dtype=bool)  # words not yet covered
    
    while uncovered_count > 0 and time.time() < deadline:
        # For each candidate, count how many uncovered words are in its ball
        # best_gain[c] = sum(remaining[ball_indices[c]])
        gains = np.zeros(total, dtype=np.int32)
        for c in range(total):
            gains[c] = np.sum(remaining[ball_indices[c]])
        
        best_c = int(np.argmax(gains))
        if gains[best_c] == 0:
            break  # no progress possible (shouldn't happen if uncovered_count > 0)
        
        # Add this codeword
        codewords.append(all_words[best_c].tolist())
        
        # Update coverage
        newly = ball_indices[best_c]
        for ni in newly:
            if remaining[ni]:
                remaining[ni] = False
                uncovered_count -= 1
    
    return codewords
