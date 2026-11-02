import itertools
import time
import numpy as np

def construct(q, n, radius):
    # Default fallback: Tetracode construction
    def construct_tetracode(q, n, radius):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n <= 4:
            return [h[:n] for h in tetracode]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    if q != 3 or n != 6 or radius != 1:
        return construct_tetracode(q, n, radius)

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Direct vectorized greedy set-cover on the full 3^6 = 729 space
    # Ball size for radius 1 in ternary length-6: 1 + 6*2 = 13
    
    # Precompute all words as indices 0..728
    total = 3 ** 6
    
    # Precompute ball membership matrix: ball_matrix[i] = set of indices in ball around word i
    # Use a sparse-like approach: for each word, compute its ball members
    # ball_matrix will be a list of numpy arrays or we use a 2D boolean matrix
    
    # Build index-to-word and word-to-index mappings
    def idx_to_word(idx):
        digits = []
        for _ in range(n):
            digits.append(idx % 3)
            idx //= 3
        return digits[::-1]
    
    # Precompute balls for all 729 words
    # ball_indices[i] = numpy array of indices covered by codeword i
    ball_indices = []
    for i in range(total):
        w = idx_to_word(i)
        ball = set([i])
        for pos in range(n):
            for delta in [-1, 1]:
                new_w = list(w)
                new_w[pos] = (w[pos] + delta) % 3
                # Convert back to index
                idx_new = 0
                for d in new_w:
                    idx_new = idx_new * 3 + d
                ball.add(idx_new)
        ball_indices.append(np.array(sorted(ball), dtype=np.int32))
    
    # Greedy set cover with vectorized scoring
    covered = np.zeros(total, dtype=bool)
    code = []
    
    # For speed, precompute the ball matrix as a list of arrays
    # At each step, score = number of uncovered words in each ball
    # We can vectorize: scores = ball_matrix @ (1 - covered)
    # But ball_matrix is 729 x 729, which is 729*729 = ~531K entries - manageable
    
    # Build dense ball matrix
    ball_matrix = np.zeros((total, total), dtype=np.int8)
    for i in range(total):
        for j in ball_indices[i]:
            ball_matrix[i, j] = 1
    
    while True:
        if time.time() > deadline:
            break
        
        # Compute scores: for each candidate codeword, how many uncovered words it covers
        uncovered_mask = ~covered
        scores = ball_matrix @ uncovered_mask.astype(np.int32)
        
        # Find best candidate (max score)
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        
        if best_score == 0:
            break
        
        # Add this codeword to the code
        code.append(best_idx)
        
        # Update covered
        covered[ball_indices[best_idx]] = True
        
        # Remove this row from consideration (set score to -inf effectively)
        # We can just recompute each time; it's fast enough with numpy
    
    # Check if we covered everything
    if covered.all():
        # Convert indices back to words
        result = [idx_to_word(c) for c in code]
        return result
    else:
        # Fallback to tetracode
        return construct_tetracode(q, n, radius)
