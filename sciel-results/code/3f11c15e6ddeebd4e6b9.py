# EVOLVE-BLOCK-START
import numpy as np

def construct(q, n, radius):
    # Only the canonical task parameters are guaranteed (q=3, n=8, radius=1).
    # For other parameters, fall back to the simple product construction.
    if q != 3 or n != 8 or radius != 1:
        # Fallback: direct product of ternary tetracode on first 4 coords
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Canonical case: q=3, n=8, radius=1
    # Total words: 3^8 = 6561
    N = 3 ** n  # 6561
    
    # Map each integer 0..N-1 to a tuple of n digits in base q
    # We need a fast way to get the digit representation and to compute
    # neighbors (words at Hamming distance <= 1).
    
    # Precompute all words as a numpy array of shape (N, n)
    words = np.zeros((N, n), dtype=np.int8)
    for i in range(N):
        v = i
        for j in range(n):
            words[i, j] = v % q
            v //= q
    
    # Precompute index lookup: for a word (as a tuple of digits), get its index
    # We can encode a word as an integer in base q, and the index is that integer
    # So word i has digits words[i, :], and its "code" is i.
    
    # Build coverage: for each word i, the set of words j at Hamming distance <= 1
    # Instead of a full N x N boolean matrix (43 MB, fine), let's compute on the fly
    # using a smarter approach.
    
    # Approach: For each candidate codeword i, the number of newly covered words
    # is the count of uncovered words in the Hamming ball of radius 1 around i.
    
    # Let's precompute the Hamming ball neighbors for each word.
    # Ball of radius 1 around word i: the word itself + all words that differ in
    # exactly one position (n * (q-1) = 8 * 2 = 16 neighbors).
    
    # Precompute neighbor indices: neighbors[i] is a list of indices at dist <= 1 from i
    # Total entries: N * (1 + n*(q-1)) = 6561 * 17 = 111537, very manageable.
    
    neighbors = []
    for i in range(N):
        ball = [i]
        w = words[i]
        for pos in range(n):
            for d in range(q):
                if d == w[pos]:
                    continue
                # Build the modified word and find its index
                # The index is the base-q number with digits modified at pos
                idx = i + d * (q ** pos) - w[pos] * (q ** pos)
                ball.append(idx)
        neighbors.append(ball)
    
    # Convert to a numpy array for fast operations
    # neighbors is a list of lists of varying length? No, all length 1 + n*(q-1) = 17
    neighbors_arr = np.array(neighbors, dtype=np.int32)  # shape (N, 17)
    
    # Greedy set cover
    uncovered = np.ones(N, dtype=bool)
    code = []
    
    while uncovered.any():
        # For each candidate i, count how many uncovered words are in its ball
        # marginal[i] = sum of uncovered[neighbors_arr[i, :]]
        # Use fancy indexing: uncovered[neighbors_arr] gives shape (N, 17)
        ball_coverage = uncovered[neighbors_arr]  # shape (N, 17), boolean
        marginal = ball_coverage.sum(axis=1)
        
        best = int(np.argmax(marginal))
        if marginal[best] == 0:
            break
        
        code.append(words[best].tolist())
        uncovered[neighbors_arr[best]] = False
    
    return code
# EVOLVE-BLOCK-END
