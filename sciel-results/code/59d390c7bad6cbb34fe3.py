# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin vs 150s kill

    if q == 3 and n == 7 and radius == 1:
        return _ternary7_r1(deadline)

    # Generic fallback: greedy set cover with NumPy
    return _generic_greedy(q, n, radius, deadline)


def _ternary7_r1(deadline):
    n = 7
    q = 3
    total = q ** n  # 2187

    # Precompute ball indices for every word
    # For a word w, ball(w) = {w} U {w with one coord changed}
    # Index = sum w[i] * 3^(n-1-i)
    # For coord i, step = 3^(n-1-i)
    steps = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute for each word its ball as a list of indices
    # But 2187 * 22 = 48114 entries, fine
    # Actually we'll compute on the fly with a helper for speed in greedy
    # Better: precompute ball_indices as a 2D array? 2187 x 22, but variable... 
    # Use a flat approach: for each word, ball is word itself + 21 neighbors

    # Precompute all words
    all_words = np.array(np.unravel_index(np.arange(total), [3] * n)).T  # (2187, 7)

    # Precompute ball membership: for each word w (index), which codeword indices cover it?
    # We need: given a set of chosen codeword indices, which words are covered?
    # Approach: for each codeword index c, ball(c) = set of word indices within dist 1.
    # Precompute ball as a list of arrays, or a single big array with offsets.

    # For each word index c, the ball contains:
    #   c itself
    #   for each coord i: c with coord i changed to (c[i]+1)%3 and (c[i]+2)%3
    # That's 1 + 2*7 = 15 neighbors + self = 16? No wait:
    # dist 0: 1 word (itself)
    # dist 1: for each of 7 coords, 2 alternative values = 14 words
    # Total ball size = 15

    # Wait, radius 1 means dist <= 1, so 1 + 14 = 15 words per ball.

    # Precompute ball_indices: list of 2187 arrays, each of length 15
    ball_indices = np.empty((total, 15), dtype=np.int32)
    for c in range(total):
        w = all_words[c]
        ball = [c]
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val != orig:
                    w2 = w.copy()
                    w2[i] = new_val
                    idx = int(np.dot(w2, [3 ** (n - 1 - j) for j in range(n)]))
                    ball.append(idx)
        ball_indices[c] = ball

    # Verify: 2187 * 15 = 32805, but total words is 2187. Each word is in 15 balls.
    # 2187 * 15 = 32805 = 2187 * 15. OK, each word appears in exactly 15 balls.

    # Greedy set cover
    # covered: boolean array of length 2187
    covered = np.zeros(total, dtype=bool)
    # count how many times each word is covered (for marginal gain)
    # Actually for greedy: gain = number of NEW words covered by adding codeword c
    # = sum over ball(c) of (1 - covered[word])
    
    # Use a counter: for each word, how many chosen codewords cover it
    cover_count = np.zeros(total, dtype=np.int32)
    
    chosen = []
    
    # For speed, precompute ball_indices as a flat structure
    # ball_indices is (2187, 15) int32
    
    # Greedy loop
    while True:
        if time.time() > deadline - 30:  # leave time for local search
            break
        if covered.all():
            break
        
        # Compute marginal gain for each unchosen codeword
        # gain[c] = number of words in ball(c) that are not yet covered
        # = 15 - sum(covered[ball_indices[c]])
        
        # Vectorized: for each c, sum of covered over its ball
        # covered is (2187,), ball_indices is (2187, 15)
        # We need: for each row c, sum(covered[ball_indices[c]])
        # This is a gather operation
        
        # Using advanced indexing: covered[ball_indices] gives (2187, 15)
        # Then sum along axis 1
        # But we only care about unchosen codewords
        
        if len(chosen) == 0:
            # First iteration: all codewords have gain 15
            # Pick any, say 0
            chosen.append(0)
            cover_count[ball_indices[0]] += 1
            covered[ball_indices[0]] = True
            continue
        
        # Compute gains for all codewords not yet chosen
        # To speed up, we can compute for all and mask
        ball_cov = covered[ball_indices]  # (2187, 15)
        gains = (1 - ball_cov).sum(axis=1)  # (2187,)
        
        # Mask out already chosen
        for c in chosen:
            gains[c] = -1
        
        # Pick the one with max gain
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            # Should not happen if not fully covered, but safety
            break
        
        chosen.append(best)
        cover_count[ball_indices[best]] += 1
        covered[ball_indices[best]] = True
    
    # Local search polish
    if time.time() < deadline - 5:
        chosen = _local_search_polish(chosen, ball_indices, covered, cover_count, total, deadline)
    
    # Convert to list of codewords
    result = [all_words[c].tolist() for c in chosen]
    return result


def _local_search_polish(chosen, ball_indices, covered, cover_count, total, deadline):
    chosen_set = set(chosen)
    chosen_list = list(chosen)
    
    # Try to remove codewords that are redundant
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for i, c in enumerate(list(chosen_list)):
            if time.time() > deadline - 3:
                break
            # Check if removing c leaves all its ball covered by others
            # A word w in ball(c) is still covered if cover_count[w] > 1
            # (i.e., some other chosen codeword also covers w)
            ball = ball_indices[c]
            if np.all(cover_count[ball] > 1):
                # Safe to remove
                chosen_list.remove(c)
                cover_count[ball] -= 1
                improved = True
    
    # Try to add a codeword and remove multiple others? Too expensive.
    # Instead: try swapping - remove one, add one if it helps
    # Actually, let's try: for each codeword, see if replacing it with another
    # reduces the total. This is complex; skip for now.
    
    return chosen_list


def _generic_greedy(q, n, radius, deadline):
    total = q ** n
    # This is a slow fallback; for the specific task we use _ternary7_r1
    # Simple greedy
    all_words = np.array(np.unravel_index(np.arange(total), [q] * n)).T
    covered = np.zeros(total, dtype=bool)
    chosen = []
    
    while not covered.all() and time.time() < deadline - 5:
        best_gain = -1
        best_c = -1
        for c in range(total):
            if c in set(chosen):
                continue
            # Compute ball of c within radius
            # For small radius, enumerate
            ball = _ball_generic(c, all_words[c], q, n, radius)
            gain = sum(1 for w in ball if not covered[w])
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1 or best_gain <= 0:
            break
        chosen.append(best_c)
        ball = _ball_generic(best_c, all_words[best_c], q, n, radius)
        for w in ball:
            covered[w] = True
    
    return [all_words[c].tolist() for c in chosen]


def _ball_generic(c_idx, word, q, n, radius):
    # Enumerate all words within Hamming distance radius
    # For small radius, use itertools
    import itertools
    ball = [c_idx]
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(q), repeat=r):
                w2 = list(word)
                for pos, val in zip(positions, values):
                    if val != w2[pos]:
                        w2[pos] = val
                idx = int(np.dot(w2, [q ** (n - 1 - j) for j in range(n)]))
                ball.append(idx)
    return ball
# EVOLVE-BLOCK-END
