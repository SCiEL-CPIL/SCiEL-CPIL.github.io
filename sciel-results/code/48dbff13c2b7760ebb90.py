"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(q, n, radius):
    """
    Greedy set cover for ternary covering code K_3(8,1).
    Returns a list of codewords (each a list of n ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Generate all words in {0,1,2}^n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    N = len(all_words)  # 6561

    # For each word, compute the set of words within Hamming distance <= radius.
    # radius = 1: a word covers itself and all words differing in exactly one position.
    # Precompute for each word the indices it covers.
    # Instead of storing all covers (too large), we'll use a vectorized approach.

    # For radius 1, a codeword c covers:
    #   - c itself
    #   - for each position i, for each symbol s in {0,1,2} \ {c[i]}, the word with c[i] replaced by s
    # Total covered per codeword: 1 + n*(q-1) = 1 + 8*2 = 17

    # Strategy: greedy set cover.
    # We maintain a boolean array 'covered' of length N.
    # At each step, pick the codeword that covers the most uncovered words.
    # To speed this up, we can use a vectorized scoring.

    # Precompute: for each position i and each symbol change, what's the index mapping?
    # Actually, let's think differently. We can precompute for each word w, 
    # the list of words it covers (as indices into all_words).
    # But that's N * 17 = 6561 * 17 = 111537 entries, which is fine.

    # Let's precompute the cover sets.
    # For word at index idx, its value is all_words[idx].
    # The words it covers are:
    #   all_words[idx] itself
    #   for each pos i, for each alt in range(q) if alt != all_words[idx][i]:
    #       the word obtained by replacing position i with alt

    # We need a mapping from word-tuple to index.
    word_to_idx = {}
    for idx, w in enumerate(all_words):
        word_to_idx[tuple(w)] = idx

    # Precompute cover lists
    covers = []
    for idx in range(N):
        w = all_words[idx]
        cover_set = [idx]
        for i in range(n):
            for alt in range(q):
                if alt == w[i]:
                    continue
                nw = list(w)
                nw[i] = alt
                cover_set.append(word_to_idx[tuple(nw)])
        covers.append(cover_set)

    # Greedy set cover
    covered = np.zeros(N, dtype=bool)
    selected = []
    selected_indices = []

    # For efficiency, we'll track how many uncovered words each candidate covers.
    # Initially, each codeword covers 17 words (or fewer if some are the same, but they're distinct).
    # We'll use a priority approach: repeatedly pick the best, update.

    # To make this efficient, we can use the following:
    # maintain an array 'uncovered_count' of length N, where uncovered_count[j] = number of uncovered words in covers[j]
    # Initially all 17 (or the actual count).
    # When we select codeword c, for each word w in covers[c] that was uncovered, 
    #   for each codeword d whose cover contains w, decrement uncovered_count[d].

    # This is O(N * 17 * N) in the worst case per step, which is too slow.

    # Alternative: use a simpler greedy that at each step computes the score for all candidates.
    # Score of candidate j = number of uncovered words in covers[j].
    # We can compute this as: for each j, sum(1 for w in covers[j] if not covered[w])
    # This is O(N * 17) per step = 6561 * 17 = 111537 operations per step.
    # With ~400 steps, that's ~44 million operations, which should be fast in Python with numpy.

    # Let's use numpy for the scoring.
    # Represent covers as a sparse structure or use the fact that each cover is small.

    # Actually, let's just do it in a loop but with numpy for the covered check.
    # For each candidate j, score = len([w for w in covers[j] if not covered[w]])
    # We can vectorize this: for each j, we have covers[j] as an array of indices.
    # score_j = np.sum(~covered[covers[j]])

    # Let's precompute covers as a list of numpy arrays.
    covers_np = [np.array(c, dtype=np.int32) for c in covers]

    while not covered.all():
        if time.time() > deadline:
            break

        # Compute score for each candidate
        # We only need to consider candidates that haven't been selected yet.
        # But for simplicity, let's consider all and just pick the best unselected.
        # Actually, once selected, we can skip them. Let's track selected.
        
        # To speed up, we can maintain a list of active candidates.
        # For now, let's just compute scores for all.
        
        scores = np.zeros(N, dtype=np.int32)
        for j in range(N):
            if j in selected_indices:
                continue
            scores[j] = np.sum(~covered[covers_np[j]])

        # Pick the best
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            # Shouldn't happen if not covered.all(), but safety
            break

        selected.append(all_words[best_idx].tolist())
        selected_indices.append(best_idx)

        # Update covered
        for w in covers_np[best_idx]:
            covered[w] = True

        # Early exit if we've covered everything
        if covered.all():
            break

    # If we timed out, we need to ensure we return a valid cover.
    # The greedy might not have finished. Let's check.
    if not covered.all():
        # Fallback: add more codewords to cover the rest
        # Continue greedy but with a tighter deadline
        while not covered.all() and time.time() < deadline:
            scores = np.zeros(N, dtype=np.int32)
            for j in range(N):
                if j in selected_indices:
                    continue
                scores[j] = np.sum(~covered[covers_np[j]])
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            selected.append(all_words[best_idx].tolist())
            selected_indices.append(best_idx)
            for w in covers_np[best_idx]:
                covered[w] = True

    # Final safety: if still not covered, use the direct sum construction as fallback
    if not covered.all():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        fallback = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return fallback

    return selected
# EVOLVE-BLOCK-END
