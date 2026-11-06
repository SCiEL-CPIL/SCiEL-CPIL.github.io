import time
import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    def time_remaining():
        return time_limit - (time.time() - start_time)

    def to_bit_vector(x):
        return np.array([(x >> i) & 1 for i in range(N)], dtype=np.uint8)

    def from_bit_vector(vec):
        x = 0
        for i in range(N):
            if vec[i]:
                x |= (1 << i)
        return x

    def hamming_dist_vec(a, b):
        return np.sum(a ^ b)

    # Start with the 32-word RM(1,4) seed extended to 17 bits
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)

    word_vecs = np.array([to_bit_vector(w) for w in words], dtype=np.uint8)

    def is_valid(candidate_vec, current_vecs):
        """Check if candidate_vec has min distance from all current_vecs."""
        if len(current_vecs) == 0:
            return True
        dists = np.sum(current_vecs ^ candidate_vec, axis=1)
        return np.all(dists >= MIN_DIST)

    def greedy_add(words_list, word_vecs_arr):
        """Try to add random vectors until we can't."""
        rng = np.random.default_rng()
        added = 0
        while time_remaining() > 2.0:
            # Try a batch of random candidates
            batch_size = 5000
            candidates = rng.integers(0, 2**N, size=batch_size)
            cand_vecs = np.array([to_bit_vector(c) for c in candidates], dtype=np.uint8)

            found_any = False
            for i in range(batch_size):
                if time_remaining() < 1.0:
                    break
                cv = cand_vecs[i]
                if is_valid(cv, word_vecs_arr):
                    words_list.append(from_bit_vector(cv))
                    word_vecs_arr = np.vstack([word_vecs_arr, cv])
                    added += 1
                    found_any = True
            if not found_any:
                break
        return added

    def local_search(words_list, word_vecs_arr):
        """Try to improve by swapping out poorly-placed words."""
        rng = np.random.default_rng()
        n = len(words_list)
        if n == 0:
            return 0

        # Compute current minimum distances for each word
        # For efficiency, we'll do targeted swaps

        improvements = 0
        max_iters = 200
        for _ in range(max_iters):
            if time_remaining() < 2.0:
                break

            # Pick a random word to potentially remove
            idx = rng.integers(0, n)

            # Try to find a replacement that allows adding more words
            # Remove word idx temporarily
            temp_vecs = np.delete(word_vecs_arr, idx, axis=0)
            temp_words = words_list[:idx] + words_list[idx+1:]

            # Try to add several random words
            added_count = 0
            new_vecs = temp_vecs.copy()
            new_words = list(temp_words)
            batch_size = 2000
            candidates = rng.integers(0, 2**N, size=batch_size)

            for i in range(batch_size):
                if time_remaining() < 1.5:
                    break
                cv = np.array([(candidates[i] >> j) & 1 for j in range(N)], dtype=np.uint8)
                if is_valid(cv, new_vecs):
                    new_words.append(from_bit_vector(cv))
                    new_vecs = np.vstack([new_vecs, cv])
                    added_count += 1

            # If we added at least 1 more word, accept the swap
            if added_count >= 1:
                words_list = new_words
                word_vecs_arr = new_vecs
                n = len(words_list)
                improvements += 1
            # Otherwise, keep original

        return improvements

    # Phase 1: Greedy addition
    greedy_add(words, word_vecs)
    word_vecs = np.array([to_bit_vector(w) for w in words], dtype=np.uint8)

    # Phase 2: Local search with swaps
    local_search(words, word_vecs)
    word_vecs = np.array([to_bit_vector(w) for w in words], dtype=np.uint8)

    # Phase 3: More greedy addition after local search
    greedy_add(words, word_vecs)
    word_vecs = np.array([to_bit_vector(w) for w in words], dtype=np.uint8)

    # Phase 4: Another round of local search
    local_search(words, word_vecs)
    word_vecs = np.array([to_bit_vector(w) for w in words], dtype=np.uint8)

    # Phase 5: Final greedy pass
    greedy_add(words, word_vecs)

    # Remove duplicates just in case
    unique_words = list(set(words))

    return unique_words
