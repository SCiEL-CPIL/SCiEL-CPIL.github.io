import time
import random
import itertools
import numpy as np

TOTAL_BUDGET = 60.0
SAFETY_MARGIN = 5.0
DEADLINE = TOTAL_BUDGET - SAFETY_MARGIN

def _hamming(a, b):
    return (a ^ b).bit_count()

def _check_min_distance(words, min_dist):
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if _hamming(wi, words[j]) < min_dist:
                return False
    return True

def _generate_all_vectors_up_to_weight(N, max_weight):
    """Generate all N-bit vectors with weight <= max_weight."""
    vectors = []
    for w in range(max_weight + 1):
        for positions in itertools.combinations(range(N), w):
            vec = 0
            for p in positions:
                vec |= (1 << p)
            vectors.append(vec)
    return vectors

def _expand_forbidden_set(word, N, min_dist):
    """
    For a given word, compute a bitmask representing all vectors within
    Hamming distance < min_dist from it. We represent this as a set of
    forbidden vectors for quick lookup.
    
    For efficiency, we only track the forbidden set implicitly by checking
    against existing words directly, but we use a different strategy:
    instead of storing all forbidden vectors, we store the existing words
    and check candidates against them.
    
    Actually, for N=18 and min_dist=6, the ball of radius 5 around a word
    has sum(C(18,k) for k=0..5) = 1+18+153+816+3060+8568 = 12616 vectors.
    This is manageable. But we'll use a different approach: for each candidate,
    we check against all existing words. To speed this up, we'll use
    numpy vectorized operations.
    """
    pass

def _greedy_pack_with_forbidden_set(base_words, N, min_dist, time_start, time_limit):
    """
    Greedy packing: start with base_words, then try to add random high-weight
    vectors that are at distance >= min_dist from all existing words.
    
    Key insight: vectors of weight 4-14 are the "interesting" middle range.
    We'll try random candidates from this range and check against all existing.
    """
    rng = random.Random(12345)
    best_words = list(base_words)
    
    # Precompute weights for quick filtering
    # For N=18, we can precompute bit_count for all 2^18 values
    # But that's 262144 entries, which is fine
    
    # Strategy: try random candidates, check against all existing words
    # To speed up, we can use numpy for batch checking
    
    words_array = np.array(best_words, dtype=np.uint32)
    
    def try_add_candidate(cand):
        """Check if cand is at distance >= min_dist from all existing words."""
        # Vectorized check
        diffs = words_array ^ cand
        # Count bits in each
        # For uint32, we can use a lookup or loop
        # Since we have at most a few thousand words, let's do it efficiently
        for w in best_words:
            if _hamming(cand, w) < min_dist:
                return False
        return True
    
    start = time.time()
    attempts = 0
    while time.time() - start < time_limit:
        # Generate random candidate with weight in [4, 14]
        weight = rng.randint(4, 14)
        positions = rng.sample(range(N), weight)
        cand = 0
        for p in positions:
            cand |= (1 << p)
        
        if try_add_candidate(cand):
            best_words.append(cand)
        
        attempts += 1
        # Periodically check time
        if attempts % 100 == 0 and time.time() - start >= time_limit:
            break
    
    return best_words

def _local_search_improvement(words, N, min_dist, time_start, time_limit):
    """
    Try to improve the code by:
    1. Removing a word and trying to add 2 words in its place
    2. Swapping words for better packing
    
    This is a simple 1-opt / 2-opt local search.
    """
    rng = random.Random(67890)
    best = list(words)
    start = time.time()
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        n = len(best)
        for i in range(n):
            if time.time() - start >= time_limit:
                break
            
            # Try removing word i and see if we can add 2 new words
            temp_words = best[:i] + best[i+1:]
            if not temp_words:
                continue
            
            # Find words that are "close" to word i (within distance 5)
            # These are candidates that might benefit from the removal
            word_i = best[i]
            candidates = []
            for w in temp_words:
                d = _hamming(word_i, w)
                if d < min_dist:
                    candidates.append(w)
            
            # Try to add 2 random words that fit with temp_words
            added = 0
            for _ in range(50):
                if time.time() - start >= time_limit:
                    break
                weight = rng.randint(3, 15)
                positions = rng.sample(range(N), weight)
                cand = 0
                for p in positions:
                    cand |= (1 << p)
                
                ok = True
                for w in temp_words:
                    if _hamming(cand, w) < min_dist:
                        ok = False
                        break
                if ok and cand not in temp_words:
                    temp_words.append(cand)
                    added += 1
                    if added == 2:
                        break
            
            if added >= 1:
                best = temp_words
                improved = True
    
    return best

def _reed_muller_extended():
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _get_golay_generator_matrix():
    G = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    return G

def _generate_golay_code(G):
    codewords = []
    n_rows = len(G)
    n_cols = len(G[0])
    for mask in range(1 << n_rows):
        word = 0
        for i in range(n_rows):
            if (mask >> i) & 1:
                for j in range(n_cols):
                    if G[i][j]:
                        word ^= (1 << j)
        codewords.append(word)
    return codewords

def _find_shortening_set(G, codewords, target_size=512):
    n_cols = 24
    n_rows = 12
    best_set = None
    best_count = 0

    for cols in itertools.combinations(range(n_cols), 6):
        sub = [[G[r][c] for c in cols] for r in range(n_rows)]

        matrix = [row[:] for row in sub]
        rank = 0
        for c in range(6):
            pivot = -1
            for r in range(rank, 12):
                if matrix[r][c] == 1:
                    pivot = r
                    break
            if pivot == -1:
                continue
            matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
            for r in range(12):
                if r != rank and matrix[r][c] == 1:
                    for cc in range(c, 6):
                        matrix[r][cc] ^= matrix[rank][cc]
            rank += 1
            if rank == 6:
                break

        if rank <= 3:
            count = 0
            for w in codewords:
                ok = True
                for c in cols:
                    if (w >> c) & 1:
                        ok = False
                        break
                if ok:
                    count += 1
            if count > best_count:
                best_count = count
                best_set = cols

    return best_set, best_count

def _shorten_code(codewords, cols_to_zero):
    remaining_cols = [c for c in range(24) if c not in cols_to_zero]
    shortened = []
    for w in codewords:
        if all(((w >> c) & 1) == 0 for c in cols_to_zero):
            new_word = 0
            for i, c in enumerate(remaining_cols):
                if (w >> c) & 1:
                    new_word |= (1 << i)
            shortened.append(new_word)
    return shortened

def construct():
    N = 18
    MIN_DIST = 6
    start = time.time()

    # Strategy 1: Start with all low-weight vectors (weight 0-3) as base
    # These are guaranteed to be at distance >= 4 from each other
    # Actually, two weight-3 vectors can be at distance as low as 0 if identical,
    # or 2 if they share 1 position, or 4 if they share 0 positions... wait
    # distance between two distinct weight-3 vectors:
    # - If they share k positions (k=0,1,2), distance = 2*(3-k) = 6, 4, or 2
    # So weight-3 vectors are NOT all at distance >= 6 from each other!
    # 
    # Let's reconsider: weight 0-2 vectors:
    # - Two weight-2 vectors: distance = 2*(2-k) where k is overlap
    #   k=0: dist=4, k=1: dist=2, k=2: dist=0
    # So even weight-2 vectors aren't all at distance >= 6.
    #
    # Actually, the only vectors guaranteed to be at distance >= 6 from each other
    # when all have weight <= w are... let's think.
    #
    # For minimum distance 6, we need any two codewords to differ in at least 6 positions.
    # If we take all vectors of weight 0: just {0}
    # If we take weight 0 and 1: 0 and e_i have distance 1 < 6. No good.
    #
    # So the "low weight base" idea doesn't work directly for min_dist=6.
    #
    # Let's use the Reed-Muller code as a starting point instead,
    # and then use structured greedy packing.

    # Start with Reed-Muller extended code (32 words, length 16, padded to 18)
    rm_words = _reed_muller_extended()
    # Pad to length 18 by adding 2 zero bits
    rm_words_18 = [w << 2 for w in rm_words]
    
    # Verify RM code has min distance 6
    if not _check_min_distance(rm_words_18, MIN_DIST):
        # Fallback to Golay shortening
        G = _get_golay_generator_matrix()
        golay_words = _generate_golay_code(G)
        shortening_set, count = _find_shortening_set(G, golay_words, target_size=512)
        if shortening_set is not None and count >= 512:
            code = _shorten_code(golay_words, shortening_set)
            if len(code) >= 512 and _check_min_distance(code, MIN_DIST):
                return code
        raise ValueError("RM code failed distance check")
    
    best = list(rm_words_18)
    
    elapsed = time.time() - start
    time_remaining = DEADLINE - elapsed
    
    # Strategy 2: Structured greedy packing
    # Try to add vectors of various weights that maintain min distance 6
    if time_remaining > 10.0:
        greedy_result = _greedy_pack_with_forbidden_set(
            best, N, MIN_DIST, start, time_remaining - 5.0
        )
        if len(greedy_result) > len(best) and _check_min_distance(greedy_result, MIN_DIST):
            best = greedy_result
    
    elapsed = time.time() - start
    time_remaining = DEADLINE - elapsed
    
    # Strategy 3: Local search improvement
    if time_remaining > 5.0:
        local_result = _local_search_improvement(
            best, N, MIN_DIST, start, time_remaining - 2.0
        )
        if len(local_result) > len(best) and _check_min_distance(local_result, MIN_DIST):
            best = local_result
    
    # Strategy 4: Try Golay shortening as alternative
    elapsed = time.time() - start
    time_remaining = DEADLINE - elapsed
    if time_remaining > 15.0:
        G = _get_golay_generator_matrix()
        golay_words = _generate_golay_code(G)
        shortening_set, count = _find_shortening_set(G, golay_words, target_size=512)
        if shortening_set is not None and count >= 512:
            code = _shorten_code(golay_words, shortening_set)
            if len(code) > len(best) and _check_min_distance(code, MIN_DIST):
                best = code
    
    return best
