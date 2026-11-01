"""Improved program using Iterated Local Search with repair for covering code reduction."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _encode(word, q, n):
    """Encode a word as an integer for fast distance computation."""
    code = 0
    for i in range(n):
        code = code * q + word[i]
    return code


def _decode(code, q, n):
    """Decode an integer back to a word."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = code % q
        code //= q
    return word


def _hamming_dist_enc(a, b, q, n):
    """Compute Hamming distance between two encoded words."""
    d = 0
    for i in range(n):
        if a % q != b % q:
            d += 1
            if d > 1:
                return d
        a //= q
        b //= q
    return d


def _ball_radius1(word, q, n):
    """Generate the word itself and all words at distance 1."""
    yield word
    for pos in range(n):
        orig = word[pos]
        for new_val in range(q):
            if new_val == orig:
                continue
            w = list(word)
            w[pos] = new_val
            yield w


def _is_redundant_fast(cw_enc, all_enc, n, q, radius):
    """Check if removing codeword still leaves all its ball covered by others."""
    cw = _decode(cw_enc, q, n)
    for j, other_enc in enumerate(all_enc):
        if other_enc == cw_enc:
            continue
        if _hamming_dist_enc(cw_enc, other_enc, q, n) <= radius:
            break
    else:
        return False
    
    for pos in range(n):
        orig = cw[pos]
        for new_val in range(q):
            if new_val == orig:
                continue
            neighbor = list(cw)
            neighbor[pos] = new_val
            nb_enc = _encode(neighbor, q, n)
            covered = False
            for j, other_enc in enumerate(all_enc):
                if other_enc == cw_enc:
                    continue
                if _hamming_dist_enc(nb_enc, other_enc, q, n) <= radius:
                    covered = True
                    break
            if not covered:
                return False
    return True


def _greedy_reduce(codewords, q, n, radius, time_limit):
    """Remove redundant codewords using greedy reduction with scoring."""
    start_time = time.time()
    codewords = [list(cw) for cw in codewords]
    m = len(codewords)
    
    while True:
        if time.time() - start_time > time_limit:
            break
        if m == 0:
            break
        
        m = len(codewords)
        all_enc = [_encode(cw, q, n) for cw in codewords]
        
        # Compute redundancy score: number of uniquely covered words in ball
        scores = [0] * m
        
        for i, cw in enumerate(codewords):
            if time.time() - start_time > time_limit:
                return codewords
            
            unique_count = 0
            cw_enc = all_enc[i]
            
            # Check cw itself
            covered_by_other = False
            for j, other_enc in enumerate(all_enc):
                if j == i:
                    continue
                if _hamming_dist_enc(cw_enc, other_enc, q, n) <= radius:
                    covered_by_other = True
                    break
            if not covered_by_other:
                unique_count += 1
            
            # Check neighbors
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    neighbor = list(cw)
                    neighbor[pos] = new_val
                    nb_enc = _encode(neighbor, q, n)
                    covered_by_other = False
                    for j, other_enc in enumerate(all_enc):
                        if j == i:
                            continue
                        if _hamming_dist_enc(nb_enc, other_enc, q, n) <= radius:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        unique_count += 1
            
            scores[i] = unique_count
        
        min_score = min(scores)
        if min_score > 0:
            break
        
        removable = [i for i in range(m) if scores[i] == 0]
        if removable:
            # Remove the one with highest index (heuristic for stability)
            best_idx = removable[-1]
            codewords.pop(best_idx)
    
    return codewords


def _greedy_cover(q, n, radius, time_limit, seed=42):
    """Build a covering code using greedy set cover."""
    rng = random.Random(seed)
    start_time = time.time()
    
    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls
    balls = []
    for w in all_words:
        ball = set()
        for x in _ball_radius1(w, q, n):
            ball.add(word_to_idx[x])
        balls.append(ball)
    
    # Greedy set cover
    uncovered = set(range(total))
    codewords = []
    
    while uncovered:
        if time.time() - start_time > time_limit:
            break
        
        best_idx = -1
        best_gain = -1
        for i in range(total):
            gain = len(balls[i] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            elif gain == best_gain and gain > 0:
                # Tie-break: prefer words that cover more already-covered words (overlap)
                pass
        
        if best_gain <= 0:
            break
        
        codewords.append(list(all_words[best_idx]))
        uncovered -= balls[best_idx]
    
    return codewords


def _repair(codewords, q, n, radius, time_limit):
    """Add codewords to ensure full coverage."""
    start_time = time.time()
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    # Find uncovered words
    covered = [False] * total
    for cw in codewords:
        for pos in range(n):
            orig = cw[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                neighbor = list(cw)
                neighbor[pos] = new_val
                idx = 0
                for i in range(n):
                    idx = idx * q + neighbor[i]
                covered[idx] = True
        idx = 0
        for i in range(n):
            idx = idx * q + cw[i]
        covered[idx] = True
    
    # Add codewords to cover uncovered words
    while True:
        if time.time() - start_time > time_limit:
            break
        
        uncovered = [i for i in range(total) if not covered[i]]
        if not uncovered:
            break
        
        # Pick an uncovered word and find best codeword to add
        target = uncovered[0]
        target_word = all_words[target]
        
        # Find the word that covers the most uncovered words
        best_word = None
        best_gain = -1
        for i in range(total):
            gain = 0
            for pos in range(n):
                orig = all_words[i][pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    neighbor = list(all_words[i])
                    neighbor[pos] = new_val
                    nb_idx = 0
                    for k in range(n):
                        nb_idx = nb_idx * q + neighbor[k]
                    if not covered[nb_idx]:
                        gain += 1
            cw_idx = 0
            for k in range(n):
                cw_idx = cw_idx * q + all_words[i][k]
            if not covered[cw_idx]:
                gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_word = all_words[i]
        
        if best_word is None or best_gain <= 0:
            break
        
        codewords.append(list(best_word))
        # Update covered
        for pos in range(n):
            orig = best_word[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                neighbor = list(best_word)
                neighbor[pos] = new_val
                nb_idx = 0
                for k in range(n):
                    nb_idx = nb_idx * q + neighbor[k]
                covered[nb_idx] = True
        cw_idx = 0
        for k in range(n):
            cw_idx = cw_idx * q + best_word[k]
        covered[cw_idx] = True
    
    return codewords


def construct(q, n, radius):
    deadline = time.time() + 110.0
    
    # Start with tetracode product construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Initial greedy reduction
    remaining = deadline - time.time()
    if remaining > 10.0:
        codewords = _greedy_reduce(codewords, q, n, radius, remaining - 10.0)
    
    best_codewords = codewords
    best_len = len(codewords)
    
    # Iterated Local Search
    rng = random.Random(12345)
    while time.time() < deadline - 15.0:
        # Perturb: remove k random codewords
        k = max(1, len(codewords) // 10)
        k = min(k, len(codewords) - best_len // 2)  # Don't remove too many
        if k <= 0:
            k = 1
        
        perturbed = list(codewords)
        for _ in range(k):
            if perturbed:
                idx = rng.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
        
        # Repair
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        perturbed = _repair(perturbed, q, n, radius, remaining - 10.0)
        
        # Reduce
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        perturbed = _greedy_reduce(perturbed, q, n, radius, remaining - 5.0)
        
        # Accept if better
        if len(perturbed) < best_len:
            best_len = len(perturbed)
            best_codewords = perturbed
            codewords = perturbed
        else:
            # Keep current for next perturbation (or revert)
            pass
    
    return best_codewords
# EVOLVE-BLOCK-END
