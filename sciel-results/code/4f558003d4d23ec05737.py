"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _precompute_balls(q, n, radius):
    """Precompute the set of words within Hamming distance radius of each word.
    Returns a list of length q^n where balls[i] is a set of indices within distance radius of word i."""
    total = q ** n
    balls = [None] * total
    for idx in range(total):
        # Convert index to word
        word = []
        x = idx
        for _ in range(n):
            word.append(x % q)
            x //= q
        word = tuple(word)
        ball = set()
        # Enumerate all words within distance radius
        for r in range(radius + 1):
            if r == 0:
                ball.add(idx)
            else:
                for pos in itertools.combinations(range(n), r):
                    for vals in itertools.product(range(q), repeat=r):
                        if 0 in vals:
                            continue  # skip if any value same as original? No, we need all combos
                        # Actually we need all combinations of values at those positions
                        # But we should skip the original word (r=0 case handled)
                        new_word = list(word)
                        for p, v in zip(pos, vals):
                            new_word[p] = v
                        # Check it's different from original
                        if tuple(new_word) == word:
                            continue
                        new_idx = 0
                        for i in range(n):
                            new_idx = new_idx * q + new_word[i]
                        ball.add(new_idx)
        balls[idx] = ball
    return balls


def _greedy_cover(q, n, radius, balls, rng=None):
    """Greedy set cover: repeatedly pick the word that covers the most uncovered words."""
    if rng is None:
        rng = random.Random()
    total = q ** n
    uncovered = set(range(total))
    code = []
    # Precompute coverage for each candidate
    # For efficiency, we'll track how many uncovered words each candidate covers
    # This is expensive to update, so we use a simpler approach:
    # At each step, scan all candidates and count uncovered coverage
    
    while uncovered:
        best_candidate = None
        best_count = -1
        for cand in range(total):
            if cand in uncovered:
                continue  # We can skip candidates that are already covered? No, we should consider all.
                # Actually, we should consider all candidates, but it's wasteful to check covered ones.
                # Let's just check all.
            count = len(balls[cand] & uncovered)
            if count > best_count:
                best_count = count
                best_candidate = cand
        if best_candidate is None or best_count <= 0:
            break
        code.append(best_candidate)
        uncovered -= balls[best_candidate]
    return code


def _local_search(q, n, radius, balls, code, rng, time_limit=10.0):
    """Local search: try to remove codewords, and try 1-swap moves."""
    start = time.time()
    code_set = set(code)
    
    # Convert to list of lists for easier manipulation
    code_list = list(code)
    
    # Try removing codewords
    changed = True
    while changed:
        changed = False
        for i in range(len(code_list)):
            if time.time() - start > time_limit:
                return code_list
            # Try removing code_list[i]
            test_code = code_list[:i] + code_list[i+1:]
            test_set = set(test_code)
            # Check if still covers everything
            covered = set()
            for c in test_set:
                covered |= balls[c]
            if len(covered) == q ** n:
                code_list = test_code
                changed = True
                break
    
    # Try 1-swap moves: replace one codeword with another
    for _ in range(100):  # Limited iterations
        if time.time() - start > time_limit:
            break
        i = rng.randrange(len(code_list))
        # Pick a random new codeword
        new_cand = rng.randrange(q ** n)
        if new_cand in code_set:
            continue
        test_code = code_list[:i] + [new_cand] + code_list[i+1:]
        test_set = set(test_code)
        covered = set()
        for c in test_set:
            covered |= balls[c]
        if len(covered) == q ** n:
            code_list = test_code
            changed = True
    
    return code_list


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    rng = random.Random(42)
    
    # Precompute balls
    balls = _precompute_balls(q, n, radius)
    total = q ** n
    
    # Greedy initialization
    code = _greedy_cover(q, n, radius, balls, rng)
    
    # Local search
    code = _local_search(q, n, radius, balls, code, rng, time_limit=30.0)
    
    # Multiple restarts with different random seeds
    best_code = code
    best_len = len(code)
    
    while time.time() < deadline - 5:
        # New greedy with different randomization
        code = _greedy_cover(q, n, radius, balls, rng)
        code = _local_search(q, n, radius, balls, code, rng, time_limit=10.0)
        
        if len(code) < best_len:
            best_code = code
            best_len = len(code)
        
        # Check if we've hit a good bound
        if best_len <= 186:
            break
    
    # Convert indices back to words
    result = []
    for idx in best_code:
        word = []
        x = idx
        for _ in range(n):
            word.append(x % q)
            x //= q
        result.append(word)
    
    return result
# EVOLVE-BLOCK-END
