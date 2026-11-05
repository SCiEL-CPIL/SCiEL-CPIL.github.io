import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # For q=3, n=7, radius=1: each codeword covers 1 + 7*2 = 15 words
    # Total words: 3^7 = 2187
    # Lower bound: ceil(2187/15) = 146, best known: 186
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: direct product construction
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    total_words = 3 ** 7  # 2187
    
    # Precompute all words as tuples
    all_words = list(itertools.product(range(3), repeat=7))
    word_index = {w: i for i, w in enumerate(all_words)}
    
    # For each word, precompute all words within Hamming distance 1 (including itself)
    # This is the "ball" of radius 1
    def ball_1(w):
        """Generate all words within Hamming distance 1 of w."""
        yield w
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    yield tuple(new_w)
    
    # Precompute balls for all words
    balls = [list(ball_1(w)) for w in all_words]
    ball_indices = [[word_index[b] for b in ball] for ball in balls]
    
    # Greedy construction with incremental coverage
    covered = [False] * total_words
    codewords = []
    
    def add_codeword(word_idx):
        for b_idx in ball_indices[word_idx]:
            if not covered[b_idx]:
                covered[b_idx] = True
        codewords.append(word_idx)
    
    # Greedy: repeatedly pick the word covering most uncovered words
    # Use a priority approach: track coverage count for each potential codeword
    # For speed, we use a simpler greedy: pick word that covers most uncovered
    
    # Track how many uncovered words each position covers
    # Actually, let's use a different approach: maintain for each word, how many of its ball are uncovered
    
    # For efficiency, let's track: for each word w, uncovered_count[w] = number of uncovered words in ball(w)
    uncovered_count = [0] * total_words
    for i in range(total_words):
        uncovered_count[i] = len(ball_indices[i])
    
    # Greedy selection
    remaining = set(range(total_words))
    
    for step in range(200):  # at most 200 codewords
        if time.time() > deadline:
            break
        if not remaining:
            break
        
        # Find the word with maximum uncovered coverage
        best_word = None
        best_count = -1
        for w in remaining:
            if uncovered_count[w] > best_count:
                best_count = uncovered_count[w]
                best_word = w
                if best_count == 0:
                    break
        
        if best_word is None or best_count == 0:
            break
        
        # Add this codeword
        add_codeword(best_word)
        remaining.discard(best_word)
        
        # Update uncovered counts for all words
        for b_idx in ball_indices[best_word]:
            if covered[b_idx]:
                continue
            # b_idx was just covered; decrement uncovered_count for all words whose ball contains b_idx
            # We need reverse mapping: for each word, which potential codewords cover it
            # Actually, we need: for each word x, if b_idx is in ball(x), decrement uncovered_count[x]
            # This is expensive. Let's precompute reverse balls.
        
        # Better approach: precompute reverse coverage
        # reverse_ball[i] = set of words w such that i is in ball(w)
        # But this is symmetric! ball(w) contains i iff ball(i) contains w (Hamming distance is symmetric)
        # So reverse_ball[i] = ball(i) in terms of indices
    
        # Actually since Hamming distance is symmetric, ball(w) contains i iff ball(i) contains w
        # So to update: for each newly covered word b_idx, decrement uncovered_count[w] for all w in ball(b_idx)
        for b_idx in ball_indices[best_word]:
            if not covered[b_idx]:
                continue
            # b_idx is now covered
            for w in ball_indices[b_idx]:
                if w in remaining or w == best_word:
                    uncovered_count[w] -= 1
    
    # Check if we have a valid covering
    if all(covered):
        result = [list(all_words[w]) for w in codewords]
        # Try local search to improve
        return local_search(result, all_words, word_index, ball_indices, covered, deadline)
    
    # If greedy didn't cover everything, try to fix
    # Find uncovered words and add codewords for them
    for i in range(total_words):
        if not covered[i]:
            add_codeword(i)
            # Update coverage
            for b_idx in ball_indices[i]:
                covered[b_idx] = True
                for w in ball_indices[b_idx]:
                    uncovered_count[w] = max(0, uncovered_count[w] - 1)
    
    result = [list(all_words[w]) for w in codewords]
    return local_search(result, all_words, word_index, ball_indices, covered, deadline)


def local_search(codewords, all_words, word_index, ball_indices, covered, deadline):
    """Local search to minimize codeword count."""
    start_time = time.time()
    
    # Convert to set for fast lookup
    cw_set = set(tuple(c) for c in codewords)
    cw_indices = set(word_index[tuple(c)] for c in codewords)
    
    def is_valid(cw_indices_set):
        """Check if the set of codeword indices covers all words."""
        for i in range(len(all_words)):
            if i not in cw_indices_set:
                # Check if any codeword is within distance 1 of i
                found = False
                for cw in cw_indices_set:
                    # Hamming distance check
                    w1 = all_words[cw]
                    w2 = all_words[i]
                    dist = sum(1 for a, b in zip(w1, w2) if a != b)
                    if dist <= 1:
                        found = True
                        break
                if not found:
                    return False
            else:
                # i is a codeword, so it's covered
                pass
        return True
    
    # Faster validity check using precomputed balls
    def check_coverage(cw_indices_set):
        """Check coverage using precomputed ball indices."""
        # A word i is covered if there exists cw in cw_indices_set such that i is in ball(cw)
        # Equivalently, cw is in ball(i)
        for i in range(len(all_words)):
            # Check if any codeword is in ball(i)
            for b in ball_indices[i]:
                if b in cw_indices_set:
                    break
            else:
                return False
        return True
    
    # Try to remove codewords
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each codeword
        for cw_idx in list(cw_indices):
            if time.time() > deadline:
                break
            test_set = cw_indices - {cw_idx}
            if check_coverage(test_set):
                cw_indices = test_set
                improved = True
    
    # Try swapping: remove one, add another
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for cw_idx in list(cw_indices):
                if time.time() > deadline:
                    break
                # Try removing cw_idx and adding a different word
                for new_idx in range(len(all_words)):
                    if new_idx in cw_indices:
                        continue
                    test_set = (cw_indices - {cw_idx}) | {new_idx}
                    if check_coverage(test_set):
                        # This swap doesn't reduce count, but might enable future removals
                        # Only do it if it helps (we'd need to check if we can then remove more)
                        # For now, skip swaps that don't reduce count
                        pass
    
    # Convert back to list of codewords
    result = [list(all_words[idx]) for idx in sorted(cw_indices)]
    return result


# Actually, the above local search is too slow. Let me rewrite with a better approach.

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    total_words = 2187
    all_words = list(itertools.product(range(3), repeat=7))
    word_index = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls (words within Hamming distance 1)
    balls = []
    for w in all_words:
        ball = [w]
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.append(tuple(new_w))
        balls.append(tuple(ball))
    
    # Convert to index sets for faster lookup
    ball_idx = [set(word_index[b] for b in ball) for ball in balls]
    
    # Greedy construction
    covered_count = [0] * total_words  # How many of each word's ball are covered
    is_covered = [False] * total_words
    codewords = set()
    
    # Initialize: no words covered
    # For each potential codeword w, coverage[w] = number of uncovered words in ball(w)
    # Initially all uncovered, so coverage[w] = 15 for all w
    
    # Greedy: pick word covering most uncovered
    # Use a simple approach: maintain uncovered set, pick word with max intersection
    
    uncovered = set(range(total_words))
    
    while uncovered and time.time() < deadline:
        # Find best word
        best_word = None
        best_count = -1
        for w in range(total_words):
            if w in codewords:
                continue
            # Count how many uncovered words are in ball(w)
            count = 0
            for b in ball_idx[w]:
                if b in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = w
        
        if best_word is None or best_count == 0:
            break
        
        codewords.add(best_word)
        # Remove covered words from uncovered
        for b in ball_idx[best_word]:
            uncovered.discard(b)
    
    # Check coverage
    if uncovered:
        # Add codewords for remaining uncovered words
        for w in list(uncovered):
            codewords.add(w)
            for b in ball_idx[w]:
                uncovered.discard(b)
    
    result = [list(all_words[w]) for w in sorted(codewords)]
    
    # Local search: try to remove codewords
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for cw in list(codewords):
                if time.time() > deadline:
                    break
                test = codewords - {cw}
                # Check if test covers everything
                ok = True
                for i in range(total_words):
                    # Is i covered by some word in test?
                    found = False
                    for b in ball_idx[i]:
                        if b in test:
                            found = True
                            break
                    if not found:
                        ok = False
                        break
                if ok:
                    codewords = test
                    improved = True
    
    result = [list(all_words[w]) for w in sorted(codewords)]
    return result
