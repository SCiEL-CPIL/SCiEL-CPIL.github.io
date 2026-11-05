"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque

# Deadline guard: kill after 150s wall-clock, return best found with margin
_DEADLINE = 145.0


def _hamming_distance(a, b):
    """Hamming distance between two tuples/lists of ints."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _all_neighbors(word, q, n, radius):
    """Yield all words within Hamming distance <= radius from word."""
    if radius == 0:
        yield word
        return
    # Precompute positions and values
    pos = range(n)
    if radius == 1:
        for i in pos:
            for v in range(q):
                if v == word[i]:
                    continue
                new_word = list(word)
                new_word[i] = v
                yield tuple(new_word)
    elif radius == 2:
        seen = set()
        seen.add(tuple(word))
        for i in pos:
            for v in range(q):
                if v == word[i]:
                    continue
                new_word = list(word)
                new_word[i] = v
                yield tuple(new_word)
                seen.add(tuple(new_word))
                for j in range(i + 1, n):
                    for w in range(q):
                        if w == word[j]:
                            continue
                        new_word2 = list(new_word)
                        new_word2[j] = w
                        tw = tuple(new_word2)
                        if tw not in seen:
                            yield tw
                            seen.add(tw)
    else:
        # General radius: generate all words within radius
        seen = set()
        seen.add(tuple(word))
        # BFS-like generation
        def gen_depth(cur, remaining):
            if remaining == 0:
                return
            cur_list = list(cur)
            for i in range(n):
                old = cur_list[i]
                for v in range(q):
                    if v == old:
                        continue
                    cur_list[i] = v
                    new_cur = tuple(cur_list)
                    if new_cur not in seen:
                        seen.add(new_cur)
                        yield new_cur
                        yield from gen_depth(new_cur, remaining - 1)
                cur_list[i] = old
        yield from gen_depth(tuple(word), radius)


def _greedy_cover(q, n, radius, seed=42, time_limit=140.0):
    """
    Greedy set-cover with dynamic conflict-aware scoring.
    Score = marginal_coverage - lambda * overlap_with_selected
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    total_words = q ** n
    # Represent words as integers for compactness
    # word -> int: sum(word[i] * q^(n-1-i))
    base = q
    powers = [base ** (n - 1 - i) for i in range(n)]
    
    def to_int(word):
        return sum(w * p for w, p in zip(word, powers))
    
    def to_word(idx):
        digits = []
        for i in range(n):
            p = powers[n - 1 - i]
            digits.append(idx // p)
            idx %= p
        return digits
    
    # Precompute all words as a set of ints
    all_words_set = set(range(total_words))
    
    # For each word, precompute its ball (words within radius)
    # This is expensive for large spaces, but for q=3, n=7, radius=1: 3^7=2187 words, ball size = 1 + 7*2 = 15
    # So total precomputation is 2187 * 15 = ~32805 entries, very manageable
    
    balls = {}
    for w in range(total_words):
        word = to_word(w)
        ball = set()
        ball.add(w)
        for i in range(n):
            for v in range(q):
                if v == word[i]:
                    continue
                new_word = list(word)
                new_word[i] = v
                ball.add(to_int(new_word))
        balls[w] = ball
    
    # Greedy: at each step, pick the word that maximizes (newly_covered - penalty * overlap)
    uncovered = set(all_words_set)
    selected = []
    selected_set = set()
    
    # For efficiency, maintain coverage count per word
    # But for 2187 words, we can just recompute
    
    while uncovered:
        if time.time() - start_time > time_limit:
            break
        
        best_word = None
        best_score = -1
        best_newly_covered = None
        
        # Candidate: any word not yet selected
        # For speed, we can sample or iterate all
        for w in all_words_set:
            if w in selected_set:
                continue
            newly_covered = balls[w] & uncovered
            newly_count = len(newly_covered)
            
            # Conflict penalty: how many of w's neighbors are already covered by selected
            # This encourages spreading out
            if selected_set:
                overlap = 0
                for s in selected_set:
                    if w in balls[s]:  # w is within radius of s
                        overlap += 1
                penalty = 0.1 * overlap
            else:
                penalty = 0
            
            score = newly_count - penalty
            
            if score > best_score:
                best_score = score
                best_word = w
                best_newly_covered = newly_covered
        
        if best_word is None:
            break
        
        selected.append(best_word)
        selected_set.add(best_word)
        uncovered -= best_newly_covered
    
    # Convert to list of lists
    result = [to_word(w) for w in selected]
    return result


def _local_search_improve(code, q, n, radius, seed=42, time_limit=140.0):
    """
    Try to improve the code by swapping out one codeword for another.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    base = q
    powers = [base ** (n - 1 - i) for i in range(n)]
    
    def to_int(word):
        return sum(w * p for w, p in zip(word, powers))
    
    def to_word(idx):
        digits = []
        for i in range(n):
            p = powers[n - 1 - i]
            digits.append(idx // p)
            idx %= p
        return digits
    
    total_words = q ** n
    
    # Precompute balls
    balls = {}
    for w in range(total_words):
        word = to_word(w)
        ball = set()
        ball.add(w)
        for i in range(n):
            for v in range(q):
                if v == word[i]:
                    continue
                new_word = list(word)
                new_word[i] = v
                ball.add(to_int(new_word))
        balls[w] = ball
    
    # Convert code to ints
    code_ints = [to_int(w) for w in code]
    
    def is_covering(code_ints):
        covered = set()
        for c in code_ints:
            covered |= balls[c]
        return len(covered) == total_words
    
    # Try removing one codeword: if still covering, keep it removed
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(len(code_ints)):
            if time.time() - start_time > time_limit:
                break
            trial = code_ints[:i] + code_ints[i+1:]
            if is_covering(trial):
                code_ints = trial
                improved = True
                break
    
    # Try swapping one codeword for another that allows further reduction
    for _ in range(50):  # limited iterations
        if time.time() - start_time > time_limit:
            break
        if len(code_ints) <= 1:
            break
        i = rng.randrange(len(code_ints))
        # Try replacing code_ints[i] with a random other word
        for attempt in range(20):
            j = rng.randrange(total_words)
            if j in code_ints:
                continue
            trial = code_ints[:i] + [j] + code_ints[i+1:]
            # Check if this swap allows us to remove another codeword
            for k in range(len(trial)):
                if k == i:
                    continue
                trial2 = trial[:k] + trial[k+1:]
                if is_covering(trial2):
                    code_ints = trial2
                    break
    
    # Final removal pass
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(len(code_ints)):
            if time.time() - start_time > time_limit:
                break
            trial = code_ints[:i] + code_ints[i+1:]
            if is_covering(trial):
                code_ints = trial
                improved = True
                break
    
    return [to_word(w) for w in code_ints]


def construct(q, n, radius):
    start_time = time.time()
    remaining = _DEADLINE - start_time
    
    # For the specific case q=3, n=7, radius=1, use specialized approach
    if q == 3 and n == 7 and radius == 1:
        # Try multiple greedy seeds and local search
        best_code = None
        best_len = float('inf')
        
        for seed in range(20):
            if time.time() - start_time > remaining - 5:
                break
            code = _greedy_cover(q, n, radius, seed=seed, time_limit=remaining - 10)
            code = _local_search_improve(code, q, n, radius, seed=seed, time_limit=remaining - 5)
            if len(code) < best_len:
                best_len = len(code)
                best_code = code
            if best_len <= 186:
                break
        
        if best_code is not None:
            return best_code
    
    # Fallback: direct sum construction
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        # Small case: brute force or simple greedy
        import itertools
        all_words = list(itertools.product(range(q), repeat=n))
        selected = []
        uncovered = set(all_words)
        while uncovered:
            best_word = None
            best_count = -1
            for w in all_words:
                if w in selected:
                    continue
                count = 0
                for u in uncovered:
                    if sum(1 for a, b in zip(w, u) if a != b) <= radius:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_word = w
            if best_word is None:
                break
            selected.append(best_word)
            uncovered = {u for u in uncovered if sum(1 for a, b in zip(best_word, u) if a != b) > radius}
        return [list(w) for w in selected]
# EVOLVE-BLOCK-END
