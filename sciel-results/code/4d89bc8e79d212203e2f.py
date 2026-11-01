import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Precompute utilities for q=3, n=8
    # We assume the specific task q=3, n=8, r=1 based on the prompt context, 
    # but the function signature is general. 
    # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # So we must handle the specific case efficiently but generally if possible, 
    # though optimization is usually tailored.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: simple greedy
        return _general_greedy(q, n, radius, deadline - start_time)

    total_words = 3 ** n
    # Represent words as integers 0 to 6560
    # Precompute balls: for each word, list of words in its ball of radius 1
    # Ball size is 1 + 8 * 2 = 17
    
    # To speed up, we precompute the ball indices for all words.
    # This takes some memory (6561 * 17 ints) but is fast to access.
    balls = np.empty((total_words, 17), dtype=np.int32)
    
    # Convert int to base-3 digits helper
    def int_to_base3(x, n):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = x % 3
            x //= 3
        return digits

    def base3_to_int(digits):
        res = 0
        for d in digits:
            res = res * 3 + d
        return res

    # Precomputation
    for x in range(total_words):
        digits = int_to_base3(x, n)
        ball_indices = [x]
        for j in range(n):
            for v in range(3):
                if v != digits[j]:
                    new_digits = digits[:]
                    new_digits[j] = v
                    ball_indices.append(base3_to_int(new_digits))
        balls[x] = ball_indices

    # Greedy Cover
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    # Optimization: We can precompute for each word, which codewords cover it? 
    # No, that's the inverse. 
    # Let's just do the standard greedy loop.
    # To speed up the "find best candidate" step, we can maintain a count of uncovered words per candidate.
    # But updating it is O(ball_size) per step.
    # Total steps ~ 500. Total ops ~ 500 * 17 * 6561 (if we scan all candidates) -> 55M ops. 
    # This might be slow in Python. 
    # We can use numpy to vectorize the "count uncovered in ball" step for all candidates?
    # No, because 'covered' changes.
    
    # Alternative: 
    # 1. Start with all words as candidates.
    # 2. Pick the one with max uncovered.
    # 3. Mark covered.
    # 4. Remove candidates that are now fully covered? No, a candidate can still cover other uncovered words.
    
    # Let's try to optimize the inner loop.
    # We maintain `uncovered_count` for each candidate?
    # Initially, uncovered_count[i] = 17 for all i.
    # When we pick candidate `c`, we mark its ball as covered.
    # For every other candidate `i` that shares a word with `c`'s ball, we decrement its count.
    # This requires knowing which candidates share words with `c`.
    # This is a bipartite graph problem.
    
    # Simpler approach for 6561 words:
    # Just loop through all candidates and count uncovered. 
    # 6561 candidates * 17 words/candidate = 111,537 checks per step.
    # 500 steps * 111,537 = 55 million checks.
    # In Python, 55M simple boolean checks might take ~5-10 seconds. This is acceptable.
    
    remaining = np.ones(total_words, dtype=bool)
    
    while remaining.any():
        if time.time() > deadline - 10:
            break
            
        best_candidate = -1
        best_count = -1
        
        # Vectorized count of uncovered words in each ball
        # balls is shape (6561, 17)
        # remaining is shape (6561,)
        # We want sum(remaining[balls[i]], axis=1)
        # This can be done with numpy advanced indexing
        # counts = remaining[balls].sum(axis=1)
        
        counts = remaining[balls].sum(axis=1)
        best_candidate = np.argmax(counts)
        best_count = counts[best_candidate]
        
        if best_count == 0:
            break # Should not happen if remaining.any() is true and we have valid balls
            
        codewords.append(best_candidate)
        remaining[balls[best_candidate]] = False

    # Now we have a covering set `codewords` (list of ints)
    # Perform Redundancy Removal
    # We want to remove codewords that are not essential.
    # A codeword c is redundant if every word in B(c) is covered by other codewords in the set.
    
    # Convert list to set for fast lookup
    code_set = set(codewords)
    
    # To check redundancy efficiently:
    # For each word w in the space, count how many codewords cover it.
    # If count[w] >= 2 for all w in B(c), then c is redundant.
    
    # Let's maintain a coverage count array.
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in codewords:
        cover_count[balls[c]] += 1
        
    # Now try to remove each codeword
    # We iterate until no more can be removed
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        new_codewords = []
        for c in codewords:
            # Check if c is redundant
            # c is redundant if min(cover_count[balls[c]]) >= 2
            # Because if we remove c, the count for words in B(c) decreases by 1.
            # They must remain >= 1.
            if cover_count[balls[c]].min() >= 2:
                # Redundant
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_codewords.append(c)
        codewords = new_codewords

    # Convert back to lists of ints
    result = [int_to_base3(c, n) for c in codewords]
    
    # Verify validity (optional, for safety)
    # The evaluator checks this, but we can ensure we didn't mess up.
    # If we ran out of time during greedy, we might have an incomplete cover.
    # The prompt says "Valid when every word... is within... radius".
    # If we break early due to time, we MUST return a valid code.
    # The greedy loop `while remaining.any()` ensures validity if it completes.
    # If it breaks due to time, `remaining` is not empty.
    # We need a fallback.
    
    if remaining.any():
        # Fallback: Use a known valid construction or complete the cover greedily if possible.
        # Given the time limit, if we didn't finish, we are in trouble.
        # However, the greedy step is fast. 500 steps * 0.01s = 5s. 
        # The redundancy removal is also fast.
        # It is unlikely to timeout.
        # If it does, we can try to finish the cover quickly by picking arbitrary uncovered words?
        # No, that might increase size.
        # Let's assume it finishes. If not, we return what we have? 
        # No, invalid result scores 0.
        # We must ensure validity.
        
        # If we timed out, we can try to complete the cover using a simple heuristic:
        # Pick any uncovered word, add it as a codeword. Repeat.
        # This ensures validity but might be large.
        while remaining.any():
            if time.time() > deadline - 1:
                # Absolute last resort: return a large valid code
                # e.g. all words? No, too big.
                # Just return the current list? It's invalid.
                # We must finish.
                pass
            
            # Find an uncovered word
            idx = np.where(remaining)[0][0]
            codewords.append(idx)
            remaining[balls[idx]] = False
            # Note: This simple completion might not be optimal but is valid.
            
        # Re-run redundancy removal on the completed set?
        # Time is tight. Skip.

    return result

def _general_greedy(q, n, radius, time_limit):
    # Simple fallback for non-3,8,1 cases
    # Not optimized, but functional
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    def int_to_word(x):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = x % q
            x //= q
        return w

    def word_to_int(w):
        res = 0
        for d in w:
            res = res * q + d
        return res

    # Generate balls on the fly or precompute if small
    # For general case, precomputation might be heavy if q^n is large.
    # But task is specific to 3,8,1.
    # This is just a safety net.
    
    start = time.time()
    while covered.sum() < total:
        if time.time() - start > time_limit - 5:
            break
        best_c = -1
        best_count = -1
        for c in range(total):
            if covered[c]: continue # Optimization: if c is covered, it might still cover others, but usually we pick uncovered ones? No, pick max coverage.
            # Calculate coverage
            w = int_to_word(c)
            count = 0
            # Check self
            if not covered[c]: count += 1
            # Check neighbors
            for j in range(n):
                for v in range(q):
                    if v == w[j]: continue
                    w2 = w[:]
                    w2[j] = v
                    idx = word_to_int(w2)
                    if not covered[idx]: count += 1
            if count > best_count:
                best_count = count
                best_c = c
        
        if best_c == -1: break
        codewords.append(best_c)
        # Update covered
        w = int_to_word(best_c)
        covered[best_c] = True
        for j in range(n):
            for v in range(q):
                if v == w[j]: continue
                w2 = w[:]
                w2[j] = v
                covered[word_to_int(w2)] = True
        
    return [int_to_word(c) for c in codewords]
