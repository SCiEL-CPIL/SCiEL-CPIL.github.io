import time
import itertools
import random
import heapq
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds limit to be safe
    
    # Helper to convert word to int
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val
        
    # Helper to convert int to word
    def int_to_word(w_int):
        w = []
        tmp = w_int
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        return list(reversed(w))

    num_words = q ** n
    all_words_ints = list(range(num_words))
    
    # Precompute balls for all possible words
    # ball[word_int] = frozenset of word_ints within distance radius
    # For radius=1, ball size is 1 + n*(q-1)
    ball_cache = {}
    for w_int in all_words_ints:
        w = []
        tmp = w_int
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        w = tuple(reversed(w))
        
        ball = {w_int}
        if radius >= 1:
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        new_w = list(w)
                        new_w[i] = val
                        ball.add(word_to_int(tuple(new_w)))
        # Note: For radius > 1, this is incomplete. 
        # But for K_3(6,1), radius is 1.
        # If radius > 1 is passed, we need BFS. 
        # Given the task context, we assume radius=1. 
        # To be safe, if radius > 1, we fallback to a simple construction.
        if radius > 1:
             # Simple BFS for general radius
             visited = {w_int}
             queue = [(w_int, 0)]
             while queue:
                 cur, dist = queue.pop(0)
                 if dist == radius:
                     continue
                 cur_w = int_to_word(cur)
                 for i in range(n):
                     for val in range(q):
                         if val != cur_w[i]:
                             new_w = cur_w[:]
                             new_w[i] = val
                             new_int = word_to_int(tuple(new_w))
                             if new_int not in visited:
                                 visited.add(new_int)
                                 queue.append((new_int, dist + 1))
             ball = visited
        ball_cache[w_int] = frozenset(ball)

    # 1. Generate seed codewords
    # Direct sum of tetracode (length 4) and all words of length n-4
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        seed_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        # Fallback: random or all words
        seed_codewords = [int_to_word(i) for i in range(min(num_words, 100))]
        
    codeword_ints = set(word_to_int(tuple(c)) for c in seed_codewords)
    
    # Maintain cover counts
    cover_count = [0] * num_words
    
    # Initialize cover counts
    for c_int in codeword_ints:
        for w_int in ball_cache[c_int]:
            cover_count[w_int] += 1
            
    # Build inverted index: for each word, which codewords cover it
    word_to_codewords = [[] for _ in range(num_words)]
    for c_int in codeword_ints:
        for w_int in ball_cache[c_int]:
            word_to_codewords[w_int].append(c_int)

    def compute_redundancy(c_int):
        """Compute redundancy score for a codeword. Higher = more redundant = better to remove."""
        ball = ball_cache[c_int]
        redundancy = 0
        for w_int in ball:
            if cover_count[w_int] > 1:
                redundancy += cover_count[w_int] - 1
        return redundancy

    # Greedy removal with priority
    def greedy_remove(codeword_ints, cover_count, word_to_codewords):
        counter = 0
        pq = []
        for c_int in codeword_ints:
            ball = ball_cache[c_int]
            min_count = min(cover_count[w] for w in ball)
            if min_count > 1:
                redundancy = compute_redundancy(c_int)
                heapq.heappush(pq, (-redundancy, counter, c_int))
                counter += 1
        
        while pq and time.time() < deadline - 5:
            neg_red, _, c_int = heapq.heappop(pq)
            
            if c_int not in codeword_ints:
                continue
            
            ball = ball_cache[c_int]
            removable = True
            for w_int in ball:
                if cover_count[w_int] <= 1:
                    removable = False
                    break
            
            if not removable:
                continue
            
            # Remove this codeword
            codeword_ints.remove(c_int)
            
            for w_int in ball:
                cover_count[w_int] -= 1
                if c_int in word_to_codewords[w_int]:
                    word_to_codewords[w_int].remove(c_int)
            
            affected_codewords = set()
            for w_int in ball:
                for c_prime in word_to_codewords[w_int]:
                    if c_prime != c_int:
                        affected_codewords.add(c_prime)
            
            for c_prime in affected_codewords:
                if c_prime not in codeword_ints:
                    continue
                ball_prime = ball_cache[c_prime]
                min_count_prime = min(cover_count[w] for w in ball_prime)
                if min_count_prime > 1:
                    redundancy_prime = compute_redundancy(c_prime)
                    heapq.heappush(pq, (-redundancy_prime, counter, c_prime))
                    counter += 1
        return codeword_ints, cover_count, word_to_codewords

    best_codewords = None
    best_size = float('inf')
    
    # Multi-start with Randomized Greedy Removal and Local Search (Swaps)
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Reset state
        codeword_ints = set(word_to_int(tuple(c)) for c in seed_codewords)
        cover_count = [0] * num_words
        word_to_codewords = [[] for _ in range(num_words)]
        for c_int in codeword_ints:
            for w_int in ball_cache[c_int]:
                cover_count[w_int] += 1
                word_to_codewords[w_int].append(c_int)
                
        # Phase 1: Greedy Removal
        codeword_ints, cover_count, word_to_codewords = greedy_remove(codeword_ints, cover_count, word_to_codewords)
        
        # Phase 2: Local Search (Swap) to break local minima
        # Try to swap a codeword c with a non-codeword c' such that coverage is maintained.
        # This doesn't reduce size immediately, but might allow further removals.
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Pick a random codeword to try to swap
            if not codeword_ints:
                break
            c_int = random.choice(list(codeword_ints))
            
            # Find words covered ONLY by c_int
            critical_words = []
            ball_c = ball_cache[c_int]
            for w_int in ball_c:
                if cover_count[w_int] == 1:
                    critical_words.append(w_int)
            
            if not critical_words:
                # If no critical words, we can just remove it (handled in greedy, but maybe greedy missed it?)
                # Or we can swap it with anything to shuffle.
                # Let's just try to remove it directly if possible
                removable = True
                for w_int in ball_c:
                    if cover_count[w_int] <= 1:
                        removable = False
                        break
                if removable:
                    codeword_ints.remove(c_int)
                    for w_int in ball_c:
                        cover_count[w_int] -= 1
                        word_to_codewords[w_int].remove(c_int)
                    improved = True
                    # Restart greedy removal
                    codeword_ints, cover_count, word_to_codewords = greedy_remove(codeword_ints, cover_count, word_to_codewords)
                continue
            
            # Find a non-codeword c' that covers all critical_words
            found_swap = False
            # Randomly sample non-codewords to speed up
            # Or iterate all if time permits
            candidates = []
            for w_int in range(num_words):
                if w_int in codeword_ints:
                    continue
                # Check if w_int covers all critical_words
                # critical_words are in ball_cache[c_int]. 
                # We need w_int to be in ball_cache[critical_word] for all critical_word?
                # No, w_int must cover the critical words. i.e., critical_word is in ball_cache[w_int].
                # Since distance is symmetric, w_int is in ball_cache[critical_word].
                valid = True
                for cw in critical_words:
                    if w_int not in ball_cache[cw]:
                        valid = False
                        break
                if valid:
                    # Additional check: does swapping c_int with w_int maintain coverage for other words?
                    # Words covered by c_int but not w_int must be covered by others.
                    # Words covered by w_int but not c_int are new.
                    # We only care about words that were ONLY covered by c_int.
                    # Those are critical_words.
                    # If w_int covers all critical_words, then coverage is maintained.
                    # We don't need to check others because they had cover_count >= 2 (otherwise they would be critical).
                    # Wait, if a word w was covered by c_int and one other, and we remove c_int, it's still covered by the other.
                    # So yes, only critical words matter.
                    
                    # Perform swap
                    # Remove c_int
                    codeword_ints.remove(c_int)
                    for w_int2 in ball_c:
                        cover_count[w_int2] -= 1
                        word_to_codewords[w_int2].remove(c_int)
                    
                    # Add w_int
                    codeword_ints.add(w_int)
                    for w_int2 in ball_cache[w_int]:
                        cover_count[w_int2] += 1
                        word_to_codewords[w_int2].append(w_int)
                        
                    improved = True
                    found_swap = True
                    # Restart greedy removal
                    codeword_ints, cover_count, word_to_codewords = greedy_remove(codeword_ints, cover_count, word_to_codewords)
                    break
            
            if not found_swap:
                # Try next random codeword
                continue
        
        if len(codeword_ints) < best_size:
            best_size = len(codeword_ints)
            best_codewords = codeword_ints.copy()
            
            # If we hit the theoretical lower bound or known optimum, we can stop?
            # Known optimum for K_3(6,1) is 73.
            if best_size == 73:
                # Try a few more to see if 72 is possible? Unlikely.
                # But if we have time, we can keep searching.
                # For now, break if we are confident.
                pass

    if best_codewords is None:
        # Fallback
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [int_to_word(i) for i in range(num_words)]
            
    return [int_to_word(c_int) for c_int in best_codewords]
