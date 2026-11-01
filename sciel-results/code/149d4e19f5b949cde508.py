import time
import itertools
import random

_DEADLINE = 150.0
_MARGIN = 5.0

def _all_words(q, n):
    return itertools.product(range(q), repeat=n)

def _hamming_dist(a, b):
    return sum(x != y for x, y in zip(a, b))

def _neighbors(word, q, n):
    """Yield all words at Hamming distance exactly 1 from word."""
    w = list(word)
    for i in range(n):
        orig = w[i]
        for v in range(q):
            if v != orig:
                w[i] = v
                yield tuple(w)
        w[i] = orig

def construct(q, n, radius):
    start_time = time.time()
    
    # Precompute balls for all words
    all_words_list = list(_all_words(q, n))
    total_words = len(all_words_list)
    word_to_idx = {w: i for i, w in enumerate(all_words_list)}
    
    # balls[i] is a set of indices of words covered by word i
    balls = [set() for _ in range(total_words)]
    for i, w in enumerate(all_words_list):
        balls[i].add(i)
        for nb in _neighbors(w, q, n):
            if _hamming_dist(w, nb) <= radius:
                balls[i].add(word_to_idx[nb])
                
    # Initialize code
    # Use Tetracode direct sum for q=3, n>=4, radius=1
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_words = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Fallback: Random or Greedy
        # For now, let's just pick random words until covered
        initial_words = []
        covered = set()
        while len(covered) < total_words:
            w = random.choice(all_words_list)
            if w not in initial_words:
                initial_words.append(w)
                covered |= balls[word_to_idx[w]]
                
    # Convert to indices
    code_indices = [word_to_idx[w] for w in initial_words]
    
    # Calculate cover counts
    cover_count = [0] * total_words
    for c in code_indices:
        for w in balls[c]:
            cover_count[w] += 1
            
    # Local Search
    improved = True
    while improved and time.time() < _DEADLINE - _MARGIN:
        improved = False
        
        # Phase 1: Remove redundant codewords
        for i in range(len(code_indices) - 1, -1, -1):
            if time.time() > _DEADLINE - _MARGIN:
                return [list(all_words_list[c]) for c in code_indices]
            
            c = code_indices[i]
            # Check if removing c leaves any word with count 0
            can_remove = True
            for w in balls[c]:
                if cover_count[w] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                for w in balls[c]:
                    cover_count[w] -= 1
                code_indices.pop(i)
                improved = True
                break # Restart loop as indices changed
        
        # Phase 2: Swap codewords
        # Try to swap each codeword with any other word to see if it enables a removal
        # Or simply if the new code is valid, accept it?
        # To avoid oscillation, we only accept a swap if it results in a state
        # where at least one codeword is redundant (cover_count[w] == 1 for some w in the new ball of the swapped-in word? No, that's complex).
        
        # Simpler heuristic: Accept swap if it reduces the "max_cover_count" or "sum of squares of cover counts" (promotes uniformity/redundancy)
        # Let's try: Accept swap if valid AND (new_max_count < old_max_count OR (new_max_count == old_max_count AND random < 0.5))
        # Actually, let's just try swaps and if valid, check if we can remove something immediately after.
        
        for i in range(len(code_indices)):
            if time.time() > _DEADLINE - _MARGIN:
                return [list(all_words_list[c]) for c in code_indices]
                
            c_old = code_indices[i]
            
            # Try swapping with every other word
            # Optimization: Only try words that are not already in the code
            # And maybe only try words that cover "hard" words?
            
            for c_new in range(total_words):
                if c_new == c_old:
                    continue
                if c_new in code_indices:
                    continue
                
                # Check validity of swap
                # New cover count = old - balls[c_old] + balls[c_new]
                # We need min(new_count) >= 1
                
                # To speed up, we can pre-check:
                # If there is a word w in balls[c_old] with count 1, then c_new MUST cover w.
                # If there is a word w in balls[c_new] \ balls[c_old], its count increases, so no risk of 0 there.
                # Risk is only in balls[c_old].
                
                valid_swap = True
                # Check words that lose coverage
                for w in balls[c_old]:
                    if cover_count[w] == 1:
                        # This word is only covered by c_old
                        # So c_new must cover it
                        if w not in balls[c_new]:
                            valid_swap = False
                            break
                
                if not valid_swap:
                    continue
                
                # If valid_swap is true, it means all words with count 1 in balls[c_old] are covered by c_new.
                # Words with count > 1 in balls[c_old] are fine (count decreases but stays >= 1).
                # Words in balls[c_new] not in balls[c_old] get count + 1 (fine).
                # Words in intersection get count unchanged.
                
                # So the swap is valid.
                # Should we accept it?
                # Let's accept it if it creates a redundancy opportunity.
                # Check if in the new state, any codeword can be removed.
                # This is expensive.
                
                # Alternative: Accept swap if it reduces the number of "critical" words (count == 1).
                # Or just accept randomly to explore?
                
                # Let's try a simple hill climbing on "slack": sum(count[w] for w in balls[c] for c in code)
                # No, let's just accept the swap if it is valid.
                # To prevent oscillation, we can keep track of visited states? Too expensive.
                # Let's accept the swap, apply it, and then immediately try to remove.
                
                # Apply swap
                for w in balls[c_old]:
                    cover_count[w] -= 1
                for w in balls[c_new]:
                    cover_count[w] += 1
                code_indices[i] = c_new
                improved = True
                break # Restart outer loop to try removals first
                
            if improved:
                break
                
    # Convert back to lists
    return [list(all_words_list[c]) for c in code_indices]
