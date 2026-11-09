import itertools
import random
import time

def _word_to_tuple(w):
    return tuple(w)

def _tuple_to_word(t):
    return list(t)

def _get_covers(q, n):
    """
    Precompute for every word in {0..q-1}^n, the list of words within distance 1.
    Returns a dictionary: word_tuple -> list of word_tuples covering it.
    Actually, we need: word_tuple -> set of words it covers (for greedy gain calculation)
    And: word_tuple -> list of words it covers (for deletion check)
    """
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    covers = {}
    for w in all_words:
        covers[w] = set()
        # distance 0
        covers[w].add(w)
        # distance 1
        for i in range(n):
            for val in range(q):
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                covers[w].add(tuple(new_w))
                
    return all_words, covers

def _optimize_deletion(code, all_words, covers, q, n):
    """
    Greedily remove redundant codewords.
    code: list of tuples
    all_words: list of all tuples
    covers: dict tuple -> set of tuples
    """
    # Map word to index for fast counting?
    # We maintain cover_count for each word in all_words.
    # cover_count[w] = number of codewords in 'code' that cover w.
    
    # Initialize cover_count
    cover_count = [0] * len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    for c in code:
        for w in covers[c]:
            cover_count[word_to_idx[w]] += 1
            
    code_set = set(code)
    
    # Iterative deletion
    # To make it more effective, we can try removing words that are "most redundant".
    # A word c is redundant if for all w in covers[c], cover_count[w] > 1.
    # We can sort candidates by "safety margin" or just iterate.
    # Iterating and removing the first valid one is standard.
    # To improve, we can try all and remove the one that leaves the "weakest" coverage?
    # Or just remove any. Let's stick to any, but restart the loop.
    
    changed = True
    while changed:
        changed = False
        # Shuffle to avoid bias? Or just iterate.
        # Let's iterate in a fixed order for determinism in one pass, 
        # but we can shuffle in the outer loop if we call this multiple times.
        for c in list(code_set):
            # Check if c can be removed
            can_remove = True
            for w in covers[c]:
                if cover_count[word_to_idx[w]] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                code_set.remove(c)
                for w in covers[c]:
                    cover_count[word_to_idx[w]] -= 1
                changed = True
                break # Restart iteration as code_set changed
    
    return list(code_set)

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # Precompute space and covers
    all_words, covers = _get_covers(q, n)
    
    best_code = None
    best_size = float('inf')
    
    # Strategy 1: Direct Sum Construction
    if n >= 4:
        # Tetracode for q=3, n=4
        # If q != 3, this specific construction doesn't apply directly.
        # The task specifies q=3, n=6, radius=1.
        # But the interface is generic.
        # We should handle the specific case q=3, n=6 optimally.
        # For other cases, fallback to Greedy.
        if q == 3 and n == 6:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            init_code = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            
            # Optimize
            opt_code = _optimize_deletion(init_code, all_words, covers, q, n)
            if len(opt_code) < best_size:
                best_size = len(opt_code)
                best_code = opt_code
        
        # Strategy 2: Greedy Set Cover
        # We can run this multiple times with random tie-breaking
        
        # Helper for greedy
        def run_greedy(randomize=False):
            uncovered = set(all_words)
            code = set()
            
            while uncovered:
                best_gain = -1
                best_c = None
                candidates = []
                
                for c in all_words:
                    gain = 0
                    for w in covers[c]:
                        if w in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                        candidates = [c]
                    elif gain == best_gain and gain > 0:
                        candidates.append(c)
                
                if best_gain == 0:
                    break # Should not happen if code is valid, but safety
                    
                if randomize and len(candidates) > 1:
                    chosen = random.choice(candidates)
                else:
                    chosen = best_c
                
                code.add(chosen)
                for w in covers[chosen]:
                    uncovered.discard(w)
                    
            return code
        
        # Run deterministic greedy
        if time.time() < deadline:
            greedy_code = run_greedy(randomize=False)
            opt_code = _optimize_deletion(list(greedy_code), all_words, covers, q, n)
            if len(opt_code) < best_size:
                best_size = len(opt_code)
                best_code = opt_code
        
        # Run randomized greedy with deletion
        # Each iteration: Greedy (random) -> Deletion
        # This explores different local optima.
        
        # We can also try to "kick-start" deletion with random subsets?
        # No, greedy is a good start.
        
        while time.time() < deadline:
            greedy_code = run_greedy(randomize=True)
            opt_code = _optimize_deletion(list(greedy_code), all_words, covers, q, n)
            if len(opt_code) < best_size:
                best_size = len(opt_code)
                best_code = opt_code
            
            # Optional: Try to improve further by local search on the optimized code?
            # The deletion is already a local search (1-swap for removal).
            # We could try 1-swap (replace one word with another) to see if it allows further deletion?
            # That's more complex. Let's stick to multiple restarts of Greedy+Delete.
            
            # If we reach 73, we can stop? 73 is best known.
            if best_size <= 73:
                break

    # If no code found (e.g. q!=3 or n<4 and greedy failed?), fallback to simple greedy
    if best_code is None:
        greedy_code = run_greedy(randomize=False)
        best_code = list(greedy_code)

    # Convert tuples back to lists
    return [list(c) for c in best_code]
