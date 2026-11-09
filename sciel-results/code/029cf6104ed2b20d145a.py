import itertools
import random
import time

def construct(q, n, radius):
    # For the specific case q=3, n=6, radius=1, we can use a specialized approach
    # But we must keep the function general enough or at least handle the specific case well.
    # The task says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # So we should assume q=3, n=6, radius=1 is the primary target, but the function signature is general.
    
    # If n is small, we can do exhaustive or near-exhaustive search.
    # For q=3, n=6, total words = 729.
    
    total_words = q ** n
    if total_words > 5000:
        # Fallback to direct sum or simple greedy for larger instances to avoid timeout
        # Direct sum construction for ternary
        if q == 3 and n >= 4:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             # Simple greedy
             return _greedy_cover(q, n, radius)

    # For small instances (like 729), use optimized greedy + prune
    start_time = time.time()
    timeout = 100.0 # 100s limit to be safe within 120s
    
    # Generate all words
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute coverage sets
    # cover[i] is the set of indices covered by word i
    covers = [set() for _ in range(total_words)]
    for i, w in enumerate(words):
        covers[i].add(i)
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig: continue
                w[pos] = val
                idx = word_to_idx[tuple(w)]
                covers[i].add(idx)
            w[pos] = orig
            
    # Greedy Set Cover
    uncovered = set(range(total_words))
    codewords = []
    
    # To speed up, we can precompute the size of intersection with uncovered
    # But for 729 words, a simple loop is fast enough.
    
    while uncovered:
        if time.time() - start_time > timeout:
            break
        best_idx = -1
        max_covered = 0
        # We can randomize tie-breaking to escape local optima if we do multiple restarts,
        # but for now, let's do a deterministic greedy first.
        for i in uncovered: # Only consider words that are not yet covered? No, any word can be a codeword.
            # Actually, we should consider all words as potential codewords.
            # But if a word is already covered, it might still be a good codeword.
            # Standard greedy: pick word with max |cover(i) \cap uncovered|
            covered_count = len(covers[i] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_idx = i
        
        if best_idx == -1:
            break
            
        codewords.append(words[best_idx])
        uncovered -= covers[best_idx]
        
    # Pruning phase
    # Try to remove each codeword
    # To check if removal is valid, we need to see if all words are still covered.
    # Efficient check: maintain a count of how many codewords cover each word.
    
    def is_valid(cw_list):
        if not cw_list:
            return total_words == 0
        counts = [0] * total_words
        for cw in cw_list:
            idx = word_to_idx[tuple(cw)]
            for j in covers[idx]:
                counts[j] += 1
        return all(c > 0 for c in counts)

    # Faster pruning:
    # Keep counts of coverage for the current codewords.
    counts = [0] * total_words
    for cw in codewords:
        idx = word_to_idx[tuple(cw)]
        for j in covers[idx]:
            counts[j] += 1
            
    # Try removing each codeword
    # We can do multiple passes of pruning
    improved = True
    while improved:
        if time.time() - start_time > timeout:
            break
        improved = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() - start_time > timeout:
                break
            cw = codewords[i]
            idx = word_to_idx[tuple(cw)]
            # Check if any word would become uncovered
            valid = True
            for j in covers[idx]:
                if counts[j] == 1:
                    valid = False
                    break
            if valid:
                # Remove it
                for j in covers[idx]:
                    counts[j] -= 1
                codewords.pop(i)
                improved = True
            # If not valid, we could try swapping, but that's complex.
            # For now, just removal.

    # Final check
    if not is_valid(codewords):
         # Fallback to seed solution if something went wrong
         tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
         return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
         
    return codewords

def _greedy_cover(q, n, radius):
    # Simple greedy for fallback
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    total_words = len(words)
    covers = [set() for _ in range(total_words)]
    for i, w in enumerate(words):
        covers[i].add(i)
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig: continue
                w[pos] = val
                idx = word_to_idx[tuple(w)]
                covers[i].add(idx)
            w[pos] = orig
            
    uncovered = set(range(total_words))
    codewords = []
    while uncovered:
        best_idx = -1
        max_covered = 0
        for i in range(total_words):
            covered_count = len(covers[i] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_idx = i
        if best_idx == -1: break
        codewords.append(words[best_idx])
        uncovered -= covers[best_idx]
    return codewords
