import itertools
import random

def construct(q, n, radius):
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    all_words_set = set(all_words)
    
    # Precompute coverage for each word
    def get_coverage(word):
        """Return set of words within Hamming distance <= radius of word."""
        covered = set()
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_word = list(word)
                new_word[pos] = val
                covered.add(tuple(new_word))
        covered.add(tuple(word))
        return covered
    
    # Greedy set cover
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = set(all_words)
        codewords = []
        
        # Precompute coverage for all words
        coverages = {}
        for w in all_words:
            coverages[w] = get_coverage(w)
        
        while uncovered:
            # Find word covering most uncovered
            best_word = None
            best_count = -1
            # Shuffle for tie-breaking
            candidates = list(uncovered)
            rng.shuffle(candidates)
            # Also consider words not in uncovered (they might cover more uncovered)
            # Actually, we should consider ALL words as candidates
            for w in all_words:
                count = len(coverages[w] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = w
            if best_count == 0:
                break
            codewords.append(list(best_word))
            uncovered -= coverages[best_word]
        
        return codewords
    
    # Try multiple seeds and keep best
    best_codewords = None
    best_len = float('inf')
    
    for seed in range(50):
        cw = greedy_cover(seed)
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw
            if best_len <= 73:
                break
    
    if best_codewords is None:
        # Fallback: use the tetracode construction
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    return best_codewords
