"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # If n==4, we already have a perfect code of size 9
    if n == 4:
        return base
    
    # For larger n, try to improve using greedy set cover + local search
    # The space is {0,1,2}^n with 3^n words
    # We need to cover all words within Hamming distance <= radius
    
    # Deadline guard: 145 seconds (5s safety margin)
    deadline = time.time() + 145.0
    
    # Build the universe of all words
    # For n=7, 3^7 = 2187 words, manageable
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute neighborhoods: for each word, all words within distance <= radius
    # For radius=1, each word covers itself and its 2*(n-1) neighbors... wait, q-1 per position
    # For ternary q=3, radius=1: each word covers 1 + n*(q-1) = 1 + 7*2 = 15 words
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage sets
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
    
    # For efficiency, precompute for each word which other words are within radius
    # coverage[word] = set of words covered by placing a codeword at 'word'
    coverage = {}
    for w in all_words:
        cov = set()
        for v in all_words:
            if hamming_dist(w, v) <= radius:
                cov.add(v)
        coverage[w] = cov
    
    # Greedy set cover: start with the base code, then try to reduce
    # Actually, let's do a fresh greedy set cover from scratch, seeded with good initial choices
    
    # Better approach: use the tetracode construction as initial solution, then try to remove redundant codewords
    # and add better ones
    
    # Start with the base code
    current_code = list(base)
    current_code_set = set(tuple(c) for c in current_code)
    
    # Check if current code covers everything
    def is_covering(code_set):
        # Check every word is within radius of some codeword
        covered = set()
        for cw in code_set:
            for v in all_words:
                if hamming_dist(cw, v) <= radius:
                    covered.add(v)
        return covered == set(all_words)
    
    # Verify initial code covers
    if not is_covering(current_code_set):
        # Shouldn't happen, but fall back
        return current_code
    
    # Try to reduce: for each codeword, check if removing it still maintains coverage
    # Sort by some heuristic (e.g., how many words it uniquely covers)
    
    def unique_coverage(code_set, cw):
        """Count words covered only by this codeword (not by any other in the set)"""
        other_set = set(code_set - {cw})
        covered_by_others = set()
        for ocw in other_set:
            for v in all_words:
                if hamming_dist(ocw, v) <= radius:
                    covered_by_others.add(v)
        unique = 0
        for v in all_words:
            if hamming_dist(cw, v) <= radius and v not in covered_by_others:
                unique += 1
        return unique
    
    # Iteratively remove codewords that don't uniquely cover anything
    improved = True
    while improved and time.time() < deadline:
        improved = False
        code_list = list(current_code)
        for i in range(len(code_list) - 1, -1, -1):
            cw = tuple(code_list[i])
            if unique_coverage(current_code_set, cw) == 0:
                current_code_set.remove(cw)
                code_list.pop(i)
                improved = True
                if time.time() > deadline:
                    return current_code_set
        current_code = code_list
    
    # Now try to swap: remove one codeword and add another that covers more uncovered... 
    # Actually after removal, everything is still covered. Try local search:
    # For each pair (remove cw1, add cw2), check if still covering and if it helps
    
    # Better: try to replace a codeword with another that covers a different set, enabling further reductions
    # Use simulated annealing or simple hill climbing
    
    # Hill climbing: try removing each codeword, if it breaks coverage, try adding the best single codeword to fix
    # This is complex. Let's try a different approach:
    
    # Random restart greedy set cover
    best_code = list(current_code)
    
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        
        # Greedy set cover with random tie-breaking
        uncovered = set(all_words)
        code = []
        
        # Precompute for each candidate word, which uncovered words it covers
        # For efficiency, iterate
        while uncovered and time.time() < deadline - 5:
            # Find the word that covers the most uncovered words
            best_word = None
            best_count = -1
            best_cov = None
            
            # Sample candidates for speed if space is large
            candidates = list(all_words)
            random.shuffle(candidates)
            
            for cw in candidates:
                cov_uncovered = 0
                for v in uncovered:
                    if hamming_dist(cw, v) <= radius:
                        cov_uncovered += 1
                if cov_uncovered > best_count:
                    best_count = cov_uncovered
                    best_word = cw
                    best_cov = set()
                    for v in uncovered:
                        if hamming_dist(cw, v) <= radius:
                            best_cov.add(v)
            
            if best_word is None or best_count == 0:
                break
            
            code.append(best_word)
            uncovered -= best_cov
        
        if len(uncovered) == 0 and len(code) < len(best_code):
            best_code = code
    
    # Final local search on best_code: try removing redundant words
    best_set = set(tuple(c) for c in best_code)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        code_list = list(best_set)
        for i in range(len(code_list) - 1, -1, -1):
            cw = tuple(code_list[i])
            if unique_coverage(best_set, cw) == 0:
                best_set.remove(cw)
                code_list.pop(i)
                improved = True
    
    return [list(c) for c in best_set]
# EVOLVE-BLOCK-END
