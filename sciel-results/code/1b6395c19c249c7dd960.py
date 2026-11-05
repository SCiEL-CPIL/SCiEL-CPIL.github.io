"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from collections import deque

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    # Special case: the known optimal construction for q=3, n=8, radius=1
    # uses the ternary tetracode (9 codewords of length 4) tensor product
    # with all words on remaining coordinates. This gives 9 * 3^4 = 729 words.
    # But we can do better with a greedy set-cover approach.
    
    if q == 3 and n == 8 and radius == 1:
        return _greedy_cover(q, n, radius, start_time, deadline)
    
    # Fallback: direct product construction
    if n % 4 == 0 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        remaining = n // 4 - 1
        result = []
        for _ in range(remaining):
            pass
        # For n=8: tetracode x all 3^4 tails
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic greedy fallback
    return _greedy_cover(q, n, radius, start_time, deadline)


def _greedy_cover(q, n, radius, start_time, deadline):
    """Greedy set-cover: repeatedly pick the codeword that covers the most uncovered words."""
    total_words = q ** n
    
    # Precompute all words as integers for fast distance computation
    # We'll use a bitset-like approach with numpy for speed
    
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    # all_words shape: (q^n, n)
    
    # For radius=1, a codeword covers itself and all words at Hamming distance 1
    # Number of words covered by one codeword: 1 + n*(q-1)
    
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    # Precompute for each position, the indices of words that differ only in that position
    # Actually, let's use a different approach: represent words as base-q integers
    # and precompute neighbors
    
    # Convert all words to base-q integers
    powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    word_ints = all_words.T.dot(powers)
    
    # For each word index, precompute which other words are at distance 1
    # This is expensive for large spaces. For q=3, n=8: 6561 words, each has 16 neighbors
    # So 6561 * 16 = ~105k edges, manageable.
    
    # Build adjacency: for each word, list of words at distance <= 1
    if radius == 1:
        # For each position i, for each word, the neighbors are obtained by changing
        # coordinate i to each other value in 0..q-1
        # Precompute: for each word index, the list of indices at distance 1
        # We'll compute coverage incrementally
        
        # For speed, precompute for each position i and each value v, the mask of words
        # that have value v at position i
        # Then neighbors of word w at distance 1 = union over positions i of {words with same
        # values as w except position i can be anything}
        
        # Alternative: just iterate through all candidate codewords and compute how many
        # new words they cover. With 6561 candidates and 6561 words, this is 6561^2 ~ 43M
        # operations which might be too slow in pure Python. Use numpy.
        
        # Better approach: maintain a count of how many codewords cover each word.
        # When we add a codeword, increment counts for all words within distance 1.
        # A word is "covered" when count > 0.
        
        # Precompute neighbor lists
        # For word at index idx, its neighbors at distance 1:
        # For each position pos (0..n-1), for each alt_val in 0..q-1 except current:
        #   new_word = word with position pos changed to alt_val
        #   find its index in the all_words array
        
        # Create a lookup: from word tuple to index
        # Since words are in lexicographic order from itertools.product,
        # word (w0, w1, ..., w_{n-1}) has index w0 * q^{n-1} + w1 * q^{n-2} + ... + w_{n-1}
        
        def word_to_index(w):
            idx = 0
            for j, val in enumerate(w):
                idx = idx * q + val
            return idx
        
        # Precompute for each word index, the list of indices at distance 1
        # This is 6561 * 16 = 104,976 entries
        neighbor_lists = []
        for idx in range(total_words):
            # Convert idx to word
            w = []
            tmp = idx
            for j in range(n - 1, -1, -1):
                w.append(tmp % q)
                tmp //= q
            w.reverse()
            
            neighbors = [idx]  # include self
            for pos in range(n):
                orig_val = w[pos]
                for alt_val in range(q):
                    if alt_val == orig_val:
                        continue
                    w[pos] = alt_val
                    nidx = word_to_index(w)
                    neighbors.append(nidx)
                w[pos] = orig_val
            neighbor_lists.append(neighbors)
        
        # Now greedy: repeatedly pick the word that covers the most uncovered words
        uncovered_count = np.ones(total_words, dtype=np.int32)  # how many uncovered words each candidate covers
        # Actually, we need: for each candidate codeword, how many currently-uncovered words it covers
        
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        
        # Initial: every candidate covers 1 + n*(q-1) words
        # But we need to track which are already covered
        
        # For efficiency, maintain for each word whether it's covered, and for each
        # candidate, the count of uncovered words it covers.
        
        # When we cover a word w, we need to decrement the count for every candidate
        # that covers w. This is the expensive part.
        
        # Alternative: use a priority queue. For each candidate, compute its coverage.
        # Pick the best, mark those words covered, then update.
        
        # For q=3, n=8, this should be feasible with careful implementation.
        
        # Let's use a simpler approach: iterate, each time pick the candidate with max
        # uncovered coverage, add it, update.
        
        # Precompute: for each word w, which candidates cover w (i.e., candidates at distance <= 1 from w)
        # This is the reverse adjacency.
        # word_covered_by[w] = list of candidate indices that cover w
        
        # Actually, candidates ARE all words. So candidate c covers word w iff dist(c, w) <= radius.
        # For radius=1, w is covered by c iff c == w or dist(c,w) == 1.
        # So word_covered_by[w] = [w] + [c for c at distance 1 from w]
        # Which is exactly neighbor_lists[w]!
        
        # So: when word w becomes covered, for each c in neighbor_lists[w], decrement c's uncovered count.
        
        # Initialize uncovered_count[c] = len(neighbor_lists[c]) for all c
        uncovered_count = np.array([len(nl) for nl in neighbor_lists], dtype=np.int32)
        
        max_candidates = 5000  # safety
        while time.time() < deadline - 5.0 and len(codewords) < max_candidates:
            # Find candidate with max uncovered_count
            best_idx = int(np.argmax(uncovered_count))
            if uncovered_count[best_idx] <= 0:
                break
            
            codewords.append(all_words[best_idx].tolist())
            
            # Mark all words in neighbor_lists[best_idx] as covered
            for w in neighbor_lists[best_idx]:
                if not covered[w]:
                    covered[w] = True
                    # Decrement uncovered_count for all candidates that cover w
                    for c in neighbor_lists[w]:
                        uncovered_count[c] -= 1
        
        # Check if fully covered
        if not covered.all():
            # Add remaining uncovered words as codewords
            for w in range(total_words):
                if not covered[w]:
                    codewords.append(all_words[w].tolist())
        
        return codewords
    else:
        # General radius: fallback to direct product
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Just return all words (worst case)
            return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
