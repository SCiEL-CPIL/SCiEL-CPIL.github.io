"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    # For the specific case of ternary length 7 radius 1, we use a greedy set cover
    # with a strong heuristic (choose the set that covers the most uncovered elements)
    # combined with local search / hill climbing to reduce the number of codewords.
    #
    # The seed uses the direct sum of the ternary tetracode: 9 * 3^3 = 243 words.
    # We aim to do better than 243, ideally approaching the known best of 186.
    #
    # Strategy:
    # 1. Start with a reasonable initial covering (e.g., greedy).
    # 2. Use local search: try removing a codeword if the coverage is still complete,
    #    or swap a codeword for another that covers more uncovered words.
    #
    # Given the 120s time limit, we need an efficient approach.
    #
    # For q=3, n=7, radius=1:
    # Total words: 3^7 = 2187
    # Each codeword covers itself and its 2*7=14 neighbors (distance 1), so 15 words per codeword.
    # Lower bound: 2187/15 = 145.8 -> 146. Known best: 186.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to the seed approach for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Specialized solver for q=3, n=7, radius=1
    # Precompute all words and their neighbors
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)  # 2187
    
    # For each word, compute the set of codewords that would cover it (i.e., words at distance <= 1)
    # A word w is covered by codeword c if Hamming distance <= 1.
    # So for each word w, the potential codewords are w itself and all words at distance 1 from w.
    # There are 1 + 2*7 = 15 such words.
    
    # Build the incidence: for each word index, list of codeword indices that cover it
    # Actually, we want: for each codeword c, which words does it cover?
    # Let's build: cover[c] = set of word indices covered by codeword c
    
    # Precompute neighbors for each word
    def get_neighbors(w):
        """Return all words at Hamming distance <= 1 from w."""
        neighbors = [w]
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    neighbors.append(tuple(new_w))
        return neighbors
    
    # Build cover sets: for each possible codeword (which is a word in the space), 
    # what words does it cover?
    # Since any word can be a codeword, we have 2187 possible codewords.
    # cover[i] = list of word indices covered by codeword i
    
    cover = [None] * N
    for i, w in enumerate(all_words):
        neighbors = get_neighbors(w)
        cover[i] = [word_to_idx[nb] for nb in neighbors]
    
    # Greedy set cover to get an initial solution
    uncovered = set(range(N))
    codewords = []
    
    # For efficiency, use a list of uncovered and a count of how many codewords cover each uncovered word
    # Greedy: at each step, pick the codeword that covers the most uncovered words
    
    # To make this efficient, we'll track for each codeword how many uncovered words it covers
    # Initially, each codeword covers 15 words, but we need to track which are uncovered
    
    # Simple greedy with periodic recomputation
    # Given N=2187 and 2187 possible codewords, each covering 15 words, this is manageable
    
    # More efficient greedy: maintain a count of uncovered words per codeword
    # When we pick a codeword, decrement the count for all codewords that share uncovered words with it
    
    # Let's use a simpler approach: since the problem is small enough, use a standard greedy
    # with a priority queue or just scan each time (2187 * 2187/15 ~ 300k operations, fine)
    
    # Actually, let's use a smarter greedy:
    # 1. Start with all words uncovered
    # 2. Repeatedly pick the codeword that covers the most uncovered words
    # 3. Mark those words as covered
    
    # To avoid O(N^2) per step, we can precompute the cover sets and use a heap
    # But for 2187 words, even O(N^2) total is ~4.8M, which is fine in 120s
    
    # Let's do a straightforward greedy:
    import heapq
    
    # For each codeword, count how many uncovered words it covers
    # Initially all are uncovered, so each codeword covers 15 words
    # But as we add codewords, we need to update counts
    
    # Use a max-heap: (-count, codeword_idx)
    # We'll need to lazily delete stale entries
    
    uncovered_set = set(range(N))
    
    # count[i] = number of uncovered words covered by codeword i
    count = [15] * N  # Each codeword covers 15 words initially
    
    heap = [(-15, i) for i in range(N)]
    heapq.heapify(heap)
    
    codewords = []
    
    while uncovered_set:
        # Get the best codeword
        while heap:
            neg_count, i = heap[0]
            if -neg_count == count[i] and count[i] > 0:
                break
            heapq.heappop(heap)
        
        if not heap:
            break
        
        neg_count, i = heapq.heappop(heap)
        if count[i] == 0:
            continue
        
        # Add codeword i
        codewords.append(i)
        
        # Mark covered words
        for w in cover[i]:
            if w in uncovered_set:
                uncovered_set.remove(w)
                # For all codewords that cover w, decrement their count
                # We need to know which codewords cover w
                # Precompute: for each word w, which codewords cover it?
                # That's just the neighbors of w, i.e., get_neighbors(w)
                # But we already have that implicitly
    
    # This is getting complex. Let me use a simpler but effective approach:
    # Start with the tetracode-based solution (243 words), then try to reduce it.
    
    # Actually, let me just implement a clean greedy from scratch:
    
    uncovered = [True] * N
    num_uncovered = N
    
    # For each codeword, track how many uncovered words it covers
    # cover_count[i] = number of j such that uncovered[j] and j in cover[i]
    cover_count = [15] * N  # Initially all 15 are uncovered
    
    # We also need: for each word j, which codewords cover it?
    # That's the same as the neighbors of word j
    # Let's precompute: covers_word[j] = list of codeword indices that cover word j
    covers_word = [None] * N
    for j in range(N):
        w = all_words[j]
        neighbors = get_neighbors(w)
        covers_word[j] = [word_to_idx[nb] for nb in neighbors]
    
    # Greedy set cover
    codewords = []
    
    # Use a simple approach: repeatedly find the codeword with max cover_count
    # N=2187, each iteration is O(N) to find max, and we do ~200 iterations, so ~437k ops
    
    while num_uncovered > 0:
        # Find codeword with max cover_count
        max_idx = -1
        max_count = -1
        for i in range(N):
            if cover_count[i] > max_count:
                max_count = cover_count[i]
                max_idx = i
        
        if max_idx == -1 or max_count == 0:
            break
        
        codewords.append(max_idx)
        
        # Update: mark words covered by max_idx as covered
        for w in cover[max_idx]:
            if uncovered[w]:
                uncovered[w] = False
                num_uncovered -= 1
                # Decrement cover_count for all codewords that cover w
                for cw in covers_word[w]:
                    cover_count[cw] -= 1
    
    # Now we have an initial solution. Try to improve it with local search.
    # Local search: try to remove a codeword if possible, or swap.
    
    # First, let's see how many we got
    # The greedy should give something around 150-200
    
    # Local search: for each codeword, check if we can remove it
    # A codeword can be removed if all words it uniquely covers are also covered by other codewords
    
    # Build a coverage count: for each word, how many codewords cover it
    cov_count = [0] * N
    for i in codewords:
        for w in cover[i]:
            cov_count[w] += 1
    
    # Try removing codewords
    changed = True
    while changed:
        changed = False
        for idx in range(len(codewords) - 1, -1, -1):
            i = codewords[idx]
            # Check if all words covered by i are covered by at least one other codeword
            can_remove = True
            for w in cover[i]:
                if cov_count[w] == 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove it
                for w in cover[i]:
                    cov_count[w] -= 1
                codewords.pop(idx)
                changed = True
    
    # Convert indices back to codewords (lists of integers)
    result = [list(all_words[i]) for i in codewords]
    
    return result
# EVOLVE-BLOCK-END
