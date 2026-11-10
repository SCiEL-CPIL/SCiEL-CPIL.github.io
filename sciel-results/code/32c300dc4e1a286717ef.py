"""Football pool problem for 8 matches: ternary covering code K_3(8,1)"""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Total words in space
    total_words = q ** n
    
    # Each codeword covers 1 + n*(q-1) words within radius
    # For q=3, n=8, radius=1: 1 + 8*2 = 17 words per codeword
    
    # Generate all words as integers 0..6560
    # Convert int to list of n digits in base q
    def int_to_word(x):
        w = [0] * n
        for i in range(n):
            w[i] = x % q
            x //= q
        return w
    
    def word_to_int(w):
        x = 0
        for i in range(n):
            x = x * q + w[i]
        return x
    
    # Precompute: for each word, the set of words within radius 1
    # For radius 1, neighbors are: self + change one position to each other value
    # We'll use a bitset approach for speed
    # 6561 bits = 6561/8 ≈ 821 bytes, use Python integers as bitsets
    
    # For each position i and value change, precompute the shift
    # Actually, let's precompute the neighborhood mask for each word
    # neighborhood[w] = bitmask of all words within radius 1 of w
    
    # Better: precompute for each word index, the list of neighbor indices
    # But that's 6561 * 17 = 111537 entries, manageable
    
    # Let's use a different approach: greedy set cover
    # Start with a good initial solution, then try to improve
    
    # Initial solution: direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert to set of tuples for fast lookup
    code_set = set(tuple(c) for c in initial_code)
    
    # Compute coverage: which words are covered
    # A word is covered if it's within radius 1 of some codeword
    # For radius 1: word w is covered if w is in code_set, or 
    #   for some position i, changing w[i] to another value gives a codeword
    
    def is_covered(w, code_set):
        # Check if w is in code_set
        if tuple(w) in code_set:
            return True
        # Check if changing one position gives a codeword
        for i in range(n):
            for v in range(q):
                if v == w[i]:
                    continue
                nw = list(w)
                nw[i] = v
                if tuple(nw) in code_set:
                    return True
        return False
    
    # This is too slow for 6561 words. Let's use a smarter approach.
    
    # Precompute for each position i, group words by their value at position i
    # Actually, let's think differently.
    
    # For the greedy approach:
    # 1. Start with empty code
    # 2. Repeatedly add the codeword that covers the most uncovered words
    # 3. Stop when all words are covered
    
    # But we need a good initial solution. Let's use the tetracode sum as seed
    # and then try to remove redundant codewords.
    
    # First, let's verify the initial solution covers everything
    # and count how many words each codeword uniquely covers.
    
    # Better strategy: Start with the 729-word solution, then greedily remove
    # codewords that don't reduce coverage.
    
    # For each codeword, check if removing it leaves all its covered words still covered
    # by other codewords.
    
    # To check this efficiently, for each codeword c, find all words within radius 1
    # of c. For each such word w, check if w is within radius 1 of some other codeword.
    
    # This is expensive. Let's use a counting approach:
    # For each word w, count how many codewords are within radius 1 of w.
    # If all counts >= 1, we have a valid cover.
    # A codeword c is redundant if for all words w within radius 1 of c,
    # the count of other codewords within radius 1 of w is >= 1.
    
    # Let's compute coverage counts.
    
    # For each codeword, generate all words within radius 1 and increment their count.
    # Then a codeword is redundant if removing it doesn't make any count go to 0.
    
    # Step 1: Compute coverage counts for the initial 729-word code
    coverage_count = [0] * total_words
    
    for c in initial_code:
        ct = tuple(c)
        # The codeword itself
        ci = word_to_int(c)
        coverage_count[ci] += 1
        # All words at distance 1
        for i in range(n):
            for v in range(q):
                if v == c[i]:
                    continue
                nc = list(c)
                nc[i] = v
                ni = word_to_int(nc)
                coverage_count[ni] += 1
    
    # Now try to remove redundant codewords
    # A codeword c is removable if for every word w within radius 1 of c,
    # coverage_count[w] > 1 (i.e., some other codeword also covers w)
    
    # We'll iterate: find removable codewords, remove them, update counts, repeat.
    
    code_list = list(initial_code)
    code_set_current = set(tuple(c) for c in code_list)
    
    # We need to be careful: when we remove a codeword, we update counts,
    # and other codewords may become removable.
    
    # Use a queue of codewords to check for removability.
    # Start with all codewords.
    
    import heapq
    
    # Let's just do iterative passes
    removed = set()
    progress = True
    pass_count = 0
    
    while progress and time.time() < deadline:
        progress = False
        pass_count += 1
        
        # Check each codeword not yet removed
        for idx, c in enumerate(code_list):
            if idx in removed:
                continue
            
            # Check if this codeword is redundant
            redundant = True
            ct = tuple(c)
            ci = word_to_int(c)
            
            # Check the codeword itself
            if coverage_count[ci] <= 1:
                redundant = False
                break
            
            # Check all distance-1 neighbors
            for i in range(n):
                if not redundant:
                    break
                for v in range(q):
                    if v == c[i]:
                        continue
                    nc = list(c)
                    nc[i] = v
                    ni = word_to_int(nc)
                    if coverage_count[ni] <= 1:
                        redundant = False
                        break
            
            if redundant:
                # Remove this codeword
                removed.add(idx)
                progress = True
                
                # Update coverage counts
                coverage_count[ci] -= 1
                for i in range(n):
                    for v in range(q):
                        if v == c[i]:
                            continue
                        nc = list(c)
                        nc[i] = v
                        ni = word_to_int(nc)
                        coverage_count[ni] -= 1
    
    # Build the final code
    final_code = [c for idx, c in enumerate(code_list) if idx not in removed]
    
    # Verify coverage
    # Recompute coverage from scratch to be safe
    final_set = set(tuple(c) for c in final_code)
    
    # Check if all words are covered
    # For each word, check if it's within radius 1 of some codeword
    # This is expensive, so let's use the coverage count approach again
    
    verify_count = [0] * total_words
    for c in final_code:
        ci = word_to_int(c)
        verify_count[ci] += 1
        for i in range(n):
            for v in range(q):
                if v == c[i]:
                    continue
                nc = list(c)
                nc[i] = v
                ni = word_to_int(nc)
                verify_count[ni] += 1
    
    # Check all words are covered
    all_covered = all(verify_count[w] >= 1 for w in range(total_words))
    
    if not all_covered:
        # Fallback to initial code
        return initial_code
    
    return final_code

# EVOLVE-BLOCK-END
