"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit

    # For the specific case q=3, n=8, radius=1
    # Best known: 486 (van Laarhoven et al., 1989)
    # Lower bound: 403
    
    # Strategy:
    # 1. Start with a known good construction (486 words)
    # 2. Try to improve via local search / greedy deletion
    
    if q == 3 and n == 8 and radius == 1:
        # Generate the 486-word code from van Laarhoven et al.
        # The construction uses the tetracode (perfect ternary code of length 4, radius 1, size 9)
        # Direct product approach doesn't give 486 directly.
        
        # Actually, let's use a different approach:
        # The 486-word code can be constructed as follows:
        # Take all 3^4 = 81 words of length 4, and for each, add 6 "complementary" patterns
        # on the other 4 positions, but this isn't quite right either.
        
        # Let me think about this more carefully.
        # K_3(8,1) = 486 is achieved by a specific construction.
        # One way: use the fact that 486 = 2 * 3^5 = 2 * 243
        
        # Alternative: start with a large initial code and greedily remove codewords
        # while maintaining coverage.
        
        # Initial code: all words where the first 4 coordinates form a tetracode word
        # and the last 4 coordinates are arbitrary. This gives 9 * 3^4 = 729 words.
        # Then try to reduce.
        
        # Actually, let's use a smarter initial code.
        # The tetracode has 9 words of length 4. If we take the direct sum with all
        # words of length 4, we get 9 * 81 = 729 words.
        
        # A better approach: use a greedy algorithm to find a minimal covering code.
        
        # Let's use a different strategy:
        # 1. Start with the 729-word code (tetracode x all words of length 4)
        # 2. Greedily remove codewords that are "redundant" (their coverage is already
        #    provided by other codewords)
        
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        
        # Generate initial code: tetracode x all 4-tuples
        initial_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=4):
                initial_code.append(h + list(tail))
        
        # Now try to reduce the code size via local search
        # For each codeword, check if removing it still maintains coverage
        # This is expensive, so we'll use a heuristic approach.
        
        # Actually, let's try a different approach: use a simulated annealing
        # or greedy algorithm to find a smaller code.
        
        # For now, let's just return the 729-word code if we don't have time
        # to do better.
        
        # Better approach: use the known 486-word construction.
        # The construction from van Laarhoven et al. uses a specific set of 486 words.
        
        # Let me try to construct it:
        # One known construction: take the set of all words (x1, x2, x3, x4, x5, x6, x7, x8)
        # such that some linear conditions are satisfied.
        
        # Actually, I recall that 486 = 2 * 3^5, so perhaps the code has a
        # linear structure or is a union of two cosets of a linear code.
        
        # Let's try: take a linear [8,5] ternary code with covering radius 1,
        # and add its coset.
        
        # A ternary [8,5] code has 3^5 = 243 words. Two cosets give 486.
        
        # Let's construct a ternary [8,5] code. The generator matrix could be:
        # G = [I_5 | P] where P is a 5x3 matrix.
        
        # For the code to have covering radius 1, every word must be within distance 1
        # of some codeword.
        
        # Let's try G = [I_5 | P] where P is chosen appropriately.
        
        # Actually, let's just use a greedy approach with time limit.
        
        # Start with the 729-word code and try to reduce it.
        
        # For efficiency, let's use numpy for distance computations.
        
        code = initial_code  # 729 words
        code_set = set(tuple(w) for w in code)
        
        # Precompute the "sphere" of each codeword: all words within distance 1
        # Sphere size for q=3, n=8, radius=1: 1 + 8*2 = 17
        
        # For each codeword, compute the set of words it covers
        # Then greedily remove codewords whose coverage is fully contained in
        # the union of others' coverage.
        
        # This is O(m^2 * 17) where m is the code size, which is feasible for m=729.
        
        # But we need to be careful about time. Let's check time first.
        elapsed = time.time() - start_time
        if elapsed > 10:
            return code  # Return the 729-word code if we're running low on time
        
        # Compute coverage sets
        def sphere(word):
            """Return the set of all words within Hamming distance 1 of word."""
            result = {tuple(word)}
            for i in range(n):
                for d in range(q):
                    if d != word[i]:
                        new_word = list(word)
                        new_word[i] = d
                        result.add(tuple(new_word))
            return result
        
        # Compute coverage for each codeword
        coverages = [sphere(w) for w in code]
        
        # Greedy removal: repeatedly remove a codeword if its coverage is
        # contained in the union of the remaining codewords' coverages.
        
        # To speed this up, we'll use a heuristic: remove codewords in a random
        # order and check if they're redundant.
        
        indices = list(range(len(code)))
        random.seed(42)
        random.shuffle(indices)
        
        active = [True] * len(code)
        
        # Union of all active coverages
        # We'll maintain a count for each word in the space of how many codewords cover it.
        # But the space has 3^8 = 6561 words, which is manageable.
        
        # Map each word in the space to an index
        space = list(itertools.product(range(3), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(space)}
        
        # Count coverage for each word in the space
        cover_count = np.zeros(6561, dtype=np.int32)
        for i, w in enumerate(code):
            for covered_word in coverages[i]:
                cover_count[word_to_idx[covered_word]] += 1
        
        # Now greedily remove codewords
        removed = 0
        for idx in indices:
            if not active[idx]:
                continue
            if time.time() > deadline:
                break
            
            # Check if removing this codeword still maintains full coverage
            # i.e., for each word in its sphere, cover_count[word] > 1
            redundant = True
            for covered_word in coverages[idx]:
                if cover_count[word_to_idx[covered_word]] <= 1:
                    redundant = False
                    break
            
            if redundant:
                active[idx] = False
                removed += 1
                for covered_word in coverages[idx]:
                    cover_count[word_to_idx[covered_word]] -= 1
        
        # Build the final code
        final_code = [code[i] for i in range(len(code)) if active[i]]
        
        return final_code
    
    else:
        # General case: use the tetracode-based construction
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Fallback: return all words (valid but not minimal)
            return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
