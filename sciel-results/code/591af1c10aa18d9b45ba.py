"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 17
MIN_D = 6

def _hamming(a, b):
    return (a ^ b).bit_count()

def _construct_seed():
    """
    Construct a good initial code.
    Strategy: Start with the [17, 8, 7] shortened Hamming code? No, that's 256 words?
    Wait, the prompt says: "The 256-word linear reference is the [18,9,6] extended quadratic-residue code shortened by one coordinate; the seed has 32 words."
    Actually, the seed in the current program is the First-order Reed-Muller code RM(1,4) extended?
    The current program returns 32 words.
    A_2(17, 6) is between 258 and 340.
    We can likely find a code with > 32 words easily.
    Even a greedy randomized approach might find ~100+ words quickly.
    Let's use a greedy algorithm with a good random initialization or just pure greedy.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # 1. Generate a strong initial seed
    # The prompt mentions a 256-word linear code exists.
    # The [17, 8, 7] shortened Hamming code?
    # Hamming code [15, 11, 3]. Extended [16, 11, 4].
    # Let's look for a known code structure or just run a fast greedy.
    
    # Greedy approach is O(M^2 * N) if naive, but with bit operations it's fast.
    # We can try to build a large code greedily.
    
    # To beat the seed (32 words), we just need to find a valid set of > 32 words.
    # A random greedy construction often finds ~50-100 words for N=17, d=6.
    
    # Let's implement a fast greedy with restarts.
    # Representation: list of integers.
    
    best_code = []
    best_size = 0
    
    # Try to construct a code using a greedy method with multiple random orderings.
    # Or, use the fact that the [17, 8, 7] code (if it exists) has 256 words?
    # Actually, A_2(17, 6) >= 258.
    # There is a known code: The shortened [18, 9, 6] quadratic residue code has 2^9 = 512 words?
    # Wait, [18, 9, 6] has 512 words. Shortened to 17, it has 512 words?
    # If we can construct the [17, 9, 5] or [17, 8, 6] code, we get a large score.
    
    # Let's try to construct the quadratic residue code or a similar structured code.
    # Quadratic residues mod 19: 1, 4, 16, 6, 7, 11, 9, 5, 17.
    # The [18, 9, 6] code is related to mod 19.
    # It's easier to just run a very fast greedy algorithm that might stumble upon a large code,
    # or use a local search (flipping bits) to improve a random initial code.
    
    # Given the 80s budget, we can run a local search.
    # Start with a small random code, then add words greedily, then try to improve by flipping bits.
    
    # However, the most reliable way to get a high score quickly is to find a known large code.
    # Let's try to generate the [17, 8, 7] shortened Hamming code?
    # No, let's just use a robust randomized greedy + local improvement.
    
    # Step 1: Greedy construction
    def greedy_construct(num_trials=10, time_limit=10.0):
        best = []
        for trial in range(num_trials):
            if time.time() > start_time + time_limit:
                break
            # Random permutation of all 2^17 words? Too many.
            # Instead, start with zero word, then add random words that are far enough.
            code = [0]
            # To make it faster, we can pre-generate a pool of candidate words.
            # Or just pick random words and check against current code.
            
            # A better greedy: 
            # 1. Start with 0.
            # 2. Repeatedly pick a random word w.
            # 3. If min distance from w to code >= 6, add it.
            # 4. Stop when no more can be added or time limit.
            
            # This is slow if we pick bad words.
            # Better: Start with a set of "core" words, then fill.
            
            # Let's try a different strategy: 
            # Use the linear code structure if possible.
            # The [17, 8, 6] code? 
            # If we can find a 8x17 parity check matrix or generator matrix for a code with d>=6, 
            # we get 256 words.
            
            # Let's try to construct a linear code.
            # Random linear codes of dimension 8 in length 17 with distance >= 6.
            # Probability that a random [17, 8] code has d>=6 is non-zero?
            # Griesmer bound: For d=6, k=8, n >= 6+5+4+3+2+1+... 
            # Actually, the [17, 8, 6] code might not exist. The prompt says the 256-word reference is from [18,9,6] shortened.
            # [18,9,6] has 512 words. Shortened to 17, it has 512 words? 
            # If we shorten a linear code, the dimension stays the same? No, shortening one coordinate reduces length by 1.
            # If we take all words with 0 in the last coordinate, we get a [17, 8, 6] code?
            # If the original code has weight 6 words with a 1 in the last position, those are lost.
            # But we keep the ones with 0.
            # The minimum distance might stay 6 or become 7.
            
            # Let's just try to find ANY code with > 32 words.
            # A simple greedy with a good random seed should find ~50-100 words in seconds.
            
            # Let's implement a fast greedy that picks words from a pre-shuffled list of "good" candidates.
            # What are good candidates? Words with weight around 8 or 9.
            
            candidates = []
            for w in range(2**N):
                wt = w.bit_count()
                if 5 <= wt <= 12:  # likely to be far from 0 and each other
                    candidates.append(w)
            
            random.shuffle(candidates)
            
            code = [0]
            # Add candidates
            for w in candidates:
                if time.time() > start_time + time_limit:
                    break
                ok = True
                for c in code:
                    if _hamming(w, c) < 6:
                        ok = False
                        break
                if ok:
                    code.append(w)
            
            if len(code) > len(best):
                best = code
        
        return best
    
    # Let's run the greedy construction
    current_code = greedy_construct(num_trials=5, time_limit=10.0)
    
    # Now, try to improve via local search (bit flipping)
    # We can try to add more words or replace words to allow more additions.
    # A simple improvement: 
    # 1. Take the current code.
    # 2. Try to add more random words.
    # 3. If stuck, try to flip bits in existing words to create space? (Complex)
    
    # Given the time, even a decent greedy result (e.g., 100 words) is a 3x improvement over 32.
    
    # Let's try a more aggressive greedy with multiple random restarts and longer time.
    # We have 80s.
    # Let's use the remaining time for more trials.
    
    end_time = start_time + 70.0
    while time.time() < end_time:
        # Generate a new random candidate list
        candidates = []
        for w in range(2**N):
            wt = w.bit_count()
            if 4 <= wt <= 13:
                candidates.append(w)
        
        random.shuffle(candidates)
        
        code = [0]
        for w in candidates:
            if time.time() > end_time:
                break
            ok = True
            for c in code:
                if _hamming(w, c) < 6:
                    ok = False
                    break
            if ok:
                code.append(w)
        
        if len(code) > len(current_code):
            current_code = code
    
    return current_code

# EVOLVE-BLOCK-END
