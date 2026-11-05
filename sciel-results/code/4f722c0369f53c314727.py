import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code using a greedy heuristic.
    
    Args:
        q: Alphabet size
        n: Length of codewords
        radius: Covering radius
        
    Returns:
        A list of codewords (lists of integers)
    """
    total_words = q ** n
    
    # Map words to integers 0..q^n-1
    # We will work with integer representations for speed
    # Precompute spheres for all words
    # sphere[i] is a list of integers j such that dist(i, j) <= radius
    
    # Helper to convert integer to word (list of digits)
    def int_to_word(x):
        word = [0] * n
        for i in range(n):
            word[i] = x % q
            x //= q
        return word

    # Helper to convert word to integer
    def word_to_int(word):
        x = 0
        for i in range(n):
            x = x * q + word[i]
        return x

    # Precompute Hamming distances or just generate spheres directly
    # For radius 1, sphere is {i} U {i +/- 1 at each position}
    # For general radius, we can generate neighbors.
    # Given small q^n, we can precompute all spheres.
    
    # To precompute spheres efficiently:
    # 1. Generate all words as integers.
    # 2. For each word, generate all neighbors within radius.
    
    # Let's use a more direct approach for sphere generation
    # We'll store spheres as lists of integers.
    
    spheres = []
    for i in range(total_words):
        w = int_to_word(i)
        neighbors = set()
        neighbors.add(i)
        
        # Generate neighbors within Hamming distance 'radius'
        # This can be done by changing up to 'radius' positions
        # For small radius and n, this is fine.
        # For radius=1:
        if radius == 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w[pos] = new_val
                        neighbors.add(word_to_int(w))
                w[pos] = orig
        else:
            # General case: iterate over all words and check distance?
            # Or use combinatorial generation.
            # For the specific task q=3, n=8, radius=1, the above is fast.
            # If radius > 1, we need a more robust neighbor generator.
            # Let's implement a general one using itertools.combinations for positions
            import itertools
            # Change 1 position
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w[pos] = new_val
                        neighbors.add(word_to_int(w))
                w[pos] = orig
            
            if radius >= 2:
                # Change 2 positions
                for pos1, pos2 in itertools.combinations(range(n), 2):
                    orig1 = w[pos1]
                    orig2 = w[pos2]
                    for v1 in range(q):
                        if v1 == orig1: continue
                        w[pos1] = v1
                        for v2 in range(q):
                            if v2 == orig2: continue
                            w[pos2] = v2
                            neighbors.add(word_to_int(w))
                    w[pos1] = orig1
                    w[pos2] = orig2
            # Note: For this specific task, radius is 1. 
            # The general case implementation above is sufficient for small n, radius.
            
        spheres.append(list(neighbors))

    # Greedy Covering Algorithm
    # uncovered[i] is 1 if word i is not yet covered, 0 otherwise
    uncovered = np.ones(total_words, dtype=np.uint8)
    remaining = total_words
    code = []
    
    # For efficiency, we can precompute the list of candidates
    candidates = list(range(total_words))
    
    while remaining > 0:
        best_c = -1
        best_score = -1
        
        # Find the word that covers the most uncovered words
        for c in candidates:
            # Calculate score: number of uncovered words in sphere[c]
            # Using numpy for vectorized sum might be faster if spheres were arrays
            # But spheres are lists of indices.
            # Let's just sum manually, it's fast enough for 6561 * 17 * ~500 steps
            score = 0
            for u in spheres[c]:
                if uncovered[u]:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_c = c
        
        if best_c == -1:
            break # Should not happen if remaining > 0
            
        # Add best_c to code
        code.append(best_c)
        
        # Mark covered words
        for u in spheres[best_c]:
            if uncovered[u]:
                uncovered[u] = 0
                remaining -= 1
                
        # Optional optimization: Remove candidates that are now "dominated" or have 0 score?
        # If a candidate has 0 uncovered neighbors, it's useless.
        # We can filter candidates, but checking if score is 0 requires another pass.
        # Given the small size, full scan is fine.
        
    # Convert integer codes back to lists of integers
    result = [int_to_word(c) for c in code]
    return result
