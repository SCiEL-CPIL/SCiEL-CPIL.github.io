import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Returns a list of codewords, each a list of n integers in 0..q-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Handle trivial cases or small instances
    if n == 0:
        return [[]]
    if radius >= n:
        # One codeword covers everything if radius >= n? 
        # Actually, max distance is n. So radius >= n means 1 codeword is enough.
        return [[0] * n]

    total_words = q ** n
    
    # If the instance is small, we can use a general solver.
    # For q=3, n=6, total_words = 729.
    
    def to_int(word):
        """Convert list of digits to integer index."""
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    def to_word(idx):
        """Convert integer index to list of digits."""
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbors for each word
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius
    # For efficiency, we compute this on the fly or precompute.
    # Precomputing for 729 words is fast.
    
    def hamming_dist_int(a, b):
        # Calculate distance between two integer representations
        # This is slow if done naively.
        # Better to precompute neighbor lists.
        pass

    # Precompute neighbor lists
    # This takes O(N * q^n * n) which is fine for N=729.
    # Actually, generating neighbors:
    # For each word w, generate all words within radius r.
    
    all_words_indices = list(range(total_words))
    neighbors = [[] for _ in range(total_words)]
    
    # Efficient neighbor generation
    # For each index, convert to word, generate perturbations
    for idx in all_words_indices:
        w = to_word(idx)
        neighbors[idx].append(idx) # distance 0
        
        if radius >= 1:
            # distance 1
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val != orig:
                        w[pos] = val
                        neighbors[idx].append(to_int(w))
                w[pos] = orig
        
        if radius >= 2:
             # distance 2
             # This becomes expensive if we do it naively for large n.
             # But for n=6, it's small.
             # However, a simpler way for small N is to just compute distances between all pairs?
             # N=729, N^2 = 531k pairs. Distance calc is O(n)=6.
             # Total ops ~ 3 million. Very fast.
             pass

    # Let's just compute distance matrix or neighbor lists via pairwise comparison
    # Since N is small, let's do pairwise distance for simplicity and correctness.
    # N=729 is small.
    
    dist_matrix = np.zeros((total_words, total_words), dtype=np.int8)
    words_list = [to_word(i) for i in range(total_words)]
    
    for i in range(total_words):
        wi = words_list[i]
        for j in range(i, total_words):
            wj = words_list[j]
            d = 0
            for k in range(n):
                if wi[k] != wj[k]:
                    d += 1
                    if d > radius:
                        break
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Build neighbor lists from dist_matrix
    for i in range(total_words):
        neighbors[i] = [j for j in range(total_words) if dist_matrix[i, j] <= radius]

    best_code = None
    best_size = float('inf')
    
    num_restarts = 200 # Adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Greedy Set Cover
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        # To speed up greedy, we can maintain a count of uncovered neighbors for each candidate
        # But simple greedy is:
        # For each step, find c that maximizes sum(1 for j in neighbors[c] if not covered[j])
        
        # Optimization: 
        # Instead of scanning all 729 candidates every time, 
        # we can just do it. 729 * 73 steps * 13 neighbors/step ~ 700k ops. Fast.
        
        while not covered.all():
            best_c = -1
            best_gain = -1
            
            for c in range(total_words):
                # Skip if c is already in code? 
                # No, we can pick the same c again? No, distinct codewords.
                # We should maintain a set of chosen codewords to avoid duplicates.
                # But greedy usually picks distinct ones.
                
                # Calculate gain
                gain = 0
                for j in neighbors[c]:
                    if not covered[j]:
                        gain += 1
                        if gain > best_gain:
                            # Early exit if we already beat best? 
                            # No, we need exact gain to compare with other c's later?
                            # Actually, if we just want the max, we can break inner loop if gain > best_gain?
                            # No, we need the full gain to compare with other candidates in the outer loop.
                            # But if gain is already > best_gain, we can break the inner loop for this c?
                            # No, we need to know if another c has an even higher gain.
                            # But for the current c, if gain > best_gain, we can stop counting for this c?
                            # No, we just need to know if it's the best.
                            # Actually, we can just compute the full gain. It's small.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    
            if best_c == -1:
                break
                
            code.append(best_c)
            for j in neighbors[best_c]:
                covered[j] = True
        
        # Pruning Phase
        # Remove redundant codewords
        # We need to check if removing c leaves any neighbor uncovered.
        # This requires knowing if a neighbor is covered by OTHER codewords.
        
        # Efficient Pruning:
        # Maintain cover_count[j] = number of codewords in code that cover j.
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            for j in neighbors[c]:
                cover_count[j] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if any neighbor of c has cover_count == 1
                # If yes, c is essential.
                # If no, c is redundant.
                is_redundant = True
                for j in neighbors[c]:
                    if cover_count[j] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove c
                    for j in neighbors[c]:
                        cover_count[j] -= 1
                    code.pop(i)
                    changed = True
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
            # If we reached the theoretical lower bound or a known good value, we might stop?
            # Lower bound for K_3(6,1) is 71.
            # If we hit 71, it's optimal.
            if best_size <= 71:
                break

    # Convert best_code indices to words
    if best_code is None:
        # Fallback to seed construction if something went wrong
        # Tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [to_word(c) for c in best_code]
