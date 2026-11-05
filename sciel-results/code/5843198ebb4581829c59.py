import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Parameters specific to the task
    # q=3, n=7, radius=1
    
    total_words = q ** n
    
    # Precompute neighbors for all words
    # neighbors[i] is a list of words at Hamming distance <= radius from i
    neighbors = [[] for _ in range(total_words)]
    
    # Convert word index to tuple of digits
    def index_to_word(idx):
        return [idx // (q ** (n - i - 1)) % q for i in range(n)]
    
    def word_to_index(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precomputation
    for i in range(total_words):
        w = index_to_word(i)
        # The word itself
        neighbors[i].append(i)
        # Neighbors at distance 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    w[pos] = new_val
                    neighbors[i].append(word_to_index(w))
            w[pos] = orig_val
            
    # Initial code: Direct sum of tetracode
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            initial_code.append(word_to_index(word))
            
    # Convert to list of indices for fast processing
    # initial_code is a list of indices (0..2186)
    
    def greedy_reduce(code_indices, rng):
        # code_indices: list of int indices
        # Returns a list of int indices that is a valid covering code
        # and is reduced as much as possible via greedy deletion.
        
        # Use a set for O(1) lookup? No, list is fine for 243 items.
        # But we need to track coverage counts.
        
        coverage_count = np.zeros(total_words, dtype=np.int32)
        
        # Initialize coverage
        for c in code_indices:
            for w in neighbors[c]:
                coverage_count[w] += 1
                
        current_code = list(code_indices)
        
        while True:
            # Find redundant codewords
            # A codeword c is redundant if for all w in neighbors[c], coverage_count[w] > 1
            # i.e., no word is uniquely covered by c.
            
            redundant_indices = []
            
            for idx, c in enumerate(current_code):
                is_redundant = True
                for w in neighbors[c]:
                    if coverage_count[w] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_indices.append(idx)
                    
            if not redundant_indices:
                break
                
            # Pick a random redundant codeword to remove
            # To improve search, we might prefer removing one that covers more "shared" words?
            # For now, random selection.
            rem_idx = redundant_indices[rng.randint(0, len(redundant_indices) - 1)]
            c_to_remove = current_code[rem_idx]
            
            # Remove it
            current_code.pop(rem_idx)
            
            # Update coverage counts
            for w in neighbors[c_to_remove]:
                coverage_count[w] -= 1
                
        return current_code

    # Main search loop
    deadline = time.time() + 110  # 10s safety margin
    
    best_code = initial_code
    best_len = len(initial_code)
    
    rng = random.Random(42)
    
    # First, try pure greedy on the initial code with different random seeds/orders
    # Since greedy_reduce uses random selection, calling it multiple times with different rng states
    # explores different paths.
    
    while time.time() < deadline:
        # Generate a perturbed version of the initial code?
        # Or just run greedy on the initial code?
        # Running greedy on the initial code repeatedly with random tie-breaking is a good start.
        
        # Let's also try perturbations of the current best code if we are stuck.
        # For simplicity, let's first just run greedy on the initial code many times.
        
        code_to_reduce = list(initial_code)
        # Shuffle to ensure different random paths if the implementation depended on order
        # But our implementation picks randomly from redundant set, so order of input list
        # doesn't strictly matter for the *set* of redundant candidates, but the *sequence*
        # of removals depends on the random choices.
        
        reduced = greedy_reduce(code_to_reduce, rng)
        
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_code = reduced
            # If we find a very good one, we can try to perturb it further
            # But for now, just keep searching.
            
    # Convert best_code indices back to lists of integers
    final_codewords = []
    for idx in best_code:
        final_codewords.append(index_to_word(idx))
        
    return final_codewords
