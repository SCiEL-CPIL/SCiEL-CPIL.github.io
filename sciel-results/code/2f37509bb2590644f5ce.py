import numpy as np
import time
import itertools

def construct(q, n, radius):
    """
    Constructs a q-ary code of length n with covering radius radius.
    Minimizes the number of codewords.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Pre-compute all possible words in the space
    # Total words = q^n. For q=3, n=7, this is 2187.
    # We represent each word as an integer index in [0, q^n)
    total_words = q ** n
    
    # Pre-compute the list of words as indices
    # word_idx -> tuple of digits
    # For efficiency in distance calculation, we might precompute neighbors
    
    # Neighbors of each word within radius
    # Since radius is small (1), we can precompute the list of neighbors for each word.
    # However, 2187 * (1 + n*(q-1)) is small.
    # Let's precompute the 'cover' sets: for each codeword candidate c, which words does it cover?
    # Actually, it's easier to think: for each word w, which codewords cover it?
    # Or: for each codeword c, which words does it cover?
    
    # Let's use the perspective: 
    # We have a set of all words U.
    # We want to select a subset C of U such that union_{c in C} N(c) = U.
    # N(c) is the set of words within distance radius of c.
    
    # Precompute N(c) for all c in U.
    # U is indexed 0..total_words-1.
    # To compute N(c), we generate all neighbors of c.
    
    # Helper: convert index to tuple
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[i] = idx % q
            idx //= q
        return tuple(t)
        
    # Helper: convert tuple to index
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors for all indices
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius
    neighbors = [[] for _ in range(total_words)]
    
    # To speed up neighbor generation, we can precompute shifts.
    # But for n=7, q=3, brute force neighbor generation is fast enough if done once.
    # Or better: precompute for each position i, and each digit d, the list of indices that differ at i with digit d?
    # No, let's just generate neighbors for each center.
    
    # Optimization: Precompute the base digits for all indices?
    # digits = np.array([list(idx_to_tuple(i)) for i in range(total_words)])
    # This is 2187x7, very small.
    
    all_digits = np.zeros((total_words, n), dtype=np.uint8)
    for i in range(total_words):
        idx = i
        for j in range(n):
            all_digits[i, j] = idx % q
            idx //= q
            
    # Precompute neighbors using vectorized ops?
    # For each center i, neighbors are i itself + all words at distance 1.
    # Distance 1 means changing one coordinate.
    
    # Let's build neighbors[i]
    for i in range(total_words):
        # i itself
        neighbors[i].append(i)
        # Change one coordinate
        for pos in range(n):
            original_val = all_digits[i, pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                # Create new index
                # We can precompute the 'step' for changing coordinate pos from v to v'
                # step = (new_val - original_val) * (q ** pos)
                # new_idx = i + step
                # But we must handle wrap-around? No, index arithmetic works if we treat it as base-q.
                # i = sum d_k q^k.
                # Changing d_pos to new_val:
                # new_i = i - original_val * q^pos + new_val * q^pos
                # new_i = i + (new_val - original_val) * q^pos
                step = (new_val - original_val) * (q ** pos)
                new_idx = i + step
                if 0 <= new_idx < total_words:
                    neighbors[i].append(new_idx)
                    
    # Now we have neighbors[i] for all i.
    # We want to select a subset of indices C such that every index j is in neighbors[c] for some c in C.
    # This is equivalent to: for every j, C intersects neighbors[j]? No.
    # j is covered by c if j in neighbors[c].
    # So we need: for all j, exists c in C such that j in neighbors[c].
    
    # Greedy Set Cover:
    # Uncovered set U starts as all indices.
    # At each step, pick c that maximizes |neighbors[c] \intersect U|.
    # Add c to C.
    # Remove neighbors[c] from U.
    
    # To make it fast:
    # Maintain an array 'count' where count[j] is the number of selected codewords that cover j.
    # Initially 0.
    # When we select c, we increment count[j] for all j in neighbors[c].
    # We want to pick c that maximizes the number of j in neighbors[c] where count[j] == 0.
    
    # To speed up selection, we can maintain for each candidate c, a score: 
    # score[c] = number of neighbors of c that are currently uncovered.
    # When we select c*, we update scores for all other candidates c.
    # For each neighbor j of c*, if count[j] becomes 1 (was 0), then for all c such that j in neighbors[c], 
    # score[c] decreases by 1.
    # This is expensive if done naively.
    
    # Alternative: 
    # Since total_words is small (2187), we can just iterate through all candidates to find the max score each time.
    # Number of iterations ~ 200.
    # Cost per iteration: 2187 candidates * avg neighbors size.
    # Avg neighbors size = 1 + n*(q-1) = 1 + 7*2 = 15.
    # 2187 * 15 = ~33000 ops per greedy step.
    # 200 steps * 33000 = 6.6M ops.
    # This is very fast in Python/NumPy if vectorized, or even pure Python.
    
    def greedy_construct(random_seed=0):
        rng = np.random.default_rng(random_seed)
        
        # Initial state
        # count[j] = 0 for all j
        count = np.zeros(total_words, dtype=np.uint8)
        selected = []
        
        # Precompute neighbor lists as numpy arrays for faster indexing?
        # Or just lists.
        
        # To allow randomization, we can break ties randomly or shuffle candidates.
        # Let's shuffle the order in which we evaluate candidates for ties.
        
        # Main loop
        while True:
            # Check if all covered
            if np.all(count > 0):
                break
                
            # Find best candidate
            # score[c] = sum(1 for j in neighbors[c] if count[j] == 0)
            
            # Vectorized calculation of scores
            # We can't easily vectorize the "sum over neighbors" without a matrix.
            # Let's create a matrix M where M[c, j] = 1 if j in neighbors[c].
            # This is a 2187 x 2187 matrix. 4M entries. Memory is fine.
            # But building it takes time.
            
            # Let's stick to iterating candidates but optimizing the inner loop.
            # Or, precompute the neighbor lists as np arrays.
            
            best_c = -1
            best_score = -1
            # To add randomness, we can pick a random subset of top candidates or just shuffle
            # Let's just find the max. If multiple, pick random one.
            
            # Optimization: 
            # We can maintain a list of 'active' candidates? No, all can be picked.
            
            # Let's try a simpler vectorized approach for score calculation:
            # We have 'count' (1D array).
            # We want score[c] = sum(count[j]==0 for j in neighbors[c]).
            
            # Let's precompute neighbors as a list of np arrays.
            if not hasattr(greedy_construct, 'neighbor_arrays'):
                greedy_construct.neighbor_arrays = [np.array(nbrs) for nbrs in neighbors]
                
            nbr_arrays = greedy_construct.neighbor_arrays
            
            # Calculate scores for all c
            # scores = np.array([np.sum(count[nbr_arrays[c]] == 0) for c in range(total_words)])
            # This is a list comprehension, might be slow.
            
            # Better: 
            # We can't easily vectorize across c because neighbor lists have variable lengths (though here they are fixed length 15).
            # Since length is fixed, we can make a 2D array!
            
            if not hasattr(greedy_construct, 'neighbor_matrix'):
                # All neighbors have same length?
                # Yes, for radius 1 in q-ary n-space, size is 1 + n(q-1).
                nbr_size = len(neighbors[0])
                matrix = np.zeros((total_words, nbr_size), dtype=np.uint16)
                for i in range(total_words):
                    matrix[i, :] = neighbors[i]
                greedy_construct.neighbor_matrix = matrix
                greedy_construct.nbr_size = nbr_size
                
            matrix = greedy_construct.neighbor_matrix
            nbr_size = greedy_construct.nbr_size
            
            # count is shape (total_words,)
            # We need to check if count[j] == 0 for j in matrix[c, :]
            # uncovered_flags = (count[matrix] == 0) # Shape (total_words, nbr_size)
            # scores = uncovered_flags.sum(axis=1)
            
            # This creates a 2187x15 array of booleans. ~33k elements. Very fast.
            uncovered_flags = (count[matrix] == 0)
            scores = uncovered_flags.sum(axis=1)
            
            # Find max score
            max_score = np.max(scores)
            if max_score == 0:
                # Should not happen if not all covered
                break
                
            # Get indices with max score
            candidates = np.where(scores == max_score)[0]
            
            # Pick one randomly
            c = candidates[rng.integers(0, len(candidates))]
            
            selected.append(c)
            
            # Update counts
            # For all j in neighbors[c], increment count[j]
            nbrs_c = matrix[c, :]
            count[nbrs_c] += 1
            
        return selected

    def is_covered(selected, count):
        # Re-verify coverage to be safe
        # count should be > 0 everywhere
        return np.all(count > 0)

    def reduce_code(selected):
        # Try to remove codewords
        # Rebuild count from scratch
        count = np.zeros(total_words, dtype=np.uint8)
        
        # Reuse neighbor matrix
        if not hasattr(reduce_code, 'neighbor_matrix'):
            reduce_code.neighbor_matrix = greedy_construct.neighbor_matrix
        matrix = reduce_code.neighbor_matrix
            
        for c in selected:
            count[matrix[c, :]] += 1
            
        # Try to remove each codeword
        # If removing c makes any count[j] drop to 0, c is essential.
        # We can do this iteratively.
        
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                c = selected[i]
                # Temporarily remove
                count[matrix[c, :]] -= 1
                if np.all(count > 0):
                    # Safe to remove
                    selected.pop(i)
                    changed = True
                else:
                    # Put back
                    count[matrix[c, :]] += 1
                    
        return selected, count

    best_code = None
    best_len = float('inf')
    
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        # Random seed for each restart
        code = greedy_construct(restarts)
        
        # Reduce
        code, count = reduce_code(code)
        
        # Verify
        if not is_covered(code, count):
            # Should not happen, but just in case
            continue
            
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
            
        restarts += 1
        
        # If we hit a very good score, we might want to polish more
        # But for now, just keep restarting.
        
        # Optional: Local search / Hill climbing on the reduced code?
        # Swap two codewords?
        # Given time constraints, simple restarts are effective.
        
        # Check time again
        if time.time() > deadline - 5:
            break

    # If no code found (should not happen), fallback to seed
    if best_code is None:
        # Fallback to deterministic seed logic
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert indices back to lists of integers
    result = []
    for idx in best_code:
        result.append(list(idx_to_tuple(idx)))
        
    return result
