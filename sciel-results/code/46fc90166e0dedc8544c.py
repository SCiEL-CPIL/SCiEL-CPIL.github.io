import itertools
import random
import time

def construct(q, n, radius):
    # Parameters
    N = q ** n
    # Map word to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
        
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighborhoods for all words in the space?
    # No, we only need neighborhoods for candidates.
    # But we need to know which indices are covered.
    
    # Generate candidates
    # 1. Direct sum construction
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    candidates = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            candidates.append(h + list(tail))
            
    # 2. Add random candidates to help greedy escape local optima
    # The space is small, so we can add many randoms
    num_randoms = 2000
    for _ in range(num_randoms):
        candidates.append([random.randint(0, q - 1) for _ in range(n)])
    
    # Remove duplicates
    candidates = list(set([tuple(c) for c in candidates]))
    
    # Precompute neighborhood indices for each candidate
    # neighborhood[candidate_idx] = list of word indices covered
    # We need to map candidate tuple to its index in the list, or just store the list directly.
    # Let's store a list of lists: cov_lists[i] is the list of word indices covered by candidates[i]
    
    # Efficient neighborhood generation
    def get_neighbors(word):
        neighbors = [word_to_idx(word)]
        for i in range(n):
            orig = word[i]
            for v in range(q):
                if v != orig:
                    word[i] = v
                    neighbors.append(word_to_idx(word))
            word[i] = orig
        return neighbors

    # Precompute coverage for all candidates
    cov_lists = []
    for c in candidates:
        cov_lists.append(get_neighbors(list(c)))
        
    # Greedy Set Cover
    covered = [False] * N
    selected_indices = []
    
    # To speed up, we can maintain a count of uncovered words per candidate
    # But updating it is O(N) per selection.
    # Given N=2187 and ~200 selections, O(200 * 200 * 1200) is fine.
    
    deadline = time.time() + 110  # Safety margin
    
    while True:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_count = -1
        
        for i in range(len(candidates)):
            count = 0
            for idx in cov_lists[i]:
                if not covered[idx]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
                
        if best_count == 0:
            # All covered or no candidates left
            break
            
        # Select best
        selected_indices.append(best_idx)
        for idx in cov_lists[best_idx]:
            covered[idx] = True
            
        # Check if all covered
        # Optimization: keep track of uncovered count
        # If uncovered_count == 0, break
        
    # Convert indices back to words
    result = [list(candidates[i]) for i in selected_indices]
    
    # Optional: Local search / Polishing
    # Try to remove redundant codewords? Greedy set cover usually produces minimal sets, 
    # but sometimes a different set of same size or smaller can be found by swapping.
    # Given time constraints, a simple "try to remove" pass is safe.
    # If removing a codeword leaves all words covered, remove it.
    
    # Let's verify coverage and try to shrink
    # Re-calculate covered set for the result
    final_covered = [False] * N
    for c in result:
        for idx in get_neighbors(c):
            final_covered[idx] = True
            
    # Try to remove each codeword
    # This is O(K * N) where K is number of codewords
    # If K is ~200, 200 * 2187 * 29 is too slow for multiple passes.
    # We'll skip extensive polishing to ensure we return within time.
    
    return result
