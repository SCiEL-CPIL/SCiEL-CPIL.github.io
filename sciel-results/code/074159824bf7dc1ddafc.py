"""Ternary covering code via bitset-accelerated greedy set cover and systematic local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    
    Strategy:
    1. Precompute Hamming balls as boolean masks.
    2. Use a greedy set-cover algorithm to find an initial covering code.
    3. Aggressively reduce the code size by iteratively removing redundant codewords.
    4. Perform a local search: try swapping each codeword with a random alternative,
       followed by a reduction pass, to find a smaller valid code.
    5. Return the best valid code found.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    N = q ** n
    if N <= 0:
        return []
    
    # Precompute all words as integer indices 0..N-1
    all_words = np.zeros((N, n), dtype=np.int32)
    for idx in range(N):
        rem = idx
        for j in range(n - 1, -1, -1):
            all_words[idx, j] = rem % q
            rem //= q
    
    # Compute Hamming distance matrix
    hamming = np.zeros((N, N), dtype=np.int32)
    for k in range(n):
        col = all_words[:, k].astype(np.int32)
        diff = (col[:, None] != col[None, :]).astype(np.int32)
        hamming += diff
    
    # Ball mask: ball_mask[c] is a boolean array of size N where True indicates
    # that word w is within Hamming distance `radius` of codeword c.
    ball_mask = hamming <= radius
    del hamming
    
    # Helper: Compute coverage of a set of codewords
    def get_coverage(code_set):
        """Return a boolean array indicating which words are covered."""
        if not code_set:
            return np.zeros(N, dtype=bool)
        # Use OR reduction for efficiency
        # Since ball_mask is a 2D array, we can index it directly
        return np.any(ball_mask[list(code_set)], axis=0)
    
    # Helper: Check if a set of codewords covers everything
    def is_valid(code_set):
        covered = get_coverage(code_set)
        return np.all(covered)
    
    # Helper: Reduce a valid code set by removing redundant codewords
    def reduce_code(code_set):
        """Iteratively remove any codeword that is redundant."""
        code_list = list(code_set)
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                if len(code_list) <= 1:
                    break
                # Try removing code_list[i]
                candidate = code_list[:i] + code_list[i+1:]
                if is_valid(candidate):
                    code_list = candidate
                    changed = True
        return code_list
    
    # Phase 1: Greedy Set Cover
    # Start with no codewords and all words uncovered
    uncovered = np.ones(N, dtype=bool)
    code_set = []
    
    while np.any(uncovered):
        # Find the codeword that covers the most uncovered words
        # new_cov[c] = number of uncovered words covered by codeword c
        new_cov = np.sum(ball_mask & uncovered[None, :], axis=1)
        best_idx = np.argmax(new_cov)
        
        if new_cov[best_idx] == 0:
            break  # Should not happen if a covering exists
        
        code_set.append(int(best_idx))
        # Update uncovered mask
        uncovered[ball_mask[best_idx]] = False
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Phase 2: Initial Reduction
    if is_valid(code_set):
        code_set = reduce_code(code_set)
    
    best_code = code_set.copy()
    best_size = len(best_code)
    
    # Phase 3: Local Search (Swap and Reduce)
    # We will try to improve the code by swapping one codeword with another
    # and then reducing.
    
    # Precompute a random permutation of all words for candidate selection
    candidates = np.random.permutation(N)
    cand_idx = 0
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try swapping each codeword in the current best code
        for i in range(len(best_code)):
            if time.time() > deadline - 5:
                break
            
            # Try a few random candidates to replace best_code[i]
            for _ in range(5):
                cand_idx = (cand_idx + 1) % N
                new_cand = candidates[cand_idx]
                
                if new_cand == best_code[i]:
                    continue
                
                # Check if new_cand is already in the set
                if new_cand in best_code:
                    continue
                
                # Create a new set with the swap
                new_set = best_code.copy()
                new_set[i] = new_cand
                
                # Check validity
                if not is_valid(new_set):
                    continue
                
                # Reduce the new set
                new_set = reduce_code(new_set)
                
                # Check if we improved
                if len(new_set) < len(best_code):
                    best_code = new_set
                    best_size = len(best_code)
                    improved = True
                    break # Restart the loop over indices since the set changed
        
        if not improved:
            # No improvement found in this full pass, we could stop or continue
            # To be safe, let's do a few more passes with different random seeds
            # But if we haven't improved in a full pass, it's likely a local minimum.
            # We'll break to save time for final checks.
            break

    # Final validity check
    if not is_valid(best_code):
        # Fallback to greedy if something went wrong
        uncovered = np.ones(N, dtype=bool)
        code_set = []
        while np.any(uncovered) and time.time() < deadline - 1:
            new_cov = np.sum(ball_mask & uncovered[None, :], axis=1)
            best_idx = np.argmax(new_cov)
            if new_cov[best_idx] == 0:
                break
            code_set.append(int(best_idx))
            uncovered[ball_mask[best_idx]] = False
        best_code = reduce_code(code_set)
    
    return [list(all_words[i]) for i in best_code]
# EVOLVE-BLOCK-END
