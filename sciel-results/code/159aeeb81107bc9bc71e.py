"""Optimized ternary covering code K_3(8,1) via peeling heuristic."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import Counter

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius r.
    Uses a peeling heuristic: start with a known good code (direct sum of tetracode),
    then iteratively remove codewords whose removal doesn't break coverage.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: simple greedy or direct sum
        if n <= 4:
            # For small n, we can be more aggressive
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            elif n == 3:
                return [[a, b, (a + b) % 3] for a in range(3) for b in range(3)]
            elif n == 2:
                return [[a, b] for a in range(3) for b in range(3)]
            elif n == 1:
                return [[a] for a in range(3)]
        
        # General fallback: direct sum approach
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n > 4:
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n-4)]
        else:
            return tetracode[:3**n]
    
    # Specialized for q=3, n=8, radius=1
    # Start with the direct sum code: 9 * 3^4 = 729 codewords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    import itertools
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    code = [tuple(c) for c in code]
    
    # Convert to numpy array for efficient operations
    code_arr = np.array(code, dtype=np.int8)
    
    # Precompute all neighbors (distance <= 1) for each codeword
    # For each codeword, its neighborhood includes itself and all words at Hamming distance 1
    # Total neighbors per codeword: 1 + 8*2 = 17
    
    # We'll track coverage counts: for each word in the space, how many codewords cover it
    # Space size: 3^8 = 6561
    space_size = 3 ** n
    coverage = np.zeros(space_size, dtype=np.int32)
    
    # Helper: convert a word (tuple of 8 digits) to an index in [0, 6561)
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx
    
    # Helper: generate all neighbors (distance <= 1) of a word
    def get_neighbors(word):
        neighbors = [word]
        for pos in range(n):
            for delta in [1, 2]:
                new_word = list(word)
                new_word[pos] = (word[pos] + delta) % 3
                neighbors.append(tuple(new_word))
        return neighbors
    
    # Initialize coverage
    for c in code:
        for neighbor in get_neighbors(c):
            coverage[word_to_idx(neighbor)] += 1
    
    # Peeling: iteratively remove codewords if possible
    # A codeword can be removed if all its neighbors are covered by at least 2 codewords
    # (i.e., removing it leaves coverage >= 1 for all its neighbors)
    
    removed = set()
    current_code = list(code)
    
    # To make this efficient, we'll use a priority queue or just iterate
    # For each codeword, check if it's "redundant": all its neighbors have coverage >= 2
    
    def is_redundant(c, coverage, removed_set):
        """Check if codeword c can be removed without breaking coverage."""
        for neighbor in get_neighbors(c):
            idx = word_to_idx(neighbor)
            if coverage[idx] <= 1:
                return False
        return True
    
    # Iterative peeling
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i, c in enumerate(current_code):
            if time.time() > deadline:
                break
            if is_redundant(c, coverage, removed):
                # Remove this codeword
                for neighbor in get_neighbors(c):
                    idx = word_to_idx(neighbor)
                    coverage[idx] -= 1
                current_code.pop(i)
                removed.add(c)
                changed = True
                break  # Restart iteration after removal
    
    # Try multiple passes with different orderings to find better solutions
    # Also try random restarts within the time budget
    
    best_code = current_code
    best_size = len(current_code)
    
    # Try additional peeling passes with shuffled order
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        import random
        random.seed(42)
        
        # Try a few random restarts with different initial orderings
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Start from the best code found so far
            trial_code = list(best_code)
            trial_coverage = np.zeros(space_size, dtype=np.int32)
            
            for c in trial_code:
                for neighbor in get_neighbors(c):
                    trial_coverage[word_to_idx(neighbor)] += 1
            
            # Shuffle and peel
            random.shuffle(trial_code)
            
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for i, c in enumerate(trial_code):
                    if time.time() > deadline - 5:
                        break
                    if is_redundant(c, trial_coverage, set()):
                        for neighbor in get_neighbors(c):
                            idx = word_to_idx(neighbor)
                            trial_coverage[idx] -= 1
                        trial_code.pop(i)
                        changed = True
                        break
            
            if len(trial_code) < best_size:
                best_code = trial_code
                best_size = len(trial_code)
    
    # Final verification and return
    # Convert back to list of lists
    result = [list(c) for c in best_code]
    
    # Quick validity check (optional, can be skipped for speed)
    if time.time() < deadline - 2:
        # Verify coverage
        final_coverage = np.zeros(space_size, dtype=np.int32)
        for c in best_code:
            for neighbor in get_neighbors(c):
                final_coverage[word_to_idx(neighbor)] += 1
        
        # Check if all words are covered
        if np.any(final_coverage == 0):
            # Fall back to the original code if something went wrong
            result = [list(c) for c in code]
    
    return result
# EVOLVE-BLOCK-END
