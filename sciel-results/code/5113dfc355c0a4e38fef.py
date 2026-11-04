"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # For the specific case q=3, n=7, radius=1, we can try to improve with local search
    if q == 3 and n == 7 and radius == 1:
        return _local_search_ternary_7_1(initial_code)
    
    return initial_code


def _local_search_ternary_7_1(initial_code):
    """Local search to reduce the number of codewords for K_3(7,1)."""
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    # Convert to numpy for efficiency
    code = np.array(initial_code, dtype=np.int8)
    n = 7
    q = 3
    
    # Precompute all words in the space
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total_words = len(all_words)
    
    # Precompute balls for each codeword position
    # A ball of radius 1 around a word w contains w itself and all words differing in exactly one position
    # For a word w, the ball has 1 + n*(q-1) = 1 + 7*2 = 15 words
    
    def compute_ball(w):
        """Return set of indices of words within Hamming distance 1 of w."""
        ball = {tuple(w)}
        for i in range(n):
            for v in range(q):
                if v != w[i]:
                    w2 = list(w)
                    w2[i] = v
                    ball.add(tuple(w2))
        return ball
    
    # Index all words
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Compute coverage: for each word in the space, how many codewords cover it?
    # We'll track coverage count for each word
    coverage = np.zeros(total_words, dtype=np.int32)
    
    # Precompute balls for initial codewords
    codeword_balls = []
    for cw in code:
        ball = compute_ball(cw)
        ball_indices = [word_to_idx[t] for t in ball if t in word_to_idx]
        codeword_balls.append(ball_indices)
        for idx in ball_indices:
            coverage[idx] += 1
    
    # Verify initial coverage is complete
    assert np.all(coverage > 0), "Initial code does not cover all words"
    
    best_code = code.copy()
    best_size = len(code)
    
    # Local search: try removing codewords that are redundant (all words in their ball are covered by others)
    # Then try 1-exchange moves: remove one codeword, add another to minimize total
    
    def is_redundant(cw_idx, balls):
        """Check if removing codeword cw_idx still leaves all words covered."""
        for idx in balls[cw_idx]:
            if coverage[idx] == 1:
                return False
        return True
    
    def try_remove(cw_idx, balls, code_arr):
        """Try removing codeword cw_idx. Returns True if successful."""
        if not is_redundant(cw_idx, balls):
            return False
        # Remove it
        for idx in balls[cw_idx]:
            coverage[idx] -= 1
        # Verify all still covered
        if np.any(coverage == 0):
            for idx in balls[cw_idx]:
                coverage[idx] += 1
            return False
        # Actually remove
        code_arr = np.delete(code_arr, cw_idx, axis=0)
        balls = [b for i, b in enumerate(balls) if i != cw_idx]
        return True, code_arr, balls
    
    # Phase 1: Remove redundant codewords
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code)):
            if time.time() >= deadline:
                break
            if is_redundant(i, codeword_balls):
                # Remove this codeword
                for idx in codeword_balls[i]:
                    coverage[idx] -= 1
                code = np.delete(code, i, axis=0)
                codeword_balls = [b for j, b in enumerate(codeword_balls) if j != i]
                changed = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()
                break  # Restart loop since indices changed
    
    # Phase 2: 1-exchange moves - try replacing one codeword with another to enable further removals
    # We'll use a greedy approach: for each codeword, try moving it to a better position
    for iteration in range(50):  # Limit iterations
        if time.time() >= deadline:
            break
        
        improved = False
        for i in range(len(code)):
            if time.time() >= deadline:
                break
            
            # Try moving codeword i to a random new position
            for attempt in range(20):
                if time.time() >= deadline:
                    break
                
                # Generate a random candidate
                candidate = np.random.randint(0, q, size=n, dtype=np.int8)
                if tuple(candidate) in [tuple(c) for c in code]:
                    continue
                
                # Check if replacing code[i] with candidate allows us to remove another codeword
                # First, compute what would be covered if we make the replacement
                old_ball = codeword_balls[i]
                new_ball = compute_ball(candidate)
                new_ball_indices = [word_to_idx[t] for t in new_ball if t in word_to_idx]
                
                # Simulate: remove old, add new
                temp_coverage = coverage.copy()
                for idx in old_ball:
                    temp_coverage[idx] -= 1
                for idx in new_ball_indices:
                    temp_coverage[idx] += 1
                
                # Check if this creates any redundancy we can exploit
                # Find a codeword j != i that becomes redundant
                best_j = -1
                for j in range(len(code)):
                    if j == i:
                        continue
                    if is_redundant(j, codeword_balls):
                        # Check if j would be redundant after the swap
                        temp_cov2 = temp_coverage.copy()
                        for idx in codeword_balls[j]:
                            temp_cov2[idx] -= 1
                        if np.all(temp_cov2 > 0):
                            best_j = j
                            break
                
                if best_j >= 0:
                    # Perform the swap and removal
                    # Remove codeword best_j
                    for idx in codeword_balls[best_j]:
                        coverage[idx] -= 1
                    
                    # Replace codeword i with candidate
                    for idx in old_ball:
                        coverage[idx] -= 1
                    code[i] = candidate
                    codeword_balls[i] = new_ball_indices
                    for idx in new_ball_indices:
                        coverage[idx] += 1
                    
                    # Remove codeword best_j from arrays
                    code = np.delete(code, best_j, axis=0)
                    codeword_balls = [b for j, b in enumerate(codeword_balls) if j != best_j]
                    
                    # Adjust index i if best_j < i
                    if best_j < i:
                        i -= 1
                    
                    improved = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()
                    break  # Restart loop since structure changed
                
                if improved:
                    break
        
        if not improved and len(code) == best_size:
            # No improvement, try a different approach: random restart with perturbation
            if time.time() >= deadline:
                break
    
    # Return the best code found
    if len(best_code) < len(code):
        return [list(cw) for cw in best_code]
    return [list(cw) for cw in code]
# EVOLVE-BLOCK-END
