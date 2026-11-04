"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _word_to_int(word, q):
    """Convert a word (list of ints 0..q-1) to an integer index."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _int_to_word(idx, q, n):
    """Convert an integer index to a word (list of ints 0..q-1)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _generate_ball_indices(word, q, n, radius):
    """Generate all word indices within Hamming distance `radius` from `word`."""
    ball = {tuple(word)}
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(1, q), repeat=r):
                new_word = list(word)
                for pos, val in zip(positions, values):
                    new_word[pos] = (word[pos] + val) % q
                ball.add(tuple(new_word))
    return ball


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Leave safety margin before 150s kill
    
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    # Precompute ball indices for each word as a set of integer indices
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    ball_masks = []
    for w in all_words:
        ball = _generate_ball_indices(w, q, n, radius)
        mask = 0
        for b in ball:
            mask |= (1 << word_to_idx[b])
        ball_masks.append(mask)
    
    # Convert to numpy uint64 arrays for fast operations
    # For total_words up to ~1000, we need multiple uint64 chunks
    num_chunks = (total_words + 63) // 64
    
    def mask_to_chunks(mask):
        chunks = np.zeros(num_chunks, dtype=np.uint64)
        for c in range(num_chunks):
            chunks[c] = (mask >> (c * 64)) & 0xFFFFFFFFFFFFFFFF
        return chunks
    
    ball_chunks = [mask_to_chunks(m) for m in ball_masks]
    
    best_solution = None
    best_size = float('inf')
    
    def is_full_coverage(covered_chunks):
        for c in range(num_chunks):
            expected = (1 << 64) - 1
            if c == num_chunks - 1 and (total_words % 64) != 0:
                expected = (1 << (total_words % 64)) - 1
            if covered_chunks[c] != expected:
                return False
        return True
    
    # Try multiple random restarts
    for attempt in range(50):
        if time.time() > deadline - 20:  # Leave 20s for final validation
            break
        
        random.seed(42 + attempt)
        
        # Greedy set cover using bitmasks
        covered_chunks = np.zeros(num_chunks, dtype=np.uint64)
        current_solution = []
        
        while not is_full_coverage(covered_chunks):
            if time.time() > deadline - 20:
                break
            
            # Find word that covers most uncovered words
            best_gain = -1
            best_word_idx = -1
            
            for i in range(total_words):
                gain = 0
                for c in range(num_chunks):
                    gain += np.bitwise_count(ball_chunks[i][c] & ~covered_chunks[c])
                if gain > best_gain:
                    best_gain = gain
                    best_word_idx = i
            
            if best_word_idx == -1 or best_gain == 0:
                break
            
            current_solution.append(best_word_idx)
            for c in range(num_chunks):
                covered_chunks[c] |= ball_chunks[best_word_idx][c]
        
        # Local search: try removing redundant codewords
        temp = 10.0
        cooling = 0.95
        max_sa_iters = 2000
        
        for sa_iter in range(max_sa_iters):
            if time.time() > deadline - 20:
                break
            
            if len(current_solution) <= 1:
                break
            
            # Try removing a random codeword
            idx_to_remove = random.randint(0, len(current_solution) - 1)
            
            # Check coverage without this word
            temp_covered = np.zeros(num_chunks, dtype=np.uint64)
            for i, w_idx in enumerate(current_solution):
                if i == idx_to_remove:
                    continue
                for c in range(num_chunks):
                    temp_covered[c] |= ball_chunks[w_idx][c]
            
            if is_full_coverage(temp_covered):
                # Safe to remove
                current_solution.pop(idx_to_remove)
                covered_chunks = temp_covered.copy()
                temp *= cooling
            else:
                # Not safe, accept with probability based on temperature
                penalty = 0
                for c in range(num_chunks):
                    penalty += np.bitwise_count(covered_chunks[c] & ~temp_covered[c])
                if penalty > 0:
                    if random.random() < np.exp(-penalty / max(temp, 0.01)):
                        current_solution.pop(idx_to_remove)
                        covered_chunks = temp_covered.copy()
                temp *= cooling
            
            # Occasionally try swapping: remove one, add another
            if random.random() < 0.3 and len(current_solution) > 1:
                idx_to_swap = random.randint(0, len(current_solution) - 1)
                # Try adding a random word and see if we can then remove more
                new_word = random.randint(0, total_words - 1)
                if new_word not in current_solution:
                    trial_solution = current_solution + [new_word]
                    trial_covered = np.zeros(num_chunks, dtype=np.uint64)
                    for w_idx in trial_solution:
                        for c in range(num_chunks):
                            trial_covered[c] |= ball_chunks[w_idx][c]
                    
                    # Try to remove words from trial_solution
                    improved = False
                    for rm_idx in range(len(trial_solution)):
                        tc = np.zeros(num_chunks, dtype=np.uint64)
                        for j, w_idx in enumerate(trial_solution):
                            if j == rm_idx:
                                continue
                            for c in range(num_chunks):
                                tc[c] |= ball_chunks[w_idx][c]
                        if is_full_coverage(tc):
                            trial_solution.pop(rm_idx)
                            improved = True
                            break
                    
                    if improved and len(trial_solution) < len(current_solution):
                        current_solution = trial_solution
                        covered_chunks = np.zeros(num_chunks, dtype=np.uint64)
                        for w_idx in current_solution:
                            for c in range(num_chunks):
                                covered_chunks[c] |= ball_chunks[w_idx][c]
        
        # Validate and update best
        if is_full_coverage(covered_chunks):
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                best_solution = [list(_int_to_word(w, q, n)) for w in current_solution]
    
    if best_solution is None:
        # Fallback: use direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return best_solution
# EVOLVE-BLOCK-END
