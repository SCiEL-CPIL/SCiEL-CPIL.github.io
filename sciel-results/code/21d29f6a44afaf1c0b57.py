"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _word_to_int(word, q):
    """Convert a word (list of ints) to an integer index in [0, q^n)."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _int_to_word(idx, q, n):
    """Convert an integer index to a word (list of ints)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _generate_ball_indices(word_idx, q, n, radius):
    """Generate all word indices within Hamming distance `radius` from word_idx."""
    word = _int_to_word(word_idx, q, n)
    ball = {word_idx}
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(1, q), repeat=r):
                new_word = list(word)
                for pos, val in zip(positions, values):
                    new_word[pos] = (word[pos] + val) % q
                ball.add(_word_to_int(new_word, q))
    return ball


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave safety margin before 150s kill
    
    total_words = q ** n
    
    # Precompute ball indices for all words using numpy for speed
    # balls[i] is a numpy array of indices in the ball around word i
    balls = []
    for i in range(total_words):
        balls.append(np.array(sorted(_generate_ball_indices(i, q, n, radius)), dtype=np.int32))
    
    # Precompute a lookup: for each word index, which balls cover it
    # Actually, we'll use bitmask approach with numpy
    
    # For efficient coverage tracking, use a numpy boolean array
    # covered_mask[j] = True if word j is covered
    
    best_solution = None
    best_size = float('inf')
    
    # Try multiple random restarts
    num_restarts = 50
    for attempt in range(num_restarts):
        if time.time() > deadline - 20:  # Leave 20s for final validation
            break
        
        random.seed(42 + attempt)
        
        # Greedy set cover with bitmask
        covered_mask = np.zeros(total_words, dtype=bool)
        current_solution = []
        
        # Initialize: pick a random word
        init_idx = random.randint(0, total_words - 1)
        current_solution.append(init_idx)
        covered_mask[balls[init_idx]] = True
        
        uncovered_count = total_words - np.sum(covered_mask)
        
        while uncovered_count > 0:
            if time.time() > deadline - 20:
                break
            
            # Find word that covers most uncovered words
            # For each candidate word, count how many uncovered words it covers
            best_gain = -1
            best_word_idx = -1
            
            # Sample a subset of candidates for speed if total_words is large
            if total_words > 1000:
                candidates = random.sample(range(total_words), min(200, total_words))
            else:
                candidates = list(range(total_words))
            
            for i in candidates:
                if i in current_solution:
                    continue
                # Count uncovered words in ball i
                ball_i = balls[i]
                gain = np.sum(~covered_mask[ball_i])
                if gain > best_gain:
                    best_gain = gain
                    best_word_idx = i
            
            if best_word_idx == -1 or best_gain == 0:
                # Try all words if sampling didn't find anything
                for i in range(total_words):
                    if i in current_solution:
                        continue
                    ball_i = balls[i]
                    gain = np.sum(~covered_mask[ball_i])
                    if gain > best_gain:
                        best_gain = gain
                        best_word_idx = i
            
            if best_word_idx == -1 or best_gain == 0:
                break
            
            current_solution.append(best_word_idx)
            covered_mask[balls[best_word_idx]] = True
            uncovered_count = total_words - np.sum(covered_mask)
        
        # Local search: try to remove redundant codewords
        # Repeat until no improvement
        improved = True
        while improved:
            if time.time() > deadline - 20:
                break
            improved = False
            
            for idx in range(len(current_solution)):
                if time.time() > deadline - 20:
                    break
                
                # Try removing codeword at idx
                word_to_remove = current_solution[idx]
                
                # Compute coverage without this word
                temp_mask = np.zeros(total_words, dtype=bool)
                for j, w in enumerate(current_solution):
                    if j != idx:
                        temp_mask[balls[w]] = True
                
                if np.sum(temp_mask) == total_words:
                    # Safe to remove
                    current_solution.pop(idx)
                    improved = True
                    break  # Restart the loop since indices changed
        
        # Also try swapping: remove one, add another that covers more
        for _ in range(100):
            if time.time() > deadline - 20:
                break
            
            if len(current_solution) <= 1:
                break
            
            # Try removing a random codeword and adding a better one
            idx_to_remove = random.randint(0, len(current_solution) - 1)
            removed_word = current_solution[idx_to_remove]
            
            # Compute coverage without this word
            temp_mask = np.zeros(total_words, dtype=bool)
            for j, w in enumerate(current_solution):
                if j != idx_to_remove:
                    temp_mask[balls[w]] = True
            
            uncovered = total_words - np.sum(temp_mask)
            if uncovered == 0:
                # Safe to remove
                current_solution.pop(idx_to_remove)
                continue
            
            # Find a word that covers the most uncovered words
            best_gain = -1
            best_add_idx = -1
            
            for i in range(total_words):
                if i in current_solution:
                    continue
                ball_i = balls[i]
                gain = np.sum(~temp_mask[ball_i])
                if gain > best_gain:
                    best_gain = gain
                    best_add_idx = i
            
            if best_gain > 0 and best_add_idx != -1:
                # Replace: remove idx_to_remove, add best_add_idx
                current_solution[idx_to_remove] = best_add_idx
        
        # Validate and update best
        final_mask = np.zeros(total_words, dtype=bool)
        for w in current_solution:
            final_mask[balls[w]] = True
        
        if np.sum(final_mask) == total_words:
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                best_solution = [_int_to_word(w, q, n) for w in current_solution]
    
    if best_solution is None:
        # Fallback: use direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return best_solution
# EVOLVE-BLOCK-END
