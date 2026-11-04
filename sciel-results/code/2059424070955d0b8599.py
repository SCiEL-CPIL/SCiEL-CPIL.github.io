"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _word_to_int(word, q):
    """Convert a word (list of ints) to an integer index."""
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
    """Generate all word indices within Hamming distance `radius` from `word_idx`."""
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
    deadline = start_time + 110  # Leave 40s safety margin before 150s kill
    
    total_words = q ** n
    all_words = [_int_to_word(i, q, n) for i in range(total_words)]
    
    # Precompute balls as bitmasks for fast coverage tracking
    balls = [None] * total_words
    for i in range(total_words):
        balls[i] = _generate_ball_indices(i, q, n, radius)
    
    # Convert balls to bitmasks for O(1) union operations
    ball_masks = [0] * total_words
    for i in range(total_words):
        mask = 0
        for idx in balls[i]:
            mask |= (1 << idx)
        ball_masks[i] = mask
    
    best_solution = None
    best_size = float('inf')
    
    # Try multiple random restarts
    for attempt in range(30):
        if time.time() > deadline - 20:
            break
        
        random.seed(42 + attempt)
        
        # Greedy set-covering with bitmask tracking
        current_solution = []
        covered_mask = 0
        
        # Greedy: add words that cover the most uncovered words
        while covered_mask != (1 << total_words) - 1:
            if time.time() > deadline - 20:
                break
            
            # Find word that covers most uncovered words
            best_gain = -1
            best_word_idx = -1
            
            for i in range(total_words):
                gain = bin(ball_masks[i] & ~covered_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_word_idx = i
            
            if best_word_idx == -1 or best_gain == 0:
                break
            
            current_solution.append(best_word_idx)
            covered_mask |= ball_masks[best_word_idx]
        
        # Local search: try to remove redundant codewords
        temp = 10.0
        cooling = 0.95
        
        for sa_iter in range(300):
            if time.time() > deadline - 20:
                break
            
            if len(current_solution) <= 1:
                break
            
            # Try removing a random codeword
            idx_to_remove = random.randint(0, len(current_solution) - 1)
            removed_idx = current_solution[idx_to_remove]
            
            # Check if still covered without this word
            temp_mask = 0
            for j, w_idx in enumerate(current_solution):
                if j != idx_to_remove:
                    temp_mask |= ball_masks[w_idx]
            
            if temp_mask == (1 << total_words) - 1:
                # Safe to remove
                current_solution.pop(idx_to_remove)
                temp *= cooling
            else:
                # Not safe, accept with probability based on temperature
                penalty = total_words - bin(temp_mask).count('1')
                if penalty > 0 and random.random() < np.exp(-penalty / max(temp, 0.01)):
                    current_solution.pop(idx_to_remove)
                temp *= cooling
        
        # Validate and update best
        final_mask = 0
        for w_idx in current_solution:
            final_mask |= ball_masks[w_idx]
        
        if final_mask == (1 << total_words) - 1:
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                best_solution = [all_words[i] for i in current_solution]
    
    if best_solution is None:
        # Fallback: use direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return best_solution
# EVOLVE-BLOCK-END
