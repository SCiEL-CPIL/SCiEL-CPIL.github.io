"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _word_to_int(word, q, n):
    """Convert a word (list of ints) to an integer index."""
    idx = 0
    for i, v in enumerate(word):
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
    ball = set()
    ball.add(word_idx)
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(1, q), repeat=r):
                new_word = list(word)
                for pos, val in zip(positions, values):
                    new_word[pos] = (word[pos] + val) % q
                ball.add(_word_to_int(new_word, q, n))
    return ball


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Leave safety margin before 150s kill
    
    total_words = q ** n
    full_mask = (1 << total_words) - 1
    
    # Precompute balls as bitmasks (using Python's arbitrary-precision ints)
    # Each ball is represented as an integer where bit i is set if word i is in the ball
    balls = [0] * total_words
    for idx in range(total_words):
        ball_indices = _generate_ball_indices(idx, q, n, radius)
        mask = 0
        for i in ball_indices:
            mask |= (1 << i)
        balls[idx] = mask
    
    all_masks = balls  # alias
    
    best_solution = None
    best_size = float('inf')
    
    # Try multiple random restarts with greedy + local search
    num_restarts = 50
    for attempt in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        rng = random.Random(12345 + attempt)
        
        # Greedy set cover using bitmasks with bit_count for speed
        current_solution = []
        covered_mask = 0
        
        # Start with a random word
        first_idx = rng.randint(0, total_words - 1)
        current_solution.append(first_idx)
        covered_mask |= all_masks[first_idx]
        
        # Greedy: repeatedly add the word that covers the most uncovered words
        while covered_mask != full_mask:
            if time.time() > deadline - 20:
                break
            
            uncovered_mask = full_mask ^ covered_mask
            best_gain = -1
            best_idx = -1
            
            for i in range(total_words):
                if i in current_solution:
                    continue
                gain = (all_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
            
            current_solution.append(best_idx)
            covered_mask |= all_masks[best_idx]
        
        # Local search: try to remove redundant codewords using coverage count
        def _try_remove_redundant(sol):
            """Remove any redundant codewords. Returns modified sol."""
            improved = True
            while improved:
                if time.time() > deadline - 20:
                    break
                improved = False
                # Compute coverage count for each word
                cov_count = [0] * total_words
                for idx in sol:
                    m = all_masks[idx]
                    # Count bits set in m that are covered by this word
                    # We need to check if any word is ONLY covered by this codeword
                    # Faster: compute union of all other masks and check
                # Simpler approach: for each codeword, check if removing it still covers all
                for i in range(len(sol)):
                    remaining_mask = 0
                    for j in range(len(sol)):
                        if j != i:
                            remaining_mask |= all_masks[sol[j]]
                    if remaining_mask == full_mask:
                        sol.pop(i)
                        improved = True
                        break
            return sol
        
        current_solution = _try_remove_redundant(current_solution)
        
        # Move-based local search: try replacing one codeword with another
        # that maintains coverage, potentially enabling further removals
        move_iterations = 0
        max_moves = 200
        while move_iterations < max_moves:
            if time.time() > deadline - 20:
                break
            if len(current_solution) < 2:
                break
            
            move_iterations += 1
            i = rng.randint(0, len(current_solution) - 1)
            old_idx = current_solution[i]
            
            # Compute coverage without old_idx
            mask_without_old = 0
            for j in range(len(current_solution)):
                if j != i:
                    mask_without_old |= all_masks[current_solution[j]]
            
            # Find words that could replace old_idx to maintain coverage
            # A word w can replace old_idx if mask_without_old | all_masks[w] == full_mask
            # i.e., all_masks[w] must cover (full_mask & ~mask_without_old)
            uncovered_by_others = full_mask & ~mask_without_old
            if uncovered_by_others == 0:
                # old_idx is redundant, remove it
                current_solution.pop(i)
                current_solution = _try_remove_redundant(current_solution)
                continue
            
            # Find candidate replacements: words whose mask covers all of uncovered_by_others
            # For efficiency, just try random candidates
            found_replacement = False
            for _ in range(30):  # Try up to 30 random candidates
                if time.time() > deadline - 20:
                    break
                new_idx = rng.randint(0, total_words - 1)
                if new_idx in current_solution:
                    continue
                if (all_masks[new_idx] & uncovered_by_others) == uncovered_by_others:
                    # This replacement maintains coverage
                    current_solution[i] = new_idx
                    found_replacement = True
                    break
            
            if found_replacement:
                # Try to remove redundant ones after the move
                current_solution = _try_remove_redundant(current_solution)
        
        # Additional: try adding a word then removing two (net -1)
        add_remove_iterations = 0
        max_add_remove = 50
        while add_remove_iterations < max_add_remove:
            if time.time() > deadline - 20:
                break
            add_remove_iterations += 1
            
            # Try adding a random word
            new_idx = rng.randint(0, total_words - 1)
            if new_idx in current_solution:
                continue
            
            current_solution.append(new_idx)
            
            # Now try to remove two words (including possibly the new one)
            removed = 0
            for i in range(len(current_solution) - 1, -1, -1):
                if removed >= 2:
                    break
                remaining_mask = 0
                for j in range(len(current_solution)):
                    if j != i:
                        remaining_mask |= all_masks[current_solution[j]]
                if remaining_mask == full_mask:
                    current_solution.pop(i)
                    removed += 1
            
            if removed < 2:
                # Net size didn't decrease, revert if we added
                if len(current_solution) > 0 and current_solution[-1] == new_idx and removed < 1:
                    # The new word is still there and we didn't remove enough
                    # Revert: remove the new word if it's still present
                    if new_idx in current_solution:
                        current_solution.remove(new_idx)
        
        # Final cleanup
        current_solution = _try_remove_redundant(current_solution)
        
        # Validate and update best
        final_mask = 0
        for idx in current_solution:
            final_mask |= all_masks[idx]
        
        if final_mask == full_mask:
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                best_solution = [_int_to_word(idx, q, n) for idx in current_solution]
    
    if best_solution is None:
        # Fallback: use direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return best_solution
# EVOLVE-BLOCK-END
