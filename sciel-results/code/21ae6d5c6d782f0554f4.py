"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def _hamming_distance(a, b):
    """Compute Hamming distance between two tuples/lists of integers."""
    return sum(x != y for x, y in zip(a, b))


def _generate_ball(center, q, n, radius):
    """Generate all words within Hamming distance <= radius from center."""
    ball = [center]
    for r in range(1, radius + 1):
        # For each position, change to each other symbol
        for pos in range(n):
            orig = center[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                new_word = list(center)
                new_word[pos] = new_val
                new_word = tuple(new_word)
                ball.append(new_word)
    return set(ball)


def construct(q, n, radius):
    """
    Greedy set-covering with simulated annealing refinement for ternary covering code.
    """
    start_time = time.time()
    time_limit = 115  # leave some buffer before 120s timeout
    
    # Total number of words in the space
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = [tuple(word) for word in np.ndindex(*([q] * n))]
    all_words_set = set(all_words)
    
    # For greedy set cover: we want to select minimum number of centers
    # such that every word is within radius of at least one center.
    
    # Strategy 1: Greedy set covering
    # Start with no centers, repeatedly pick the center that covers the most uncovered words.
    
    # To make this efficient, we can use a priority queue or just scan.
    # Given the space is 729 for q=3, n=6, this is feasible.
    
    uncovered = set(all_words_set)
    centers = []
    
    # Precompute balls for all possible centers? That's 729 * (1 + 6*2 + ...) which is manageable.
    # Actually, let's just compute balls on the fly or precompute them.
    
    # Precompute all balls
    all_balls = {}
    for w in all_words:
        all_balls[w] = _generate_ball(w, q, n, radius)
    
    # Greedy set cover
    while uncovered:
        if time.time() - start_time > time_limit * 0.5:
            break
        
        best_center = None
        best_coverage = -1
        best_newly_covered = None
        
        for center in all_words:
            ball = all_balls[center]
            newly_covered = ball & uncovered
            coverage = len(newly_covered)
            if coverage > best_coverage:
                best_coverage = coverage
                best_center = center
                best_newly_covered = newly_covered
        
        if best_center is None or best_coverage <= 0:
            break
        
        centers.append(best_center)
        uncovered -= best_newly_covered
    
    # If greedy didn't cover everything (shouldn't happen), add more centers
    if uncovered:
        for w in uncovered:
            centers.append(w)
            uncovered -= all_balls[w]
    
    # Now we have a valid covering. Let's try to improve it with simulated annealing.
    # We can try to remove centers or swap them to reduce the count.
    
    current_centers = list(centers)
    current_count = len(current_centers)
    
    # Simulated annealing: try to remove a center if coverage is still maintained,
    # or swap a center with another to potentially allow removal of others.
    
    best_solution = list(current_centers)
    best_count = current_count
    
    # Track coverage counts: for each word, how many centers cover it
    coverage_count = np.zeros(total_words, dtype=np.int32)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    for c in current_centers:
        for w in all_balls[c]:
            idx = word_to_idx[w]
            coverage_count[idx] += 1
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    random.seed(42)
    
    while T > T_min and time.time() - start_time < time_limit:
        # Try a random move: remove a random center, or swap two centers
        move_type = random.choice(['remove', 'swap', 'add_remove'])
        
        if move_type == 'remove' and len(current_centers) > 1:
            # Try removing a random center
            idx = random.randint(0, len(current_centers) - 1)
            c = current_centers[idx]
            
            # Check if removing c still covers everything
            # A word is still covered if coverage_count[word] > 1 after removing c
            removable = True
            for w in all_balls[c]:
                idx_w = word_to_idx[w]
                if coverage_count[idx_w] <= 1:
                    removable = False
                    break
            
            if removable:
                # Remove c
                for w in all_balls[c]:
                    idx_w = word_to_idx[w]
                    coverage_count[idx_w] -= 1
                current_centers.pop(idx)
                new_count = len(current_centers)
                
                if new_count < best_count:
                    best_count = new_count
                    best_solution = list(current_centers)
            # If not removable, do nothing (or we could accept with some probability, but it's invalid)
        
        elif move_type == 'swap' and len(current_centers) >= 2:
            # Try swapping two centers
            i, j = random.sample(range(len(current_centers)), 2)
            c1 = current_centers[i]
            c2 = current_centers[j]
            
            # Swap them (this doesn't change the set, so no benefit)
            # Instead, let's try: remove c1, add a random new center c_new
            pass
        
        elif move_type == 'add_remove':
            # Try removing a center and adding a different one
            if len(current_centers) < 2:
                continue
            idx_remove = random.randint(0, len(current_centers) - 1)
            c_remove = current_centers[idx_remove]
            
            # Pick a random new center
            c_new = random.choice(all_words)
            
            # Check if we can remove c_remove and add c_new such that all words are still covered
            # Words that were only covered by c_remove need to be covered by c_new
            only_covered_by_remove = []
            for w in all_balls[c_remove]:
                idx_w = word_to_idx[w]
                if coverage_count[idx_w] == 1:
                    only_covered_by_remove.append(w)
            
            # Check if c_new covers all words in only_covered_by_remove
            ball_new = all_balls[c_new]
            if all(w in ball_new for w in only_covered_by_remove):
                # Accept the move
                # Remove c_remove
                for w in all_balls[c_remove]:
                    idx_w = word_to_idx[w]
                    coverage_count[idx_w] -= 1
                
                # Add c_new
                for w in ball_new:
                    idx_w = word_to_idx[w]
                    coverage_count[idx_w] += 1
                
                current_centers[idx_remove] = c_new
        
        # Cool down
        T *= cooling_rate
    
    # Use the best solution found
    final_centers = best_solution
    
    # Verify coverage
    covered = set()
    for c in final_centers:
        covered |= all_balls[c]
    
    if covered != all_words_set:
        # Fallback to greedy solution
        final_centers = centers
    
    # Return as list of lists
    return [list(c) for c in final_centers]
# EVOLVE-BLOCK-END
