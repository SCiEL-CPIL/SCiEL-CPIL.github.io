"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def _ball_indices(word, q, n, radius):
    """Yield indices of all words within Hamming distance radius of word."""
    for r in range(radius + 1):
        for positions in itertools.combinations(range(n), r):
            for deltas in itertools.product(range(1, q), repeat=r):
                yield tuple(word[i] + d if i in positions else word[i] for i, d in
                           ((pos, delta) for pos, delta in zip(positions, deltas)))

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius radius.
    Uses greedy construction followed by simulated annealing for local optimization.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin for 120s limit
    
    # Handle trivial cases
    if radius == 0:
        # Every word must be in the code
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    if n == 0:
        return [[]]
    
    # Special case: ternary tetracode construction for q=3, n=4, radius=1
    if q == 3 and n == 4 and radius == 1:
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # For q=3, n=6, radius=1: use known good construction
    if q == 3 and n == 6 and radius == 1:
        # Start with a reasonable initial code
        # Use direct product or other constructions
        # The tetracode gives 9 words of length 4; for length 6, we can extend
        # Or use greedy + SA
        
        # Let's use a combination approach
        # First, create a base code using greedy
        code = _greedy_construct(q, n, radius)
        
        # Then apply simulated annealing to improve
        code = _simulated_annealing(code, q, n, radius, deadline)
        
        return code
    
    # General case
    code = _greedy_construct(q, n, radius)
    code = _simulated_annealing(code, q, n, radius, deadline)
    return code


def _greedy_construct(q, n, radius):
    """Greedy set cover construction."""
    # Represent all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    num_words = len(all_words)
    
    # Precompute ball for each word
    balls = {}
    for w in all_words:
        ball = set()
        for r in range(radius + 1):
            for positions in itertools.combinations(range(n), r):
                for deltas in itertools.product(range(1, q), repeat=r):
                    new_word = list(w)
                    for pos, delta in zip(positions, deltas):
                        new_word[pos] = (new_word[pos] + delta) % q
                    ball.add(tuple(new_word))
        balls[w] = ball
    
    # Greedy set cover
    uncovered = set(range(num_words))
    code = []
    
    while uncovered:
        if not uncovered:
            break
        
        # Find the word that covers the most uncovered words
        best_word = None
        best_count = -1
        for w in all_words:
            if w in uncovered or True:  # Can pick any word as codeword
                count = len(balls[w] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = w
        
        if best_word is None or best_count == 0:
            break
        
        code.append(list(best_word))
        uncovered -= balls[best_word]
    
    return code


def _simulated_annealing(code, q, n, radius, deadline):
    """Simulated annealing to minimize code size while maintaining coverage."""
    start_time = time.time()
    
    # Convert to tuples for hashing
    code_tuples = [tuple(c) for c in code]
    
    # Precompute balls for current code
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    num_words = len(all_words)
    
    def compute_balls(codeword):
        ball = set()
        for r in range(radius + 1):
            for positions in itertools.combinations(range(n), r):
                for deltas in itertools.product(range(1, q), repeat=r):
                    new_word = list(codeword)
                    for pos, delta in zip(positions, deltas):
                        new_word[pos] = (new_word[pos] + delta) % q
                    ball.add(tuple(new_word))
        return ball
    
    # Initial coverage
    covered = set()
    for cw in code_tuples:
        covered |= compute_balls(cw)
    
    best_code = list(code_tuples)
    best_size = len(code_tuples)
    current_temp = 100.0
    cooling_rate = 0.99
    
    rng = random.Random(42)
    
    while time.time() < deadline and current_temp > 0.01:
        if len(code_tuples) <= best_size:
            best_code = list(code_tuples)
            best_size = len(code_tuples)
        
        # Try to remove a codeword
        if len(code_tuples) > 1:
            idx = rng.randint(0, len(code_tuples) - 1)
            cw = code_tuples[idx]
            ball = compute_balls(cw)
            
            # Check if removing this codeword maintains coverage
            # A word is still covered if it's in the ball of some other codeword
            still_covered = True
            for w in ball:
                if w not in covered:
                    still_covered = False
                    break
                # Check if w is covered by another codeword
                found = False
                for j, other_cw in enumerate(code_tuples):
                    if j == idx:
                        continue
                    if w in compute_balls(other_cw):
                        found = True
                        break
                if not found:
                    still_covered = False
                    break
            
            if still_covered:
                # Accept removal
                code_tuples.pop(idx)
                covered -= ball
                continue
        
        # Try to swap: remove one, add another
        if len(code_tuples) > 0 and rng.random() < 0.5:
            remove_idx = rng.randint(0, len(code_tuples) - 1)
            add_word = rng.choice(all_words)
            
            if add_word not in code_tuples:
                remove_ball = compute_balls(code_tuples[remove_idx])
                add_ball = compute_balls(add_word)
                
                # Check coverage after swap
                new_covered = (covered - remove_ball) | add_ball
                if len(new_covered) == num_words:
                    covered = new_covered
                    code_tuples[remove_idx] = add_word
        
        current_temp *= cooling_rate
    
    # Return the best code found
    return [list(c) for c in best_code]


# EVOLVE-BLOCK-END
