import time
import random
import numpy as np

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1
    # We implement a generic solver that works for these parameters efficiently.
    # If parameters differ, we fallback to a simple greedy or the seed logic, 
    # but the task contract specifies this is for the 6-match football pool (q=3, n=6, r=1).
    
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic for other parameters to ensure validity
        # (Though the task is specific, robustness is good)
        if n > 4:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             import itertools
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
             import itertools
             return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimized solver for q=3, n=6, r=1
    total_words = q ** n  # 729
    
    # Precompute balls of radius 1 for all 729 words
    # ball[i] is a set of words covered by codeword i
    balls = []
    for i in range(total_words):
        # Convert integer to base-q digits
        word = [0] * n
        temp = i
        for j in range(n):
            word[j] = temp % q
            temp //= q
        
        ball = set()
        ball.add(i)
        # For each position, change digit to other q-1 values
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_word = word[:]
                    new_word[pos] = new_val
                    # Convert back to int
                    val = 0
                    for k in range(n):
                        val = val * q + new_word[k]
                    ball.add(val)
        balls.append(ball)
    
    # Convert balls to numpy arrays for faster set operations if needed, 
    # but sets of small ints are quite fast in Python.
    # Let's stick to sets for simplicity and sufficient speed for 729 items.
    
    best_code = None
    best_size = float('inf')
    
    # 1. Initial Greedy Construction
    code = []
    coverage_counts = np.zeros(total_words, dtype=np.int32)
    uncovered_mask = np.ones(total_words, dtype=bool) # True if uncovered
    
    while np.any(uncovered_mask):
        best_cand = -1
        max_cover = 0
        # To speed this up, we can sample or iterate all. 729 is small.
        for c in range(total_words):
            # Count how many uncovered words are in ball[c]
            # We can use numpy intersection or just sum
            # balls[c] is a set. uncovered_mask is a numpy array.
            # Creating a boolean array from set every time is slow.
            # Better: maintain a list of uncovered words?
            # Or just use the set.
            
            # Optimization: Precompute balls as numpy arrays of indices?
            # Let's use a different approach for speed:
            # Keep a list of uncovered words.
            pass
        
        # Let's switch to a slightly more efficient greedy using a list of uncovered words
        # Reset for a cleaner greedy implementation
        code = []
        coverage_counts = np.zeros(total_words, dtype=np.int32)
        uncovered_set = set(range(total_words))
        
        while uncovered_set:
            best_cand = -1
            max_cover = 0
            for c in range(total_words):
                # Intersect ball[c] with uncovered_set
                # len(balls[c].intersection(uncovered_set))
                # This is O(|ball|) which is 13. 729 * 13 is ~10k ops per step.
                # Max 73 steps. 73 * 10k = 730k ops. Very fast.
                cover_count = len(balls[c].intersection(uncovered_set))
                if cover_count > max_cover:
                    max_cover = cover_count
                    best_cand = c
            
            if best_cand == -1:
                break # Should not happen if radius covers space
            
            code.append(best_cand)
            # Update coverage and uncovered
            for w in balls[best_cand]:
                coverage_counts[w] += 1
                if coverage_counts[w] == 1:
                    uncovered_set.remove(w)
        
    best_code = code[:]
    best_size = len(code)
    
    # 2. Local Search (Iterated Local Search)
    # State: code (list of ints), coverage_counts (numpy array)
    
    def is_removable(c, coverage_counts):
        # Check if all words in ball[c] have count > 1
        for w in balls[c]:
            if coverage_counts[w] <= 1:
                return False
        return True

    def remove_codeword(c, code, coverage_counts):
        code.remove(c)
        for w in balls[c]:
            coverage_counts[w] -= 1

    def add_codeword(c, code, coverage_counts):
        code.append(c)
        for w in balls[c]:
            coverage_counts[w] += 1

    current_code = best_code[:]
    current_coverage = np.zeros(total_words, dtype=np.int32)
    for c in current_code:
        for w in balls[c]:
            current_coverage[w] += 1
            
    best_code = current_code[:]
    best_size = len(current_code)
    
    # Helper to try to reduce
    def try_reduce(code, coverage_counts):
        reduced = True
        while reduced:
            reduced = False
            # Try to remove any codeword
            # Shuffle to break symmetry/ties
            indices = list(range(len(code)))
            random.shuffle(indices)
            for idx in indices:
                c = code[idx]
                if is_removable(c, coverage_counts):
                    remove_codeword(c, code, coverage_counts)
                    reduced = True
                    break # Restart loop after a change
        return len(code)

    # Local Search Loop
    while time.time() < deadline:
        # Try to reduce current solution
        current_size = try_reduce(current_code, current_coverage)
        
        if current_size < best_size:
            best_size = current_size
            best_code = current_code[:]
            
        # If we can't reduce further, we are in a local minimum.
        # Apply perturbation:
        # 1. Remove a random codeword (breaking coverage)
        # 2. Greedily repair
        if len(current_code) > 0:
            # Pick a random codeword to remove
            # To encourage diversity, maybe pick one that is "critical" or just random
            c_to_remove = random.choice(current_code)
            
            # Save state
            old_code = current_code[:]
            old_coverage = current_coverage.copy()
            
            remove_codeword(c_to_remove, current_code, current_coverage)
            
            # Identify uncovered words
            uncovered_set = set()
            for w in range(total_words):
                if current_coverage[w] == 0:
                    uncovered_set.add(w)
            
            # Greedily repair
            repair_steps = 0
            while uncovered_set and repair_steps < 100: # Safety limit
                best_cand = -1
                max_cover = 0
                # Only consider adding codewords that cover at least one uncovered word
                # We can iterate all candidates, it's fast enough
                for c in range(total_words):
                    cover_count = len(balls[c].intersection(uncovered_set))
                    if cover_count > max_cover:
                        max_cover = cover_count
                        best_cand = c
                
                if best_cand == -1 or max_cover == 0:
                    break # No improvement possible or stuck (shouldn't happen if radius=1)
                
                add_codeword(best_cand, current_code, current_coverage)
                
                # Update uncovered_set
                for w in balls[best_cand]:
                    if current_coverage[w] == 1:
                        uncovered_set.discard(w)
                
                repair_steps += 1
            
            # Now try to reduce again
            new_size = try_reduce(current_code, current_coverage)
            
            # Accept the new state if it's not significantly worse, or always accept to explore?
            # Usually ILS accepts if better, or with some probability if worse.
            # Here, since we want the global min, and repair is greedy, 
            # the new state is likely similar size. 
            # We keep the current state as the working state regardless, 
            # but update best if improved.
            
            # If the perturbation didn't lead to a reduction or improvement, 
            # we might have increased the size. 
            # To avoid drifting to very large sizes, we can revert if size increased too much.
            if new_size > best_size + 5:
                # Revert to previous state
                current_code = old_code
                current_coverage = old_coverage
            # Else, keep the perturbed state to explore further from there
        else:
            break

    # Final cleanup: ensure we return the best found
    # Convert integers back to lists of digits
    result = []
    for c in best_code:
        word = [0] * n
        temp = c
        for j in range(n):
            word[j] = temp % q
            temp //= q
        result.append(word)
        
    return result
