import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 120.0 # Keep a safety margin from 150s
    
    # For the specific task q=3, n=6, radius=1
    # But the function signature is generic.
    # If q, n, radius differ, we fall back to a simple heuristic or the seed logic.
    # However, the prompt implies this is for the specific task.
    # The evaluator passes q, n, radius.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: use the seed logic if applicable, else simple greedy
        if n == 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Generic fallback: Greedy Set Cover
        return _greedy_cover(q, n, radius)

    # Specific optimization for 3, 6, 1
    N = q ** n
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (ball of radius 1) for each word
    # ball[i] is a set of indices of words within distance 1 of all_words[i]
    balls = []
    for w in all_words:
        ball = set()
        ball.add(word_to_idx[tuple(w)])
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig: continue
                new_w = w.copy()
                new_w[pos] = val
                ball.add(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    # 1. Greedy Set Cover to get initial solution
    uncovered = set(range(N))
    code_indices = []
    
    # Sort words by some heuristic? Or just iterate.
    # To speed up, we can precompute for each word, the list of words it covers.
    # Greedy step: pick word covering most uncovered.
    
    while uncovered:
        best_word = -1
        max_covered = 0
        # Optimization: If N is small, iterating all is fine.
        for i in range(N):
            if i in code_indices: continue # Not strictly necessary for set cover, but code should be distinct
            # Calculate overlap with uncovered
            # We can maintain a count array for efficiency
            # But for N=729, simple set intersection is fast enough for first pass
            # Let's use a counter array for speed in the main loop if we do many iterations
            # For initial greedy, simple is fine.
            inter = len(balls[i] & uncovered)
            if inter > max_covered:
                max_covered = inter
                best_word = i
                
        if best_word == -1:
            break # Should not happen
            
        code_indices.append(best_word)
        uncovered -= balls[best_word]
        
    initial_code = [all_words[i] for i in code_indices]
    
    # 2. Greedy Reduction: Remove any redundant codeword
    def is_valid(code_set):
        # Check if union of balls covers all words
        # Use a boolean array for speed
        covered = np.zeros(N, dtype=bool)
        for idx in code_set:
            covered[list(balls[idx])] = True
        return np.all(covered)
        
    # Faster validity check using sets
    def is_valid_set(code_set):
        covered = set()
        for idx in code_set:
            covered.update(balls[idx])
        return len(covered) == N

    # Remove redundant
    changed = True
    while changed:
        changed = False
        for i in range(len(code_indices)):
            # Try removing code_indices[i]
            new_set = set(code_indices)
            new_set.remove(code_indices[i])
            if is_valid_set(new_set):
                code_indices.pop(i)
                changed = True
                break # Restart loop as indices changed
    
    # 3. Local Search
    # We try to find a smaller cover.
    # Move: Replace one codeword with another, then try to remove one.
    # Or just try to remove one (failed in greedy reduction, but maybe after swaps?)
    # Better: Simulated Annealing on the set of codewords.
    
    current_code_set = set(code_indices)
    best_code_set = set(code_indices)
    best_size = len(best_code_set)
    
    # Precompute list of all words not in current code? No, just pick random.
    
    # To speed up validity checks, we can maintain the coverage counts.
    # cov_count[i] = number of codewords in code_set that cover word i.
    # A set is valid iff all cov_count[i] > 0.
    
    def build_cov_counts(code_set):
        counts = np.zeros(N, dtype=int)
        for idx in code_set:
            counts[list(balls[idx])] += 1
        return counts
        
    current_counts = build_cov_counts(current_code_set)
    
    def try_remove(code_set, counts, idx_to_remove):
        # Check if removing idx_to_remove keeps all counts > 0
        # Only words covered by idx_to_remove are affected
        # If any of those words had count 1, it becomes 0 -> invalid
        for w in balls[idx_to_remove]:
            if counts[w] == 1:
                return False
        return True
        
    def do_remove(code_set, counts, idx_to_remove):
        code_set.remove(idx_to_remove)
        for w in balls[idx_to_remove]:
            counts[w] -= 1
            
    def do_add(code_set, counts, idx_to_add):
        code_set.add(idx_to_add)
        for w in balls[idx_to_add]:
            counts[w] += 1
            
    # Local Search Loop
    # We accept moves that reduce size.
    # We also accept neutral moves (same size) to explore.
    
    temperature = 1.0
    cooling = 0.99
    
    while time.time() - start_time < deadline - 5.0:
        if len(current_code_set) < best_size:
            best_size = len(current_code_set)
            best_code_set = set(current_code_set)
            
        # If we hit the theoretical lower bound or known best, we can stop?
        # Best known is 73. Lower bound 71.
        if best_size <= 73:
             # We found a very good solution.
             # Can we go lower?
             pass
            
        # Pick a random codeword to remove
        if not current_code_set:
            break
            
        idx_remove = random.choice(list(current_code_set))
        
        # Pick a random word to add (could be same as remove, effectively a no-op swap, or different)
        # To encourage change, pick from all words
        idx_add = random.randint(0, N - 1)
        
        if idx_remove == idx_add:
            continue
            
        # Check if we can remove idx_remove first?
        # Strategy: 
        # 1. Try to remove idx_remove. If valid, do it. (Size decreases)
        # 2. If not, try swapping: remove idx_remove, add idx_add.
        #    Check if new set is valid.
        #    If valid, and if we can then remove another word, great.
        #    If valid, accept swap (neutral move).
        
        # Step 1: Pure removal
        if try_remove(current_code_set, current_counts, idx_remove):
            do_remove(current_code_set, current_counts, idx_remove)
            # Reset temperature on improvement?
            temperature = 1.0
            continue
            
        # Step 2: Swap
        # Tentatively remove idx_remove
        # We need to check validity of (current - idx_remove + idx_add)
        # This is equivalent to:
        # For all w in balls[idx_remove] \ balls[idx_add]: count must be > 1 (so removing idx_remove doesn't kill it)
        # For all w in balls[idx_add] \ balls[idx_remove]: count can be anything (adding helps)
        # For w in intersection: count changes -1+1 = 0 net change, so must be > 0 originally?
        # Actually, if w is in intersection, it was covered by idx_remove and will be covered by idx_add.
        # So the condition is:
        # For all w in balls[idx_remove]:
        #   if w NOT in balls[idx_add]:
        #      counts[w] must be > 1 (so it remains covered by others)
        #   if w IN balls[idx_add]:
        #      counts[w] must be > 0 (it was covered, will be covered by new one) -> always true if valid state
        
        valid_swap = True
        for w in balls[idx_remove]:
            if w not in balls[idx_add]:
                if current_counts[w] == 1:
                    valid_swap = False
                    break
            
        if valid_swap:
            # Perform swap
            do_remove(current_code_set, current_counts, idx_remove)
            do_add(current_code_set, current_counts, idx_add)
            
            # After swap, try to remove any word (including the new one? No, it's needed for swap validity usually, but maybe not if it covers redundant parts)
            # Let's try to remove a random word from the new set
            # Or just check if any word is redundant now
            # To save time, just try to remove idx_add? No, idx_add was just added.
            # Try to remove a random other word.
            # Actually, the swap itself is a neutral move. It might enable a future removal.
            # We accept it.
            
            # Optional: Greedy reduction after swap
            # Pick a random word from current_code_set and try to remove
            if current_code_set:
                idx_try_remove = random.choice(list(current_code_set))
                if try_remove(current_code_set, current_counts, idx_try_remove):
                    do_remove(current_code_set, current_counts, idx_try_remove)
                    temperature = 1.0
            else:
                # Should not happen
                pass
        
        temperature *= cooling

    # Convert best_code_set back to list of words
    best_code_words = [all_words[i] for i in best_code_set]
    
    # Final check
    if not is_valid_set(best_code_set):
         # Fallback to initial if something went wrong
         return initial_code
         
    return best_code_words

def _greedy_cover(q, n, radius):
    # Fallback for other parameters
    N = q ** n
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute balls
    balls = []
    for w in all_words:
        ball = set()
        ball.add(word_to_idx[tuple(w)])
        # Generate all neighbors within radius
        # For small radius, use product
        for delta in itertools.product(range(-radius, radius+1), repeat=n):
            # Check if distance is <= radius
            dist = sum(abs(d) for d in delta)
            if dist <= radius:
                new_w = [0]*n
                valid = True
                for i in range(n):
                    val = w[i] + delta[i]
                    if 0 <= val < q:
                        new_w[i] = val
                    else:
                        valid = False
                        break
                if valid:
                    ball.add(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    uncovered = set(range(N))
    code_indices = []
    
    while uncovered:
        best_word = -1
        max_covered = 0
        for i in range(N):
            if i in code_indices: continue
            inter = len(balls[i] & uncovered)
            if inter > max_covered:
                max_covered = inter
                best_word = i
        if best_word == -1: break
        code_indices.append(best_word)
        uncovered -= balls[best_word]
        
    return [all_words[i] for i in code_indices]
