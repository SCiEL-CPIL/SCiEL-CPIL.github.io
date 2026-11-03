import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # Specialized for K_3(7,1)
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters: Simple Greedy
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_index = {w: i for i, w in enumerate(all_words)}
        balls = []
        for i, w in enumerate(all_words):
            neighbors = [i]
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    nw = list(w); nw[pos] = val
                    neighbors.append(word_index[tuple(nw)])
            balls.append(neighbors)
        
        uncovered = set(range(N))
        solution = []
        while uncovered:
            best = -1
            best_count = -1
            for i in range(N):
                if i in solution: continue
                count = sum(1 for nb in balls[i] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best = i
            if best == -1: break
            solution.append(best)
            for nb in balls[best]:
                uncovered.discard(nb)
        return [all_words[i] for i in solution]

    # Specific logic for q=3, n=7, radius=1
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words) # 2187
    
    # Precompute balls as a boolean matrix for vectorization
    # balls[i] is a boolean array of length N, True if j is in ball(i)
    balls = np.zeros((N, N), dtype=bool)
    for i, w in enumerate(all_words):
        balls[i, i] = True
        for pos in range(n):
            for val in range(3):
                if val == w[pos]: continue
                nw = list(w)
                nw[pos] = val
                j = all_words.index(tuple(nw)) # This is slow, let's use index dict
                balls[i, j] = True

    # Actually, building index dict is faster for the loop above
    word_index = {w: i for i, w in enumerate(all_words)}
    balls = np.zeros((N, N), dtype=bool)
    for i, w in enumerate(all_words):
        balls[i, i] = True
        for pos in range(n):
            for val in range(3):
                if val == w[pos]: continue
                nw = list(w)
                nw[pos] = val
                j = word_index[tuple(nw)]
                balls[i, j] = True

    # Helper: Check coverage and return uncovered mask
    def get_uncovered(sol_indices):
        if not sol_indices:
            return np.ones(N, dtype=bool)
        # Sum of balls for all selected indices
        # If any selected index covers a word, the sum is > 0
        covered = np.sum(balls[sol_indices], axis=0)
        return covered == 0

    # Helper: Greedy Add - add best word to cover uncovered
    def greedy_add(uncovered_mask, current_sol_set):
        if not np.any(uncovered_mask):
            return None
        # For each candidate i, count how many uncovered words it covers
        # balls[i] is bool vector. uncovered_mask is bool vector.
        # Dot product gives count of True overlaps.
        # We want to maximize this count.
        # We can compute this for all i at once? 
        # balls.T @ uncovered_mask gives a vector of counts for each i.
        counts = balls.T @ uncovered_mask
        # Mask out already selected words (set their count to -1 or ignore)
        for idx in current_sol_set:
            counts[idx] = -1
        best_idx = np.argmax(counts)
        if counts[best_idx] <= 0:
            return None
        return best_idx

    # Helper: Local Search - Remove redundant, Add best
    def local_search(sol_indices):
        sol_set = set(sol_indices)
        uncovered = get_uncovered(sol_indices)
        
        improved = True
        while improved:
            improved = False
            
            # 1. Try to remove redundant codewords
            # A codeword is redundant if removing it doesn't create new uncovered words.
            # This is equivalent to: for each i in sol, check if balls[i] is subset of union of balls of (sol \ {i})
            # Faster check: 
            # Maintain coverage count? 
            # Let's just try removing each one and checking if uncovered becomes non-empty.
            # To be efficient, we can check:
            # Is there any word w such that w is covered ONLY by i?
            # coverage_count[w] == 1 and balls[i, w] == 1.
            
            # Compute coverage counts
            cover_counts = np.sum(balls[list(sol_set)], axis=0)
            
            for i in list(sol_set):
                # Check if i is the sole cover for any word
                # Words covered by i: balls[i]
                # Among those, are any with cover_count == 1?
                # If yes, i is essential.
                # If no, i is redundant.
                i_covers = balls[i]
                # Check intersection of i_covers and (cover_counts == 1)
                if not np.any(i_covers & (cover_counts == 1)):
                    # Redundant
                    sol_set.remove(i)
                    improved = True
                    # Recalculate uncovered and counts
                    uncovered = get_uncovered(list(sol_set))
                    if sol_set:
                        cover_counts = np.sum(balls[list(sol_set)], axis=0)
                    else:
                        cover_counts = np.zeros(N, dtype=int)
                    break # Restart removal loop
            
            # 2. If not fully covered, add best word
            if np.any(uncovered):
                best_idx = greedy_add(uncovered, sol_set)
                if best_idx is not None:
                    sol_set.add(best_idx)
                    improved = True
                    uncovered = get_uncovered(list(sol_set))
                    cover_counts = np.sum(balls[list(sol_set)], axis=0)
        
        return list(sol_set)

    # Initialize solutions
    start_time = time.time()
    max_time = 115.0
    
    solutions = []
    
    # 1. Greedy Initialization (Vectorized)
    def greedy_init():
        uncovered = np.ones(N, dtype=bool)
        sol_set = set()
        while np.any(uncovered):
            best_idx = greedy_add(uncovered, sol_set)
            if best_idx is None: break
            sol_set.add(best_idx)
            uncovered = get_uncovered(list(sol_set))
        return local_search(list(sol_set))

    # Run multiple greedy inits with randomization? 
    # The greedy_add is deterministic. 
    # To randomize, we can shuffle the order of checking or add noise.
    # Let's just run a few random restarts with random initial sets.
    
    for _ in range(5):
        sol = greedy_init()
        solutions.append(sol)
        
    # Add some random repaired solutions
    for _ in range(5):
        k = random.randint(150, 250)
        random_sol = random.sample(range(N), k)
        sol_set = set(random_sol)
        uncovered = get_uncovered(list(sol_set))
        # Repair
        while np.any(uncovered):
            best_idx = greedy_add(uncovered, sol_set)
            if best_idx is None: break
            sol_set.add(best_idx)
            uncovered = get_uncovered(list(sol_set))
        sol = local_search(list(sol_set))
        solutions.append(sol)
        
    best_solution = min(solutions, key=len)
    
    # Main Loop: Local Search on Best
    # We can also try "Swap" moves: Replace one word with another.
    # But local_search already does add/remove.
    # Let's try random perturbations of the best solution.
    
    while time.time() - start_time < max_time:
        # Perturb best solution
        if random.random() < 0.5:
            # Random removal
            trial = best_solution[:]
            if trial:
                trial.pop(random.randint(0, len(trial)-1))
            trial = local_search(trial)
        else:
            # Random addition
            trial = best_solution[:]
            r = random.randint(0, N-1)
            if r not in trial:
                trial.append(r)
            trial = local_search(trial)
            
        if len(trial) < len(best_solution):
            best_solution = trial
            # If we found a better one, we can restart the perturbation process
            # Or just continue.
            
        # Occasionally, run a full local search on a random mix of top solutions
        # To save time, we'll stick to perturbing the best.
        
    return [all_words[i] for i in best_solution]
