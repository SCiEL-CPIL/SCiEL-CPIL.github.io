import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # Fallback for general case or if not the specific hard instance
    if not (q == 3 and n == 6 and radius == 1):
        # Use the seed logic for general fallback if possible
        # The seed logic works for q=3, n>=4.
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Trivial valid code: all words
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(6, 1)
    N = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres as numpy boolean matrix
    # spheres_arr[i, j] is True if word j is within radius of word i
    spheres_arr = np.zeros((N, N), dtype=bool)
    for i, w in enumerate(all_words):
        spheres_arr[i, i] = True
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                spheres_arr[i, j] = True

    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial code: Tetracode construction (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = tuple(h) + tuple(tail)
            initial_code_indices.append(word_to_idx[word])
    
    best_code_indices = initial_code_indices[:]
    best_size = len(best_code_indices)

    # Helper function to remove redundant codewords using coverage counts
    def reduce_code_with_counts(indices):
        # Initialize coverage counts
        counts = np.zeros(N, dtype=np.int32)
        for idx in indices:
            counts[spheres_arr[idx]] += 1
        
        current_indices = list(indices)
        changed = True
        while changed:
            changed = False
            # Iterate through a copy of indices
            for idx in current_indices:
                # Check if idx is redundant
                # If min(counts[sphere(idx)]) >= 2, then idx is redundant
                sphere_mask = spheres_arr[idx]
                min_count = np.min(counts[sphere_mask])
                if min_count >= 2:
                    # Remove idx
                    current_indices.remove(idx)
                    counts[sphere_mask] -= 1
                    changed = True
                    break # Restart loop as indices changed
        return current_indices

    # Local search loop
    # We start with the tetracode code and try to improve it
    current_indices = reduce_code_with_counts(best_code_indices)
    current_size = len(current_indices)
    
    if current_size < best_size:
        best_size = current_size
        best_code_indices = current_indices[:]

    rng = np.random.default_rng(42)
    
    # Time budget for local search
    # We will perform random swaps and deletions
    while time.time() < deadline:
        # Try a random move
        # Move type 0: Swap a codeword with a random other word
        # Move type 1: Add a random word, then reduce
        # Move type 2: Remove a random word (if valid), then try to add back better ones? No, just reduce.
        
        move_type = rng.integers(0, 3)
        trial_indices = current_indices[:]
        
        if move_type == 0 and len(current_indices) > 0:
            # Swap
            idx_to_remove = current_indices[rng.integers(0, len(current_indices))]
            new_idx = rng.integers(0, N)
            if new_idx not in current_indices:
                trial_indices.remove(idx_to_remove)
                trial_indices.append(new_idx)
        elif move_type == 1:
            # Add random
            new_idx = rng.integers(0, N)
            if new_idx not in current_indices:
                trial_indices.append(new_idx)
        elif move_type == 2:
            # Remove random (if valid)
            if len(current_indices) > 1:
                idx_to_remove = current_indices[rng.integers(0, len(current_indices))]
                trial_indices.remove(idx_to_remove)
                # Check if still valid
                # We can check validity quickly
                covered = np.zeros(N, dtype=bool)
                for idx in trial_indices:
                    covered |= spheres_arr[idx]
                if not np.all(covered):
                    # Invalid move, revert
                    trial_indices = current_indices[:]
        
        # Reduce the trial code
        reduced_trial = reduce_code_with_counts(trial_indices)
        
        # Check if valid
        if reduced_trial:
             covered = np.zeros(N, dtype=bool)
             for idx in reduced_trial:
                 covered |= spheres_arr[idx]
             if np.all(covered):
                 if len(reduced_trial) < best_size:
                     best_size = len(reduced_trial)
                     best_code_indices = reduced_trial[:]
                 # Accept the move if it's better or equal (to explore)
                 if len(reduced_trial) <= len(current_indices):
                     current_indices = reduced_trial
                 else:
                     # With some probability, accept worse moves to escape local minima (Simulated Annealing style)
                     # But since we want to minimize, and time is limited, let's stick to hill climbing mostly
                     # or use a simple random restart if stuck?
                     # For now, accept if equal size to diversify.
                     if len(reduced_trial) == len(current_indices) and rng.random() < 0.5:
                         current_indices = reduced_trial

    # Final verification and conversion
    if not best_code_indices:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(all_words[i]) for i in best_code_indices]
