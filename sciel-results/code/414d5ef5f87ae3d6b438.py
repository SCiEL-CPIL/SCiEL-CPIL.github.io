import numpy as np
import time
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute all words as integers
    total_words = q ** n
    all_words = np.array(list(range(total_words)), dtype=np.int32)
    
    # Precompute base powers
    base_powers = np.array([q ** i for i in range(n)], dtype=np.int32)
    
    def int_to_word(w):
        return (w // base_powers) % q
        
    def word_to_int(w):
        return np.dot(w, base_powers)
        
    def get_neighbors(w):
        w = np.asarray(w)
        neighbors = []
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val != orig:
                    if radius >= 1:
                        nw = w.copy()
                        nw[i] = new_val
                        neighbors.append(nw)
        return neighbors

    # 1. Initial Heuristic: Greedy Set Cover
    # We want to cover all_words.
    # Coverage matrix: M[c, w] = 1 if distance(c, w) <= radius
    # To save memory and time, we can compute this on the fly or in chunks, but for n=7, q=3, total=2187.
    # Matrix size: 2187 x 2187 = ~4.8M ints. If int8, ~5MB. If int32, ~20MB.
    # Let's try to build it.
    
    # However, building the full matrix for every step of SA is too slow.
    # We need a fast way to evaluate the "gain" of adding a codeword and "loss" of removing one.
    
    # Let's use a vectorized approach for the initial greedy phase.
    # dist_matrix: (N, N) where N = total_words
    # This is 2187x2187.
    
    # Compute all pairwise distances?
    # 2187 * 2187 * 7 operations ~ 33M ops. NumPy can do this in <1s.
    
    # Flatten words to matrix
    words_matrix = np.array([int_to_word(w) for w in range(total_words)], dtype=np.int32)
    
    # Compute distances
    # dist[i, j] = sum(words_matrix[i] != words_matrix[j])
    # This creates a huge temporary array if done naively.
    # Better:
    # diff = words_matrix[:, None, :] - words_matrix[None, :, :]
    # dist = np.sum(diff != 0, axis=2)
    # This is 2187 * 2187 * 7 integers. ~33M ints. ~132MB memory. Acceptable.
    
    diff = words_matrix[:, None, :] - words_matrix[None, :, :]
    dist_matrix = np.sum(diff != 0, axis=2).astype(np.int32)
    
    # Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    code = []
    
    while not covered.all():
        # Find word with max new coverage
        # new_coverage = sum(~covered & (dist_matrix[i, :] <= radius))
        # We can precompute the "ball" of each word
        # balls[i] is a boolean vector of length total_words
        
        # To save memory, we might just compute the sum for the best candidate.
        # But finding the best candidate requires checking all N.
        # Let's precompute the boolean balls once.
        # balls: (N, N) boolean. 2187*2187 bytes = ~5MB.
        
        if not hasattr(construct, '_balls_cache'):
            construct._balls_cache = {}
        if (q, n, radius) not in construct._balls_cache:
             balls = dist_matrix <= radius
             construct._balls_cache[(q, n, radius)] = balls
        else:
             balls = construct._balls_cache[(q, n, radius)]
             
        # Calculate gain for each candidate
        # gain[i] = np.sum(balls[i] & ~covered)
        # Vectorized:
        # We need to sum over axis 1 of (balls & ~covered[None, :])
        # This is a bit heavy if done every step.
        
        # Optimization:
        # Maintain a count of how many codewords cover each word? No, that's for removal.
        # For greedy add, we just need the number of *uncovered* words in the ball.
        
        # Let's try a simpler loop for greedy, it might be fast enough for ~200 steps.
        # Or vectorize the gain calculation.
        
        # gain = (balls & ~covered[None, :]).sum(axis=1)
        # This allocates a (N, N) bool array every iteration. 5MB * 200 iterations. OK.
        
        gain = (balls & ~covered[None, :]).sum(axis=1)
        
        if gain.max() == 0:
            break
            
        best_idx = np.argmax(gain)
        
        code.append(int(best_idx))
        covered[balls[best_idx]] = True
        
        if time.time() > deadline - 30: # Reserve 30s for SA
            break

    # Convert to list of lists
    code_list = [int_to_word(c).tolist() for c in code]
    
    if time.time() > deadline - 10:
        return code_list

    # 2. Simulated Annealing
    # State: set of codewords (indices into 0..N-1)
    # Energy: number of uncovered words. We want to minimize this.
    # Ideally 0.
    
    # We represent the code as a set of indices.
    # We also maintain `cover_count`: for each word w, how many codewords cover it.
    # If cover_count[w] == 0, it's uncovered.
    # Energy = sum(cover_count == 0)
    
    cover_count = np.zeros(total_words, dtype=np.int32)
    current_code_set = set(code)
    
    # Initialize cover_count
    for c in current_code_set:
        cover_count[balls[c]] += 1
        
    current_energy = np.sum(cover_count == 0)
    
    best_code_set = set(current_code_set)
    best_energy = current_energy
    
    # SA Parameters
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.995
    
    # Moves:
    # 1. Remove a random codeword
    # 2. Add a random codeword
    # 3. Swap a random codeword with another random word
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        # Choose move
        move_type = rng.random()
        
        # Snapshot for rollback
        old_code_set = set(current_code_set)
        old_cover_count = cover_count.copy()
        
        # Perform move
        if move_type < 0.4: # Remove
            if not current_code_set: continue
            c_to_remove = rng.choice(list(current_code_set))
            current_code_set.remove(c_to_remove)
            cover_count[balls[c_to_remove]] -= 1
            
        elif move_type < 0.7: # Add
            # Prefer adding words that cover many uncovered words?
            # For SA, random is better to explore.
            # But adding random words in a large space is inefficient.
            # Let's add a word that is "close" to some uncovered word?
            # Or just random.
            c_to_add = rng.integers(0, total_words)
            if c_to_add in current_code_set:
                continue
            current_code_set.add(c_to_add)
            cover_count[balls[c_to_add]] += 1
            
        else: # Swap
            if not current_code_set: continue
            c_to_remove = rng.choice(list(current_code_set))
            c_to_add = rng.integers(0, total_words)
            if c_to_add in current_code_set:
                continue
            current_code_set.remove(c_to_remove)
            current_code_set.add(c_to_add)
            
            cover_count[balls[c_to_remove]] -= 1
            cover_count[balls[c_to_add]] += 1
            
        new_energy = np.sum(cover_count == 0)
        
        # Acceptance
        delta_E = new_energy - current_energy
        if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_code_set = set(current_code_set)
        else:
            # Rollback
            current_code_set = old_code_set
            cover_count = old_cover_count
            
        T *= cooling_rate
        
    # Return best found
    best_code_list = [int_to_word(c).tolist() for c in best_code_set]
    return best_code_list
