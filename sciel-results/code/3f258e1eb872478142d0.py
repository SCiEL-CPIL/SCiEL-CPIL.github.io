import numpy as np
import itertools
import time

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N

# Generate all words
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)

# Precompute a mapping from word tuple to index
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute balls: for each codeword index, store the indices of words it covers
# ball_indices[c] is a list/array of word indices covered by codeword c
ball_indices = []
for i in range(NUM_WORDS):
    c = all_words[i]
    ball = [i]
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            ball.append(idx)
    ball_indices.append(np.array(ball, dtype=np.int32))

# Convert ball_indices to a 2D array for vectorized operations?
# Since balls have same size (1 + N*(Q-1) = 13), we can make it a 2D array.
BALL_SIZE = 1 + N * (Q - 1)
balls_2d = np.array(ball_indices, dtype=np.int32) # Shape (729, 13)

def get_initial_code():
    """
    Generate an initial code. 
    We use a greedy approach with randomization to get a diverse starting point,
    or use the tetracode construction (81 words) which is a known valid cover.
    Here we use a randomized greedy to potentially get a smaller starting point,
    or just use the tetracode as a safe baseline.
    """
    # Tetracode construction: 81 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        if q is None or n is None or radius is None:
            return []
        # Simple greedy for general case
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        nw = q ** n
        wti = {tuple(w): i for i, w in enumerate(all_w)}
        # Build ball indices
        balls = []
        for i in range(nw):
            c = all_w[i]
            ball = [i]
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    ball.append(wti[tuple(nb)])
            balls.append(np.array(ball, dtype=np.int32))
        
        covered = np.zeros(nw, dtype=bool)
        code = []
        while not np.all(covered):
            # Greedy: pick word covering most uncovered
            scores = np.array([np.sum(covered[balls[i]] == False) for i in range(nw)])
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            covered[balls[c]] = True
        return [list(all_w[c]) for c in code]

    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Start with tetracode (81 words)
    best_code = get_initial_code()
    best_size = len(best_code)
    
    # Convert best_code to list of indices for internal processing
    # We work with indices into all_words
    best_code_idx = [word_to_idx[tuple(w)] for w in best_code]
    
    # Helper: Calculate coverage counts vectorized
    def get_coverage(code_idx):
        """
        Returns a numpy array of coverage counts for each word.
        Uses the 2D balls array for speed.
        """
        if not code_idx:
            return np.zeros(NUM_WORDS, dtype=np.int32)
        # code_idx is a list of indices into all_words (0..728)
        # We want to sum the indicators of balls for each word.
        # balls_2d has shape (729, 13).
        # We can create a mask or use bincount.
        # Let's use the fact that balls are small.
        # Actually, we can just sum the rows of balls_2d corresponding to code_idx?
        # No, balls_2d[i] is the list of words covered by codeword i.
        # We need to count how many times each word index appears in the union of balls[code_idx].
        
        # Method: Create a large array of all covered word indices and bincount it.
        # code_idx length ~ 80. Ball size 13. Total elements ~ 1000.
        # This is very fast.
        if len(code_idx) == 0:
             return np.zeros(NUM_WORDS, dtype=np.int32)
        
        # Get the balls for the selected codewords
        # code_idx is a list of ints.
        selected_balls = balls_2d[code_idx] # Shape (K, 13)
        flat_indices = selected_balls.ravel()
        cov = np.bincount(flat_indices, minlength=NUM_WORDS).astype(np.int32)
        return cov

    # Helper: Reduce code by removing redundant words
    def reduce_code(code_idx):
        """
        Remove redundant codewords. Returns new list of indices.
        A codeword is redundant if removing it doesn't uncover any word.
        i.e., min(coverage[ball(c)]) > 1 for all c in code.
        """
        if not code_idx:
            return []
        
        code_list = list(code_idx)
        # Sort to have a deterministic order for reduction? 
        # Randomization might help escape local minima in reduction?
        # But reduction should be deterministic for a given set to define the "state" clearly.
        # Let's use random order to break symmetry if multiple are removable.
        
        changed = True
        while changed:
            changed = False
            if not code_list:
                break
            
            cov = get_coverage(code_list)
            
            # Find all removable codewords
            # For each c in code_list, check if min(cov[balls_2d[c]]) > 1
            # We can vectorize this check.
            
            # Get balls for current code
            # code_list is a list of indices.
            # We need to check each one.
            
            # Optimization: If we just check one at a time and remove, we recompute cov.
            # With vectorized cov, recomputation is cheap.
            
            # Try to remove one at a time to be safe and simple, but check all candidates first?
            # Let's just iterate and remove the first one found.
            
            # To make it faster, we can check all at once.
            # But code_list changes size.
            
            # Simple loop is fine if cov is fast.
            # Let's try to find *all* removable words in the current set simultaneously.
            # This might allow removing multiple at once if they don't overlap in "uniqueness".
            # But if c1 and c2 both uniquely cover word w, we can't remove both.
            # So sequential removal is safer.
            
            # However, we can speed up the check.
            # Let's just do sequential with fast cov.
            
            # Shuffle to avoid bias
            np.random.shuffle(code_list)
            
            for i in range(len(code_list)):
                c = code_list[i]
                # Check if c is redundant
                # cov is for the *current* code_list.
                # If we remove c, the new cov for words in ball(c) decreases by 1.
                # So we need min(cov[ball(c)]) > 1.
                
                ball_c = balls_2d[c]
                if np.min(cov[ball_c]) > 1:
                    code_list.pop(i)
                    changed = True
                    # Break to recompute cov for the new set?
                    # Or continue with current cov?
                    # If we continue with current cov, the cov for other words is still valid
                    # (it only decreased for ball(c), but we are checking min > 1).
                    # Wait, if we remove c, cov for ball(c) decreases.
                    # For other balls, cov is unchanged.
                    # So if we check another word d, its ball might intersect with c's ball.
                    # If d's ball intersected c's ball, the cov for those intersection words is now lower.
                    # So we MUST recompute cov after every removal if we want to be correct.
                    # Given the cost of get_coverage is low (bincount on ~1000 items),
                    # let's recompute.
                    break
            
        return code_list

    # Simulated Annealing
    def simulated_annealing(initial_code, max_iterations=5000, T_start=10.0, T_end=0.1):
        current_code = reduce_code(initial_code)
        current_cov = get_coverage(current_code)
        current_cost = len(current_code) # Cost is size
        
        best_code = current_code.copy()
        best_cost = current_cost
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max_iterations)
        
        in_current = set(current_code)
        
        for i in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Generate neighbor
            # Move: Replace one codeword with another
            # Or Add/Remove? 
            # Since we reduce, size is always minimal for the set.
            # So we should focus on swaps that might lead to a smaller reduced set.
            
            # Strategy: 
            # 1. Pick random index in current_code.
            # 2. Pick random new codeword.
            # 3. Form new set.
            # 4. Reduce new set.
            # 5. Compare size.
            
            if not current_code:
                break
                
            idx = np.random.randint(len(current_code))
            old_c = current_code[idx]
            
            new_c = np.random.randint(NUM_WORDS)
            if new_c in in_current:
                # Try a few times? Or just skip.
                continue
            
            # Create new code list
            new_code_list = current_code.copy()
            new_code_list[idx] = new_c
            
            # Reduce
            reduced_new_code = reduce_code(new_code_list)
            new_cost = len(reduced_new_code)
            
            # Acceptance criterion
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0:
                # Improvement
                current_code = reduced_new_code
                current_cov = get_coverage(current_code)
                current_cost = new_cost
                in_current = set(current_code)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_code = current_code.copy()
            else:
                # Worse or equal
                # Accept with probability exp(-delta/T)
                # If delta is 0, accept to explore?
                # Let's accept equal moves to wander the plateau.
                if delta_cost == 0:
                    # Randomly accept to explore
                    if np.random.rand() < 0.5:
                        current_code = reduced_new_code
                        current_cov = get_coverage(current_code)
                        in_current = set(current_code)
                else:
                    # Worse
                    prob = np.exp(-delta_cost / T)
                    if np.random.rand() < prob:
                        current_code = reduced_new_code
                        current_cov = get_coverage(current_code)
                        current_cost = new_cost
                        in_current = set(current_code)
            
            T *= cooling
            
        return best_code, best_cost

    # Main Loop
    # Run SA for a while
    # We can restart if we get stuck
    
    np.random.seed(42) # For reproducibility in debugging, remove for final? No, let it be random.
    
    best_size_sa = best_size
    best_code_sa = best_code_idx.copy()
    
    # Run multiple SA runs or one long one?
    # One long one with restarts if stuck.
    
    stagnation_counter = 0
    last_improvement_time = start_time
    
    while time.time() < deadline:
        # Run SA
        sa_code, sa_size = simulated_annealing(best_code_sa)
        
        if sa_size < best_size_sa:
            best_size_sa = sa_size
            best_code_sa = sa_code
            last_improvement_time = time.time()
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # If no improvement for a while, perturb the best solution and restart
        if stagnation_counter > 5:
            # Perturb best_code_sa
            perturbed = best_code_sa.copy()
            # Add a random word? Or swap?
            # Let's swap a few words to jump to a different basin.
            for _ in range(3):
                if perturbed:
                    idx = np.random.randint(len(perturbed))
                    new_c = np.random.randint(NUM_WORDS)
                    perturbed[idx] = new_c
            # Reduce it
            perturbed = reduce_code(perturbed)
            best_code_sa = perturbed
            stagnation_counter = 0

    # Convert best_code_sa back to list of lists
    final_code = [list(all_words[c]) for c in best_code_sa]
    return final_code
