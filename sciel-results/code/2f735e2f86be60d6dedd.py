import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # This function is expected to be called with q=3, n=7, radius=1
    # But we should be somewhat general or at least robust to the specific task parameters.
    # The task says: "the evaluator passes this task's alphabet size q, length n and covering radius."
    # So we must handle general q, n, radius, but optimize for the specific case if possible?
    # The prompt says: "Keep the function names and signatures of the seed program."
    # The seed program hardcodes logic for q=3, n=7.
    # If I write a general solver, it might be slower.
    # However, the task is specifically "Football pool problem for 7 matches: ternary covering code K_3(7,1)".
    # The contract says: "Interface: construct(q, n, radius) -> list of codewords".
    # It is safer to assume q=3, n=7, radius=1, but I should probably check and fall back to a simple construction if parameters differ,
    # or just implement the greedy for the specific case and assume the evaluator only tests this case.
    # Given the "120s" limit and the specific nature of the task, I will assume q=3, n=7, radius=1.
    # If q, n, radius are different, I will return a simple valid code (like the seed or random) to ensure validity.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple valid construction for other parameters
        # Simple construction: all words? No, that's too big.
        # For small n, we can just return all words if n is very small.
        # Or use the tetracode logic if applicable.
        # For safety, let's just return a random set that is likely to cover, or the seed logic if it fits.
        # Actually, the seed logic only works for n>=4.
        # Let's just return the first q^n words? No, must be valid.
        # If parameters don't match, return a trivial valid code: all words.
        # This is valid but score will be bad. But the task is specific.
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3**7
    
    # Precompute all words as tuples of digits for ball calculation
    # Index i corresponds to the word with digits derived from base-3 representation of i
    # d_0 is least significant, d_{n-1} most significant
    def idx_to_digits(idx):
        digits = []
        for _ in range(n):
            digits.append(idx % 3)
            idx //= 3
        return digits

    words = [idx_to_digits(i) for i in range(N)]
    
    # Precompute ball masks as a boolean matrix
    # ball_matrix[i] is a boolean vector of length N, True if word j is in ball of word i
    ball_matrix = np.zeros((N, N), dtype=bool)
    
    for i in range(N):
        w = words[i]
        # Self
        ball_matrix[i, i] = True
        # Neighbors
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                # Convert new_w back to index
                new_idx = 0
                power = 1
                for d in new_w:
                    new_idx += d * power
                    power *= 3
                ball_matrix[i, new_idx] = True

    # Greedy Set Cover with NumPy vectorization
    def greedy_randomized():
        covered = np.zeros(N, dtype=bool)
        code_indices = []
        
        # We can randomize the tie-breaking by adding small random noise to gains?
        # Or just shuffle the initial order?
        # In vectorized max, ties are broken by the first index.
        # To randomize, we can add a small random epsilon to the gains.
        # Or simply, since we run many restarts, the specific tie-breaking matters less.
        # But to be safe, let's add a tiny random noise to break ties.
        
        while not covered.all():
            # Calculate gains: number of new elements covered by each candidate
            # ~covered is the set of uncovered elements
            # ball_matrix[:, ~covered] gives the intersection of each ball with uncovered set
            # .sum(axis=1) gives the count
            gains = ball_matrix[:, ~covered].sum(axis=1)
            
            # Add small random noise to break ties
            # The noise should be small enough not to change the ordering of distinct gains
            # but large enough to randomize ties.
            # Gains are integers. Max gain is 15.
            # Noise in (0, 1) is sufficient.
            noise = np.random.random(N) * 0.1
            gains_with_noise = gains + noise
            
            best_idx = np.argmax(gains_with_noise)
            
            # If best gain is 0, we have a problem (shouldn't happen if covered is not all True)
            if gains[best_idx] == 0:
                break # Should not happen if logic is correct, but safety check
                
            code_indices.append(best_idx)
            covered |= ball_matrix[best_idx]
            
        return code_indices

    # Run multiple restarts
    best_code = None
    best_len = float('inf')
    
    # Get a baseline from seed logic to ensure we have a valid code if greedy fails or is slow
    # Seed logic: tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    best_code = seed_code
    best_len = len(seed_code)
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    run_count = 0
    while time.time() < deadline:
        code_indices = greedy_randomized()
        if len(code_indices) < best_len:
            best_len = len(code_indices)
            # Convert indices back to words
            best_code = [words[i] for i in code_indices]
            # Optional: If we hit 186, we can stop or try to break it.
            # 186 is the known best. If we find 186, we can continue to see if we can get 185.
            # But 156 is lower bound.
            # If we find 186, it's a great score.
            # Let's keep going to try for better.
        
        run_count += 1
        
    return best_code
