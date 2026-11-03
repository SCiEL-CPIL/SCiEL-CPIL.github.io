import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Specific parameters for the task
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (not expected, but safe)
        # Simple greedy
        import itertools
        total = q ** n
        if total > 10000:
            return [] # Too large for simple fallback
        words = list(itertools.product(range(q), repeat=n))
        idx_map = {w: i for i, w in enumerate(words)}
        
        def dist(w1, w2):
            return sum(1 for a, b in zip(w1, w2) if a != b)
            
        covered = [False] * total
        code = []
        while not all(covered):
            best_i = -1
            best_new = 0
            for i in range(total):
                if covered[i]:
                    continue
                # Count new covers
                new_c = 0
                for j in range(total):
                    if not covered[j] and dist(words[i], words[j]) <= radius:
                        new_c += 1
                if new_c > best_new:
                    best_new = new_c
                    best_i = i
            if best_i == -1: break
            code.append(words[best_i])
            for j in range(total):
                if dist(words[best_i], words[j]) <= radius:
                    covered[j] = True
        return code

    # Specific optimization for q=3, n=8, radius=1
    N = 3 ** 8
    n_bits = 8
    
    # Precompute balls: ball[i] is list of indices within distance 1 of i
    # We can compute this efficiently.
    # Index is base-3 representation.
    # To get neighbors, we change one digit to another.
    
    # Helper to convert index to digits
    def idx_to_digits(idx):
        digits = [0] * n_bits
        for i in range(n_bits - 1, -1, -1):
            digits[i] = idx % 3
            idx //= 3
        return digits

    # Helper to convert digits to index
    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * 3 + d
        return idx

    balls = []
    for i in range(N):
        d = idx_to_digits(i)
        neighbors = [i]
        for pos in range(n_bits):
            orig = d[pos]
            for new_val in range(3):
                if new_val != orig:
                    d[pos] = new_val
                    neighbors.append(digits_to_idx(d))
            d[pos] = orig
        balls.append(neighbors)
        
    # Convert balls to numpy arrays for faster indexing if needed, 
    # but lists are fine for 17 elements.
    
    # Greedy Initialization
    covered_count = np.zeros(N, dtype=np.int32)
    code = []
    
    # To speed up greedy, we can just pick random uncovered words and add their "best" coverer?
    # Or standard greedy: pick word covering most uncovered.
    # N=6561, greedy is fast enough.
    
    # Optimization: Use a heap or just scan. Scanning 6561 * 17 * ~500 steps is ~55M ops. 
    # In Python this might be 5-10s. Acceptable.
    
    uncovered = np.ones(N, dtype=bool)
    while uncovered.any():
        best_w = -1
        max_new = 0
        # Sample a subset to speed up? No, let's do full scan first.
        # If it's too slow, we'll see.
        # Actually, we can just pick a random uncovered word u, and pick a random neighbor w of u.
        # This is a "randomized greedy" which is often good enough for init.
        # Let's try random greedy for speed.
        unc_idx = np.where(uncovered)[0]
        u = unc_idx[random.randint(0, len(unc_idx)-1)]
        # Pick a random neighbor of u
        w = balls[u][random.randint(0, len(balls[u])-1)]
        
        # Add w
        code.append(w)
        for v in balls[w]:
            if uncovered[v]:
                uncovered[v] = False
                covered_count[v] += 1 # Not strictly needed for greedy, but let's track
            else:
                covered_count[v] += 1
        
        # Check time
        if time.time() > deadline - 90:
            break

    # Ensure we have a valid code (if greedy stopped early due to time, we might have holes)
    # If holes exist, fill them greedily
    while uncovered.any() and time.time() < deadline - 80:
        unc_idx = np.where(uncovered)[0]
        u = unc_idx[0]
        # Pick a neighbor of u that covers most uncovered neighbors
        best_w = -1
        max_cov = 0
        for w in balls[u]:
            cov = 0
            for v in balls[w]:
                if uncovered[v]:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_w = w
        if best_w != -1:
            code.append(best_w)
            for v in balls[best_w]:
                if uncovered[v]:
                    uncovered[v] = False
                    covered_count[v] += 1
                else:
                    covered_count[v] += 1

    # Re-initialize covered_count properly from the code
    covered_count = np.zeros(N, dtype=np.int32)
    for c in code:
        covered_count[balls[c]] += 1
    
    U = np.sum(covered_count == 0)
    best_size = len(code)
    best_code = code[:]

    # Simulated Annealing
    # Objective: Minimize |C| + P * U
    # P should be large enough that any valid solution is preferred over invalid.
    # Max |C| is ~6561. Let P = 10000.
    P = 10000
    
    T = 1.0
    T_min = 1e-6
    cooling = 0.99995
    
    # We will run SA until time runs out
    # Move probabilities: Delete 0.4, Add 0.3, Swap 0.3
    
    code_set = set(code)
    
    while time.time() < deadline - 5:
        # Select move
        r = random.random()
        if r < 0.4:
            # Delete
            if len(code) == 0:
                continue
            c = random.choice(code)
            # Check if deletion is possible (always is, but might create holes)
            # Calculate delta U
            # Words that become uncovered are those with count 1 in ball[c]
            delta_U = 0
            for v in balls[c]:
                if covered_count[v] == 1:
                    delta_U += 1
            
            delta_E = -1 + P * delta_U
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Accept
                code.remove(c)
                code_set.remove(c)
                for v in balls[c]:
                    if covered_count[v] == 1:
                        # Becomes uncovered
                        pass # U increased by 1
                    covered_count[v] -= 1
                U += delta_U
                
                if U == 0 and len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    
        elif r < 0.7:
            # Add
            # Pick a random word, preferably near an uncovered word if any
            if U > 0:
                unc_idx = np.where(covered_count == 0)[0]
                u = unc_idx[random.randint(0, len(unc_idx)-1)]
                w = balls[u][random.randint(0, len(balls[u])-1)]
            else:
                w = random.randint(0, N - 1)
            
            if w in code_set:
                continue
                
            # Calculate delta U
            delta_U = 0
            for v in balls[w]:
                if covered_count[v] == 0:
                    delta_U -= 1 # Becomes covered
            
            delta_E = 1 + P * delta_U
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Accept
                code.append(w)
                code_set.add(w)
                for v in balls[w]:
                    if covered_count[v] == 0:
                        pass # U decreased by 1
                    covered_count[v] += 1
                U += delta_U
                
                if U == 0 and len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    
        else:
            # Swap
            if len(code) == 0:
                continue
            c = random.choice(code)
            # Pick new word w
            # Ideally w covers critical words of c
            # Critical words: v in ball[c] with count 1
            critical = [v for v in balls[c] if covered_count[v] == 1]
            
            if critical:
                # Pick a candidate w that covers at least one critical word
                # Or just pick a random neighbor of a critical word
                u = random.choice(critical)
                w = balls[u][random.randint(0, len(balls[u])-1)]
            else:
                w = random.randint(0, N - 1)
            
            if w in code_set:
                continue
                
            # Calculate delta U
            # Remove c: count[v] -= 1 for v in ball[c]
            # Add w: count[v] += 1 for v in ball[w]
            
            # We need to be careful with overlapping balls.
            # Let's simulate on a temporary copy of counts? No, too slow.
            # Let's calculate delta_U directly.
            
            # A word v changes coverage status if:
            # 1. It is in ball[c] but not ball[w]: count decreases by 1. If it was 1, it becomes 0 (U+1).
            # 2. It is in ball[w] but not ball[c]: count increases by 1. If it was 0, it becomes 1 (U-1).
            # 3. It is in both: count unchanged.
            # 4. It is in neither: unchanged.
            
            set_c = set(balls[c])
            set_w = set(balls[w])
            
            delta_U = 0
            # Words losing coverage (in C, not in W)
            for v in set_c - set_w:
                if covered_count[v] == 1:
                    delta_U += 1
            # Words gaining coverage (in W, not in C)
            for v in set_w - set_c:
                if covered_count[v] == 0:
                    delta_U -= 1
            
            delta_E = P * delta_U # Size unchanged
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Accept
                # Remove c
                code.remove(c)
                code_set.remove(c)
                for v in balls[c]:
                    covered_count[v] -= 1
                # Add w
                code.append(w)
                code_set.add(w)
                for v in balls[w]:
                    covered_count[v] += 1
                U += delta_U
                
                if U == 0 and len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]

        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: Restart with higher T if stuck? 
            # For now, just keep going.

    # Convert best_code to list of words
    result = []
    for c in best_code:
        result.append(idx_to_digits(c))
        
    return result
