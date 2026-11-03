import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    total = q ** n
    ball_size = 1 + n * (q - 1)

    rng = np.random.default_rng(12345)

    # Convert between int and word representations
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    def to_word(val):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Precompute all balls as numpy arrays of ints
    def get_ball(word_int):
        word = to_word(word_int)
        ball = [word_int]
        for pos in range(n):
            orig = word[pos]
            for new_val in range(q):
                if new_val != orig:
                    word[pos] = new_val
                    ball.append(to_int(word))
            word[pos] = orig
        return ball

    all_ints = list(range(total))
    balls = [get_ball(w) for w in all_ints]

    # Hamming distance between two int-encoded words
    def hamming_dist(a, b):
        word_a = to_word(a)
        word_b = to_word(b)
        return sum(1 for i in range(n) if word_a[i] != word_b[i])

    # Coverage check: does the set of codewords cover the entire space?
    def check_coverage(codewords):
        covered = [False] * total
        for cw in codewords:
            for x in balls[cw]:
                covered[x] = True
        return all(covered)

    def count_uncovered(codewords):
        covered = [False] * total
        for cw in codewords:
            for x in balls[cw]:
                covered[x] = True
        return sum(1 for x in range(total) if not covered[x])

    # Phase 1: Build a well-spread code using simulated annealing to maximize min distance
    # Start with a random set of size ~ total/ball_size (lower bound estimate)
    est_size = max(1, total // ball_size)
    
    # Try several initial sizes, starting near the theoretical lower bound
    best_solution = None
    best_score = float('inf')  # We want to minimize codewords

    # Start with a greedy spread initialization
    def greedy_spread_init(target_size):
        """Greedy: pick codewords to maximize minimum distance (farthest-point sampling)."""
        code = []
        # Start with a random word
        first = rng.integers(0, total)
        code.append(int(first))
        
        for _ in range(target_size - 1):
            if time.time() > deadline:
                break
            best_word = -1
            best_min_dist = -1
            for w in all_ints:
                if w in code:
                    continue
                # Compute minimum distance to existing code
                min_d = min(hamming_dist(w, c) for c in code)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_word = w
            if best_word >= 0:
                code.append(best_word)
        return code

    # Simulated annealing: try to replace a codeword with another to improve spread
    def simulated_annealing(code, max_iters=5000):
        """SA to maximize minimum pairwise distance."""
        if len(code) <= 1:
            return code
        
        current_code = list(code)
        current_min_dist = min(hamming_dist(a, b) for i, a in enumerate(current_code) for b in current_code[i+1:])
        best_code = list(current_code)
        best_min_dist = current_min_dist
        
        T = 1.0
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random codeword to replace
            idx = rng.integers(0, len(current_code))
            old_word = current_code[idx]
            
            # Try a random new word
            new_word = int(rng.integers(0, total))
            if new_word == old_word:
                continue
            
            # Check if new word is already in code
            if new_word in current_code:
                continue
            
            # Compute new min distance
            # Remove old, add new
            new_code = current_code[:idx] + [new_word] + current_code[idx+1:]
            
            # Compute min distance for new code
            new_min_d = float('inf')
            for i in range(len(new_code)):
                for j in range(i+1, len(new_code)):
                    d = hamming_dist(new_code[i], new_code[j])
                    if d < new_min_d:
                        new_min_d = d
                        if new_min_d < current_min_dist:
                            break
                if new_min_d < current_min_dist:
                    break
            
            delta = new_min_d - current_min_dist
            
            # Accept if better, or with probability exp(delta/T)
            if delta > 0 or (T > 0.01 and rng.random() < np.exp(delta / T)):
                current_code = new_code
                current_min_dist = new_min_d
                if current_min_dist > best_min_dist:
                    best_min_dist = current_min_dist
                    best_code = list(current_code)
            
            T *= cooling
        
        return best_code

    # Phase 2: Repair - add codewords to cover uncovered words
    def repair(code):
        """Add codewords to cover all uncovered words."""
        covered = [False] * total
        for cw in code:
            for x in balls[cw]:
                covered[x] = True
        
        uncovered = [x for x in range(total) if not covered[x]]
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Greedy: pick the codeword that covers the most uncovered words
            best_word = -1
            best_count = 0
            for w in all_ints:
                count = sum(1 for x in balls[w] if not covered[x])
                if count > best_count:
                    best_count = count
                    best_word = w
            
            if best_word < 0 or best_count == 0:
                break
            
            code.append(best_word)
            for x in balls[best_word]:
                if not covered[x]:
                    covered[x] = True
            
            # Update uncovered
            uncovered = [x for x in range(total) if not covered[x]]
        
        return code

    # Phase 3: Local search - try to remove redundant codewords
    def local_search_remove(code):
        """Try to remove codewords that don't reduce coverage."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            cov_count = [0] * total
            for cw in code:
                for x in balls[cw]:
                    cov_count[x] += 1
            
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                safe = True
                for x in balls[code[i]]:
                    if cov_count[x] == 1:
                        safe = False
                        break
                if safe:
                    for x in balls[code[i]]:
                        cov_count[x] -= 1
                    code.pop(i)
                    changed = True
                    break
        
        return code

    # Main loop: try different initial sizes and SA parameters
    for attempt in range(3):
        if time.time() > deadline:
            break
        
        # Estimate good size: start near lower bound, increase
        base_size = max(1, int(total / ball_size) + attempt * 10)
        
        # Greedy spread initialization
        code = greedy_spread_init(base_size)
        
        # Simulated annealing to improve spread
        code = simulated_annealing(code, max_iters=3000)
        
        # Repair to ensure full coverage
        code = repair(code)
        
        # Local search to remove redundant codewords
        code = local_search_remove(code)
        
        # Check if this solution is better
        if len(code) < best_score:
            best_score = len(code)
            best_solution = list(code)
        
        # If we've achieved a good solution, try to improve further
        if best_score <= 500:
            # Do more SA iterations on the best solution
            improved = simulated_annealing(best_solution, max_iters=5000)
            improved = repair(improved)
            improved = local_search_remove(improved)
            if len(improved) < best_score:
                best_score = len(improved)
                best_solution = list(improved)

    # Final safety: ensure we have a valid solution
    if best_solution is None or not check_coverage(best_solution):
        # Fallback: greedy set cover
        covered = [False] * total
        covered_count = 0
        codewords = []
        while covered_count < total and time.time() < deadline:
            best_word = -1
            best_new = 0
            for w in all_ints:
                new_cov = 0
                for x in balls[w]:
                    if not covered[x]:
                        new_cov += 1
                if new_cov > best_new:
                    best_new = new_cov
                    best_word = w
            if best_word < 0 or best_new == 0:
                break
            codewords.append(best_word)
            for x in balls[best_word]:
                if not covered[x]:
                    covered[x] = True
                    covered_count += 1
        best_solution = codewords

    # Convert to word representation
    result = [to_word(w) for w in best_solution]

    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)

    return unique_result
