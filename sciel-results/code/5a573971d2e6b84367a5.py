import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s limit

    total_words = q ** n
    
    # For the specific case q=3, n=6, radius=1, we can use a more efficient approach
    # But we need a general solution that works for any q, n, radius
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls (all words within radius of each word)
    # For radius=1, this is the word itself plus all words differing in exactly 1 position
    # For general radius, we need to check Hamming distance
    
    balls = [[] for _ in range(total_words)]
    
    # For radius=1, we can compute balls very efficiently
    if radius == 1:
        for i, w in enumerate(all_words):
            balls[i] = [i]  # Include self
            for pos in range(n):
                for new_val in range(q):
                    if new_val == w[pos]:
                        continue
                    new_word = list(w)
                    new_word[pos] = new_val
                    idx = word_to_idx[tuple(new_word)]
                    balls[i].append(idx)
    else:
        # General case: check Hamming distance
        for i, w in enumerate(all_words):
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    balls[i].append(j)
    
    # Helper to check if a set of codeword indices covers all points
    def is_covered(code_indices):
        covered = [0] * total_words
        for idx in code_indices:
            for nb in balls[idx]:
                covered[nb] += 1
        return all(c > 0 for c in covered)
    
    # Faster: use a coverage count array and check if all are > 0
    def get_coverage(code_indices):
        cov = [0] * total_words
        for idx in code_indices:
            for nb in balls[idx]:
                cov[nb] += 1
        return cov
    
    def count_uncovered(cov):
        return sum(1 for c in cov if c == 0)
    
    def is_valid_cov(cov):
        return all(c > 0 for c in cov)
    
    # Remove redundant codewords
    def minimize(code_indices):
        changed = True
        while changed:
            changed = False
            cov = get_coverage(code_indices)
            for i in range(len(code_indices)):
                # Try removing code_indices[i]
                test_cov = cov[:]
                for nb in balls[code_indices[i]]:
                    test_cov[nb] -= 1
                if is_valid_cov(test_cov):
                    code_indices = code_indices[:i] + code_indices[i+1:]
                    cov = test_cov
                    changed = True
                    break
        return code_indices
    
    # Greedy set cover
    def greedy_cover():
        uncovered = [1] * total_words
        code_indices = []
        while any(uncovered):
            best_idx = -1
            best_count = 0
            for i in range(total_words):
                count = sum(1 for nb in balls[i] if uncovered[nb])
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            code_indices.append(best_idx)
            for nb in balls[best_idx]:
                uncovered[nb] = 0
        return code_indices
    
    # Randomized greedy
    def randomized_greedy():
        uncovered = [1] * total_words
        code_indices = []
        while any(uncovered):
            counts = []
            for i in range(total_words):
                counts.append(sum(1 for nb in balls[i] if uncovered[nb]))
            max_count = max(counts) if counts else 0
            candidates = [i for i, c in enumerate(counts) if c == max_count and c > 0]
            if not candidates:
                break
            best_idx = random.choice(candidates)
            code_indices.append(best_idx)
            for nb in balls[best_idx]:
                uncovered[nb] = 0
        return code_indices
    
    # Local search with swap and minimize
    def local_search(initial_code, time_limit):
        current = minimize(initial_code[:])
        cov = get_coverage(current)
        current_set = set(current)
        best = current[:]
        best_size = len(best)
        
        while time.time() < time_limit:
            if len(current) == 0:
                break
            
            i = random.randint(0, len(current) - 1)
            old = current[i]
            new = random.randint(0, total_words - 1)
            
            if new in current_set:
                continue
            
            # Compute tentative coverage
            test_cov = cov[:]
            for nb in balls[old]:
                test_cov[nb] -= 1
            for nb in balls[new]:
                test_cov[nb] += 1
            
            if not is_valid_cov(test_cov):
                continue
            
            tent_code = current[:]
            tent_code[i] = new
            tent_code = minimize(tent_code)
            
            if len(tent_code) < len(current):
                current = tent_code
                cov = test_cov
                # Recompute cov after minimize
                cov = get_coverage(current)
                current_set = set(current)
                if len(current) < best_size:
                    best = current[:]
                    best_size = len(best)
            elif len(tent_code) == len(current) and random.random() < 0.05:
                # Plateau move
                current = tent_code
                cov = get_coverage(current)
                current_set = set(current)
        
        return best
    
    # Strategy 1: Greedy
    code1 = greedy_cover()
    code1 = minimize(code1)
    best_code = code1
    best_size = len(best_code)
    
    # Strategy 2: Randomized greedy (multiple runs)
    remaining_time = deadline - time.time()
    num_runs = max(1, int(remaining_time / 2.0))
    
    for run in range(num_runs):
        if time.time() > deadline - 5.0:
            break
        code = randomized_greedy()
        code = minimize(code)
        if len(code) < best_size:
            best_code = code
            best_size = len(best_code)
    
    # Strategy 3: Local search from best
    remaining = deadline - time.time()
    if remaining > 3.0:
        ls_result = local_search(best_code, deadline - 1.0)
        if len(ls_result) < best_size:
            best_code = ls_result
            best_size = len(best_code)
    
    # Final minimize
    best_code = minimize(best_code)
    
    # Convert to words
    return [list(all_words[idx]) for idx in best_code]
