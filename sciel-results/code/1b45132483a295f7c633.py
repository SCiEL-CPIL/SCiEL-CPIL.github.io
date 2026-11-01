import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n
    
    # Precompute balls
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
    
    # Convert balls to a list of sets for faster membership tests
    ball_sets = [set(row.tolist()) for row in balls]
    
    # Precompute for each word, the list of codewords whose ball contains it
    # word_to_codewords[w] = list of c such that w in ball(c)
    # This is the inverse of the ball membership
    word_to_codewords = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            word_to_codewords[w].append(c)
    
    # Convert to numpy arrays for speed where possible
    # But lists of lists might be fine for this size
    
    best_size = total
    best_set = None
    
    np.random.seed(42)
    random.seed(42)
    
    # We will use a greedy + local search approach
    # Greedy: pick codeword covering most uncovered words
    # Local search: try to remove redundant, or swap to enable removal
    
    while time.time() < deadline - 10.0:
        # 1. Greedy construction
        selected = []
        uncovered = np.ones(total, dtype=bool)
        
        while np.any(uncovered):
            # Count how many uncovered words each codeword covers
            # We can vectorize this: for each codeword c, count how many of its ball words are uncovered
            # counts[c] = sum(uncovered[balls[c]])
            # This is a matrix-vector product if we build a 0-1 matrix, but balls is int array
            # Let's use numpy indexing: uncovered[balls] gives (total, ball_size) boolean array
            # Sum along axis 1 gives counts per codeword
            # But this is 6561 * 17 = ~111k operations, which is fine
            counts = uncovered[balls].sum(axis=1)
            
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            # Pick a random codeword with max count to add diversity
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            uncovered[balls[best_idx]] = False
            
            if time.time() > deadline - 15.0:
                break
        
        if np.any(uncovered):
            continue
        
        selected = np.array(selected, dtype=np.int32)
        
        # 2. Pruning: remove redundant codewords
        # A codeword c is redundant if every word in its ball is covered by at least one other selected codeword
        # i.e., for all w in ball(c), there exists c' in selected, c' != c, such that w in ball(c')
        # Equivalently, the minimum cover count over w in ball(c) is > 1
        
        cover_count = np.zeros(total, dtype=np.int32)
        cover_count[balls[selected]] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            # For each selected codeword, find min cover count in its ball
            mins = np.min(cover_count[balls[selected]], axis=1)
            removable_mask = mins > 1
            
            if not np.any(removable_mask):
                break
            
            # Remove all removable codewords at once
            remove_indices = selected[removable_mask]
            selected = selected[~removable_mask]
            cover_count[balls[remove_indices]] -= 1
            changed = True
        
        # 3. Local search: try swaps
        # For each codeword c in selected, try swapping with a random c'
        # The swap is valid if after removing c and adding c', all words are still covered
        # i.e., every word in ball(c) that is not in ball(c') must be covered by some other codeword
        
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            if len(selected) == 0:
                break
            
            # Pick a random codeword to potentially swap
            c_idx = np.random.randint(len(selected))
            c = selected[c_idx]
            
            # Pick a random candidate to swap in
            c_prime = np.random.randint(total)
            if c_prime == c:
                continue
            
            # Check validity of swap
            # Words in ball(c) that are NOT in ball(c_prime) must have cover_count > 1
            # (since they will lose coverage from c, but not gain from c_prime)
            # Words in ball(c) that ARE in ball(c_prime) are safe
            
            # Efficient check:
            # unique_to_c = [w for w in ball(c) if w not in ball(c_prime)]
            # If any w in unique_to_c has cover_count[w] == 1, swap is invalid
            
            # Use the precomputed sets
            bc = ball_sets[c]
            bcp = ball_sets[c_prime]
            
            valid = True
            for w in bc:
                if w not in bcp:
                    if cover_count[w] == 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Perform swap
            cover_count[balls[c]] -= 1
            cover_count[balls[c_prime]] += 1
            selected[c_idx] = c_prime
            
            # Re-prune
            changed = True
            while changed and time.time() < deadline - 5.0:
                mins = np.min(cover_count[balls[selected]], axis=1)
                removable_mask = mins > 1
                if not np.any(removable_mask):
                    break
                remove_indices = selected[removable_mask]
                selected = selected[~removable_mask]
                cover_count[balls[remove_indices]] -= 1
                changed = True
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break
    
    if best_set is None:
        best_set = np.arange(total, dtype=np.int32)
    
    result = []
    for w in best_set:
        digits = []
        temp = int(w)
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
