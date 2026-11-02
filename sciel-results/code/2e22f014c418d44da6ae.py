import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s margin
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallback for other parameters
    return _general_greedy(q, n, radius, deadline)

def _general_greedy(q, n, radius, deadline):
    total = q ** n
    if total <= 10000:
        all_words = [list(c) for c in itertools.product(range(q), repeat=n)]
        uncovered = set(range(len(all_words)))
        codewords = []
        
        def hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
        
        while uncovered and time.time() < deadline - 5.0:
            best_idx = -1
            best_count = -1
            for i in range(len(all_words)):
                count = sum(1 for j in uncovered if hamming_dist(all_words[i], all_words[j]) <= radius)
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            codewords.append(all_words[best_idx])
            uncovered = {j for j in uncovered if hamming_dist(all_words[best_idx], all_words[j]) > radius}
        
        return codewords
    else:
        return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute base-3 weights
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls: balls[w, 0..16] are the 17 words within Hamming distance 1 of w
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
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # Build inverse mapping: for each word w, list of codewords that cover it
    # cov_by[w] = list of c such that w in balls[c]
    # Since balls is symmetric (w in balls[c] iff c in balls[w]), we can use balls directly
    
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % q
            idx //= q
        return digits
    
    best_size = total
    best_solution = None
    
    # Try multiple initialization strategies
    for iteration in range(30):
        if time.time() > deadline - 15.0:
            break
        
        rng_seed = iteration * 7919 + 12345
        random.seed(rng_seed)
        np.random.seed(rng_seed % 2**32)
        
        # Strategy 1: Structured initialization using known code structure
        # The optimal code for K_3(8,1) with 486 words can be constructed using
        # a specific algebraic structure. We'll use a greedy approach with
        # different randomization seeds to find good solutions.
        
        if iteration < 10:
            # Greedy with random tie-breaking
            selected = _greedy_init(balls, total, ball_size, rng_seed, deadline)
        elif iteration < 20:
            # Structured: start with a subset of a known good pattern
            selected = _structured_init(balls, total, ball_size, rng_seed, deadline)
        else:
            # Hybrid: random + greedy completion
            selected = _hybrid_init(balls, total, ball_size, rng_seed, deadline)
        
        # Local search to improve
        selected = _local_search(balls, total, ball_size, selected, deadline)
        
        # Final cleanup: remove redundant codewords
        selected = _remove_redundant(balls, total, ball_size, selected)
        
        # Verify coverage
        if _verify_coverage(balls, total, ball_size, selected):
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = selected[:]
        
        # Early termination if we hit the known optimum
        if best_size <= 486:
            break
    
    if best_solution is None:
        # Fallback: return all words (trivially valid)
        best_solution = list(range(total))
    
    # Convert to digit lists
    result = []
    for w in best_solution:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result

def _greedy_init(balls, total, ball_size, seed, deadline):
    """Standard greedy set cover with random tie-breaking."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    
    cov = np.zeros(total, dtype=np.int32)
    selected = []
    uncovered_count = total
    
    while uncovered_count > 0 and time.time() < deadline - 20.0:
        # Count how many uncovered words each candidate covers
        uncovered_mask = (cov == 0)
        counts = np.sum(uncovered_mask[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[random.randint(0, len(candidates) - 1)]
        
        selected.append(best_idx)
        cov[balls[best_idx]] += 1
        uncovered_count = np.sum(cov == 0)
    
    return selected

def _structured_init(balls, total, ball_size, seed, deadline):
    """Initialize with a structured pattern, then complete with greedy."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    
    # Create a structured initial set: pick codewords that are well-spread
    # Use a Latin hypercube-like approach
    n = 8
    q = 3
    
    # Generate random permutations for each position
    perms = [np.random.permutation(q) for _ in range(n)]
    
    # Create a grid of codewords
    selected = []
    used = set()
    
    # Add a diverse set of initial codewords
    num_initial = 150
    for i in range(num_initial):
        # Generate a random codeword with some structure
        digits = [int(perms[pos][random.randint(0, q-1)]) for pos in range(n)]
        idx = 0
        for d in digits:
            idx = idx * q + d
        if idx not in used:
            used.add(idx)
            selected.append(idx)
    
    # Complete with greedy
    cov = np.zeros(total, dtype=np.int32)
    for c in selected:
        cov[balls[c]] += 1
    
    uncovered_count = np.sum(cov == 0)
    
    while uncovered_count > 0 and time.time() < deadline - 20.0:
        uncovered_mask = (cov == 0)
        counts = np.sum(uncovered_mask[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        
        candidates = np.where(counts == max_count)[0]
        # Filter to only consider non-selected candidates
        cand_list = [c for c in candidates if c not in used]
        if not cand_list:
            cand_list = candidates.tolist()
        
        best_idx = cand_list[random.randint(0, len(cand_list) - 1)]
        
        selected.append(best_idx)
        used.add(best_idx)
        cov[balls[best_idx]] += 1
        uncovered_count = np.sum(cov == 0)
    
    return selected

def _hybrid_init(balls, total, ball_size, seed, deadline):
    """Hybrid: start with random subset, then greedy complete."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    
    # Start with a random subset
    num_initial = random.randint(100, 200)
    initial_set = random.sample(range(total), num_initial)
    selected = initial_set[:]
    used = set(initial_set)
    
    # Complete with greedy
    cov = np.zeros(total, dtype=np.int32)
    for c in selected:
        cov[balls[c]] += 1
    
    uncovered_count = np.sum(cov == 0)
    
    while uncovered_count > 0 and time.time() < deadline - 20.0:
        uncovered_mask = (cov == 0)
        counts = np.sum(uncovered_mask[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        
        candidates = np.where(counts == max_count)[0]
        cand_list = [c for c in candidates if c not in used]
        if not cand_list:
            cand_list = candidates.tolist()
        
        best_idx = cand_list[random.randint(0, len(cand_list) - 1)]
        
        selected.append(best_idx)
        used.add(best_idx)
        cov[balls[best_idx]] += 1
        uncovered_count = np.sum(cov == 0)
    
    return selected

def _local_search(balls, total, ball_size, selected, deadline):
    """Iterative local search: removal, single swap, pair swap."""
    in_code = set(selected)
    cov = np.zeros(total, dtype=np.int32)
    for c in selected:
        cov[balls[c]] += 1
    
    improved = True
    max_iterations = 500
    
    for it in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
        if not improved:
            break
        improved = False
        
        # Phase 1: Remove redundant codewords
        for c in list(selected):
            if c not in in_code:
                continue
            if np.min(cov[balls[c]]) > 1:
                # Remove c
                cov[balls[c]] -= 1
                in_code.remove(c)
                selected.remove(c)
                improved = True
        
        if not improved:
            # Phase 2: Try single swaps
            swapped = False
            for c in list(selected):
                if time.time() > deadline - 10.0:
                    break
                if c not in in_code:
                    continue
                
                # Find words uniquely covered by c
                unique_mask = (cov[balls[c]] == 1)
                unique_words = balls[c][unique_mask]
                
                if len(unique_words) == 0:
                    continue
                
                # Find candidates w that cover all unique_words
                # w must be in the intersection of balls[unique_words]
                if len(unique_words) > 5:
                    # Too many unique words, skip this swap
                    continue
                
                valid_w_set = set(balls[unique_words[0]].tolist())
                for v in unique_words[1:]:
                    valid_w_set.intersection_update(balls[v].tolist())
                
                # Filter out words already in code
                w_list = [w for w in valid_w_set if w not in in_code]
                if not w_list:
                    continue
                
                # Evaluate candidates: score = number of words with cov > 1 that w covers
                best_w = -1
                best_score = -1
                for w in w_list[:20]:  # Limit candidates
                    score = np.sum(cov[balls[w]] > 1)
                    if score > best_score:
                        best_score = score
                        best_w = w
                
                if best_w != -1:
                    # Perform swap
                    cov[balls[c]] -= 1
                    in_code.remove(c)
                    selected.remove(c)
                    
                    cov[balls[best_w]] += 1
                    in_code.add(best_w)
                    selected.append(best_w)
                    
                    improved = True
                    swapped = True
                    break  # Restart loop to re-evaluate
            
            if swapped:
                continue
        
        # Phase 3: Try pair swaps (limited)
        if not improved and len(selected) >= 2 and time.time() < deadline - 5.0:
            pair_swapped = False
            n_selected = len(selected)
            max_tries = 10
            
            for _ in range(max_tries):
                if time.time() > deadline - 5.0:
                    break
                
                i = random.randint(0, n_selected - 1)
                j = random.randint(0, n_selected - 1)
                if i == j:
                    continue
                c1 = selected[i]
                c2 = selected[j]
                if c1 not in in_code or c2 not in in_code:
                    continue
                
                # Temporarily remove c1, c2
                cov[balls[c1]] -= 1
                cov[balls[c2]] -= 1
                in_code.remove(c1)
                in_code.remove(c2)
                selected.remove(c1)
                selected.remove(c2)
                
                # Find uncovered words
                uncovered_words = np.where(cov == 0)[0]
                if len(uncovered_words) == 0:
                    # Just re-add
                    cov[balls[c1]] += 1
                    cov[balls[c2]] += 1
                    in_code.add(c1)
                    in_code.add(c2)
                    selected.append(c1)
                    selected.append(c2)
                    continue
                
                # Try to find w1, w2 that cover uncovered_words
                best_pair = None
                best_score = -1
                
                # Candidates for w1: words that cover at least one uncovered word
                cand_w1 = set()
                for v in uncovered_words[:10]:  # Limit
                    cand_w1.update(balls[v].tolist())
                cand_w1 = [w for w in cand_w1 if w not in in_code]
                
                if not cand_w1:
                    cov[balls[c1]] += 1
                    cov[balls[c2]] += 1
                    in_code.add(c1)
                    in_code.add(c2)
                    selected.append(c1)
                    selected.append(c2)
                    continue
                
                for w1 in cand_w1[:15]:  # Limit
                    covered1 = set(balls[w1].tolist())
                    remaining = [v for v in uncovered_words if v not in covered1]
                    
                    if not remaining:
                        # w1 covers all uncovered. Need w2 for redundancy benefit.
                        all_w = [w for w in range(total) if w not in in_code and w != w1]
                        if not all_w:
                            continue
                        max_red = -1
                        best_w2 = all_w[0]
                        for w2 in all_w[:10]:
                            red = np.sum(cov[balls[w2]] > 1)
                            if red > max_red:
                                max_red = red
                                best_w2 = w2
                        score = np.sum(cov[balls[w1]] > 1) + max_red
                        if score > best_score:
                            best_score = score
                            best_pair = (w1, best_w2)
                    else:
                        # Find w2 in intersection of balls[remaining]
                        if len(remaining) > 5:
                            continue
                        valid_w2 = set(balls[remaining[0]].tolist())
                        for v in remaining[1:]:
                            valid_w2.intersection_update(balls[v].tolist())
                        valid_w2 = [w for w in valid_w2 if w not in in_code and w != w1]
                        if not valid_w2:
                            continue
                        for w2 in valid_w2[:5]:
                            score = np.sum(cov[balls[w1]] > 1) + np.sum(cov[balls[w2]] > 1)
                            if score > best_score:
                                best_score = score
                                best_pair = (w1, w2)
                
                if best_pair:
                    w1, w2 = best_pair
                    cov[balls[w1]] += 1
                    cov[balls[w2]] += 1
                    in_code.add(w1)
                    in_code.add(w2)
                    selected.append(w1)
                    selected.append(w2)
                    improved = True
                    pair_swapped = True
                    break
                else:
                    # Revert
                    cov[balls[c1]] += 1
                    cov[balls[c2]] += 1
                    in_code.add(c1)
                    in_code.add(c2)
                    selected.append(c1)
                    selected.append(c2)
        
        if not improved:
            break
    
    return selected

def _remove_redundant(balls, total, ball_size, selected):
    """Remove any codeword whose ball is entirely covered by others."""
    in_code = set(selected)
    cov = np.zeros(total, dtype=np.int32)
    for c in selected:
        cov[balls[c]] += 1
    
    to_remove = []
    for c in selected:
        if np.min(cov[balls[c]]) > 1:
            to_remove.append(c)
    
    for c in to_remove:
        cov[balls[c]] -= 1
        in_code.remove(c)
        selected.remove(c)
    
    return selected

def _verify_coverage(balls, total, ball_size, selected):
    """Verify that every word is covered by at least one selected codeword."""
    cov = np.zeros(total, dtype=np.int32)
    for c in selected:
        cov[balls[c]] += 1
    return np.all(cov > 0)
