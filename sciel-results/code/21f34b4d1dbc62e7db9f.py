import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback for other parameters
    if q ** n <= 10000:
        all_words = [list(c) for c in itertools.product(range(q), repeat=n)]
        uncovered = set(range(len(all_words)))
        codewords = []
        
        def hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
        
        while uncovered:
            if time.time() > deadline - 5.0:
                break
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
    
    # Precompute weights for base-3 digits
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls (neighborhoods of radius 1)
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
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
    
    # Precompute ball membership as a boolean matrix
    ball_membership = np.zeros((total, total), dtype=bool)
    for w in range(total):
        ball_membership[w, balls[w]] = True
    
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % q
            idx //= q
        return digits
    
    def latin_hypercube_init():
        """Create a diverse initial set using Latin hypercube-like sampling."""
        selected = []
        used = set()
        
        # First, add a structured set based on linear codes
        # Use a simple approach: take words with specific patterns
        # For better diversity, use a greedy packing approach
        
        # Start with a few well-spread words
        base_words = []
        for i in range(100):
            # Generate random words and check minimum distance
            candidate = np.random.randint(0, q, size=n)
            cand_idx = 0
            for d in candidate:
                cand_idx = cand_idx * q + d
            
            if cand_idx in used:
                continue
            
            # Check minimum distance to existing words
            min_dist = float('inf')
            for w in selected:
                dist = sum(1 for a, b in zip(candidate, idx_to_digits(w)) if a != b)
                min_dist = min(min_dist, dist)
            
            if min_dist >= 3 or len(selected) < 20:
                selected.append(cand_idx)
                used.add(cand_idx)
        
        return selected
    
    def greedy_cover(selected, deadline):
        """Complete the cover using greedy set cover."""
        if not selected:
            selected = []
        
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        uncovered_mask = (cover_count == 0)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break
            
            # Count coverage for each candidate among uncovered words
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break
            
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best to encourage diversity
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        
        return selected, cover_count
    
    def local_search(selected, cover_count, deadline):
        """Apply local search to minimize the number of codewords."""
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        
        # Swap optimization
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                candidate_mask = np.ones(total, dtype=bool)
                for cw in critical_words:
                    candidate_mask &= ball_membership[:, cw]
                
                valid_candidates = np.where(candidate_mask)[0]
                valid_candidates = valid_candidates[valid_candidates != c]
                
                if len(valid_candidates) > 0:
                    c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved = True
                else:
                    new_selected.append(c)
            
            selected = new_selected
        
        # 2-Exchange: Remove 2, Add 1
        if len(selected) > 450 and time.time() < deadline - 8.0:
            improved_2ex = True
            while improved_2ex and time.time() < deadline - 5.0:
                improved_2ex = False
                new_selected = list(selected)
                
                indices = list(range(len(new_selected)))
                np.random.shuffle(indices)
                
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        if time.time() > deadline - 5.0:
                            break
                        
                        c1 = new_selected[indices[i]]
                        c2 = new_selected[indices[j]]
                        
                        words_c1 = balls[c1]
                        words_c2 = balls[c2]
                        
                        to_cover = set()
                        for w in words_c1:
                            if cover_count[w] == 1:
                                to_cover.add(w)
                        for w in words_c2:
                            if cover_count[w] == 1:
                                to_cover.add(w)
                        for w in words_c1:
                            if w in words_c2 and cover_count[w] == 2:
                                to_cover.add(w)
                        
                        if not to_cover:
                            cover_count[balls[c1]] -= 1
                            cover_count[balls[c2]] -= 1
                            new_selected = [c for c in new_selected if c != c1 and c != c2]
                            improved_2ex = True
                            break
                        
                        candidate_mask = np.ones(total, dtype=bool)
                        for w in to_cover:
                            candidate_mask &= ball_membership[:, w]
                        
                        valid_candidates = np.where(candidate_mask)[0]
                        valid_candidates = valid_candidates[valid_candidates != c1]
                        valid_candidates = valid_candidates[valid_candidates != c2]
                        
                        if len(valid_candidates) > 0:
                            c3 = valid_candidates[np.random.randint(len(valid_candidates))]
                            cover_count[balls[c1]] -= 1
                            cover_count[balls[c2]] -= 1
                            cover_count[balls[c3]] += 1
                            new_selected = [c for c in new_selected if c != c1 and c != c2]
                            new_selected.append(c3)
                            improved_2ex = True
                            break
                    
                    if improved_2ex:
                        break
                if improved_2ex:
                    selected = new_selected
        
        return selected, cover_count
    
    best_size = total
    best_set = None
    
    # Multiple restarts with different initialization strategies
    for iteration in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Different initialization strategies
        if iteration % 3 == 0:
            # Latin hypercube-like init
            selected = latin_hypercube_init()
        elif iteration % 3 == 1:
            # Pure random init with small size
            selected = list(np.random.choice(total, size=500, replace=False))
        else:
            # Structured init based on linear patterns
            selected = []
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        word = [a, b, c, a, b, c, a, b]
                        idx = 0
                        for d in word:
                            idx = idx * q + d
                        selected.append(idx)
            # Add more random words
            for _ in range(200):
                idx = np.random.randint(0, total)
                if idx not in selected:
                    selected.append(idx)
        
        # Complete the cover
        selected, cover_count = greedy_cover(selected, deadline)
        
        if np.any(cover_count == 0):
            continue  # Invalid due to timeout
        
        # Local search
        selected, cover_count = local_search(selected, cover_count, deadline)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected[:]
        
        if best_size <= 486:
            break
    
    if best_set is None:
        # Fallback: pure greedy
        selected, cover_count = greedy_cover([], deadline)
        best_set = selected
    
    result = [idx_to_digits(cw) for cw in best_set]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    
    return unique_result
