import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback for other parameters
    import itertools
    total = q ** n
    if total <= 10000:
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
                count = 0
                for j in uncovered:
                    if hamming_dist(all_words[i], all_words[j]) <= radius:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            codewords.append(all_words[best_idx])
            new_uncovered = set()
            for j in uncovered:
                if hamming_dist(all_words[best_idx], all_words[j]) > radius:
                    new_uncovered.add(j)
            uncovered = new_uncovered

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

    # Precompute ball_membership: ball_membership[w, v] = True if v in balls[w]
    # Equivalently, ball_membership[w, c] = True if c covers w (i.e., w in balls[c])
    # By symmetry of Hamming distance: w in balls[c] iff c in balls[w]
    # So ball_membership[w, c] = (c in balls[w])
    # We can build it as: for each w, for each c in balls[w], set ball_membership[w, c] = True
    ball_membership = np.zeros((total, total), dtype=np.int8)
    for w in range(total):
        ball_membership[w, balls[w]] = 1

    # Also build the inverse: for each codeword c, the list of words it covers is balls[c]
    # So we can use balls directly.

    best_size = total
    best_set = None

    # Helper: convert index to digits
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % q
            idx //= q
        return digits

    def digits_to_idx(digits):
        idx = 0
        for i in range(n):
            idx = idx * q + digits[i]
        return idx

    # Strategy: Latin Hypercube Sampling + Dual-Guided Set Cover + Local Search
    #
    # 1. LHS initialization: select a diverse set of codewords by ensuring each position
    #    has a balanced distribution of digit values, then refine with greedy.
    # 2. Dual-guided greedy: use shadow prices to prioritize hard-to-cover words.
    # 3. Local search: removal, swap, 2-exchange.

    for iteration in range(50):
        if time.time() > deadline - 10.0:
            break

        np.random.seed(iteration * 1000 + 42)

        # Phase 1: Latin Hypercube Sampling for initial diverse set
        # For n=8 positions, q=3, we want to select m codewords such that
        # in each position, each digit 0,1,2 appears roughly m/3 times.
        # Start with a reasonable m, then add more via greedy to cover all.
        
        # Use a structured approach: create a "Latin square" like structure
        # For 8 positions and 3 digits, we can create 3^k codewords where
        # in each position, digits are balanced.
        
        # Actually, let's use a simpler diverse initialization:
        # Select m random codewords, but enforce diversity by using
        # a greedy diversity maximization: pick codeword that maximizes
        # min Hamming distance to already selected set.
        
        m_init = 200  # Start with 200 codewords
        
        selected = []
        if m_init > 0:
            # First codeword: random
            first = np.random.randint(total)
            selected.append(first)
            
            # For subsequent codewords, pick one that maximizes min distance to selected
            # To make this fast, we'll use a sampling approach
            for _ in range(m_init - 1):
                if time.time() > deadline - 10.0:
                    break
                # Sample a few candidates and pick the one with best diversity
                best_cand = -1
                best_min_dist = -1
                for _ in range(50):
                    cand = np.random.randint(total)
                    if cand in selected:
                        continue
                    # Compute min Hamming distance to all selected
                    min_d = ball_size + 1
                    for s in selected:
                        # Hamming distance via balls: if s in balls[cand] then dist <= 1
                        # But we need exact distance up to some bound
                        # For diversity, we want large distance. Let's compute via digits.
                        d = 0
                        for pos in range(n):
                            dw = weights[pos]
                            dc = (cand // dw) % q
                            ds = (s // dw) % q
                            if dc != ds:
                                d += 1
                                if d >= min_d:
                                    break
                        if d > best_min_dist:
                            best_min_dist = d
                            best_cand = cand
                if best_cand != -1:
                    selected.append(best_cand)
        
        # Phase 2: Dual-guided greedy set cover to fill gaps
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        uncovered_mask = (cover_count == 0)
        
        if np.any(uncovered_mask):
            # Dual-guided greedy:
            # hardness[w] = 1 / (number of codewords that cover w)
            # For uncovered words, hardness is infinity, but we handle them first.
            # For covered words, we might still want to "reinforce" them if they have low coverage.
            # The dual variable for word w is 1 / cover_count[w] (high when few coverers).
            # Score for candidate c = sum_{w in balls[c]} hardness[w]
            
            while np.any(uncovered_mask):
                if time.time() > deadline - 10.0:
                    break
                
                # Compute hardness for all words
                # For words with cover_count 0, hardness = large (treat as 1.0 for now, 
                # but we must prioritize them)
                hardness = np.zeros(total, dtype=np.float64)
                nonzero = cover_count > 0
                hardness[nonzero] = 1.0 / cover_count[nonzero].astype(np.float64)
                hardness[~nonzero] = 1.0  # Uncovered words get priority weight 1.0
                
                # Score for each candidate c: sum of hardness over balls[c]
                scores = hardness[balls].sum(axis=1)
                
                # But we must ensure uncovered words get covered.
                # Add a large bonus for candidates that cover uncovered words.
                uncov_in_ball = uncovered_mask[balls].sum(axis=1)
                scores += uncov_in_ball * 100.0
                
                max_score = np.max(scores)
                candidates = np.where(scores == max_score)[0]
                
                # Random tie-breaking for diversity
                best_idx = candidates[np.random.randint(len(candidates))]
                
                selected.append(best_idx)
                cover_count[balls[best_idx]] += 1
                uncovered_mask = (cover_count == 0)
        
        if not np.any(uncovered_mask):
            # Valid solution, run local search
            # Phase 3: Local Search
            
            # 3a: Remove redundant codewords
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
            
            # 3b: Swap phase
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                new_selected = []
                for c in selected:
                    # Try to remove
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        improved = True
                        continue
                    
                    # Critical words: covered only by c
                    critical_mask = (cover_count[balls[c]] == 1)
                    critical_words = balls[c][critical_mask]
                    
                    if len(critical_words) == 0:
                        cover_count[balls[c]] -= 1
                        improved = True
                        continue
                    
                    # Find candidates c' that cover all critical words
                    candidate_mask = np.ones(total, dtype=bool)
                    for cw in critical_words:
                        candidate_mask &= (ball_membership[:, cw] == 1)
                    
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
            
            # 3c: 2-Exchange (Remove 2, Add 1)
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
                            
                            # Find words that become uncovered if we remove c1 and c2
                            to_cover = set()
                            for w in words_c1:
                                if cover_count[w] == 1:
                                    to_cover.add(w)
                            for w in words_c2:
                                if cover_count[w] == 1:
                                    to_cover.add(w)
                            # Intersection words with count 2
                            set_c1 = set(words_c1.tolist())
                            set_c2 = set(words_c2.tolist())
                            inter = set_c1 & set_c2
                            for w in inter:
                                if cover_count[w] == 2:
                                    to_cover.add(w)
                            
                            if not to_cover:
                                # Both are redundant
                                cover_count[balls[c1]] -= 1
                                cover_count[balls[c2]] -= 1
                                new_selected = [c for c in new_selected if c != c1 and c != c2]
                                improved_2ex = True
                                break
                            
                            # Find c3 that covers all words in to_cover
                            candidate_mask = np.ones(total, dtype=bool)
                            for w in to_cover:
                                candidate_mask &= (ball_membership[:, w] == 1)
                            
                            valid_candidates = np.where(candidate_mask)[0]
                            valid_candidates = valid_candidates[(valid_candidates != c1) & (valid_candidates != c2)]
                            
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
            
            # Update best
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected[:]
                
                if best_size <= 486:
                    break
    
    if best_set is None:
        # Fallback: return a valid (large) code
        # Use pure greedy from scratch
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask) and time.time() < deadline - 5.0:
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        
        best_set = selected
    
    # Convert to digit lists
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
