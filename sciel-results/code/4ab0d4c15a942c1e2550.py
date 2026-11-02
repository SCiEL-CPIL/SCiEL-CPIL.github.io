import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback for other parameters
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
    ball_membership = np.zeros((total, total), dtype=np.int8)
    for w in range(total):
        ball_membership[w, balls[w]] = 1

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

    best_size = total
    best_set = None

    # Strategy: Multi-restart with diverse initializations + vectorized local search + MILP polish
    
    for iteration in range(30):
        if time.time() > deadline - 20.0:
            break

        np.random.seed(iteration * 7919 + 13)

        # Phase 1: Diverse greedy initialization
        # Use a structured approach: start with random seed, then greedily add
        # codewords that cover the most uncovered words
        
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        
        # Initial diverse seed: pick a few random well-spread codewords
        num_seed = 30
        for _ in range(num_seed):
            cand = np.random.randint(total)
            if cand not in selected:
                selected.append(cand)
                cover_count[balls[cand]] += 1
        
        uncovered_mask = (cover_count == 0)
        
        # Greedy completion: add codewords that cover most uncovered words
        while np.any(uncovered_mask):
            if time.time() > deadline - 20.0:
                break
            
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        
        if not np.any(uncovered_mask):
            # Valid solution, run local search
            
            # Phase 2a: Remove redundant codewords (vectorized)
            changed = True
            while changed and time.time() < deadline - 15.0:
                changed = False
                new_selected = []
                for c in selected:
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        changed = True
                    else:
                        new_selected.append(c)
                selected = new_selected
            
            # Phase 2b: Vectorized 1-swap optimization
            improved = True
            while improved and time.time() < deadline - 15.0:
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
            
            # Phase 2c: Vectorized 2-exchange (Remove 2, Add 1)
            if len(selected) > 450 and time.time() < deadline - 10.0:
                improved_2ex = True
                max_2ex_rounds = 20
                for round_num in range(max_2ex_rounds):
                    if time.time() > deadline - 10.0 or not improved_2ex:
                        break
                    improved_2ex = False
                    
                    # Sample random pairs to check
                    num_pairs = min(200, len(selected) * (len(selected) - 1) // 2)
                    for _ in range(num_pairs):
                        if time.time() > deadline - 10.0:
                            break
                        
                        i = np.random.randint(len(selected))
                        j = np.random.randint(len(selected))
                        if i == j:
                            continue
                        
                        c1 = selected[i]
                        c2 = selected[j]
                        
                        # Find words that become uncovered if we remove c1 and c2
                        to_cover_list = []
                        for w in balls[c1]:
                            if cover_count[w] == 1:
                                to_cover_list.append(w)
                        for w in balls[c2]:
                            if cover_count[w] == 1:
                                to_cover_list.append(w)
                        # Intersection words with count 2
                        set_c1 = set(balls[c1].tolist())
                        set_c2 = set(balls[c2].tolist())
                        inter = set_c1 & set_c2
                        for w in inter:
                            if cover_count[w] == 2:
                                to_cover_list.append(w)
                        
                        if not to_cover_list:
                            # Both are redundant
                            cover_count[balls[c1]] -= 1
                            cover_count[balls[c2]] -= 1
                            selected = [c for idx, c in enumerate(selected) if idx != i and idx != j]
                            improved_2ex = True
                            break
                        
                        # Find c3 that covers all words in to_cover_list
                        candidate_mask = np.ones(total, dtype=bool)
                        for w in to_cover_list:
                            candidate_mask &= (ball_membership[:, w] == 1)
                        
                        valid_candidates = np.where(candidate_mask)[0]
                        valid_candidates = valid_candidates[(valid_candidates != c1) & (valid_candidates != c2)]
                        
                        if len(valid_candidates) > 0:
                            c3 = valid_candidates[np.random.randint(len(valid_candidates))]
                            cover_count[balls[c1]] -= 1
                            cover_count[balls[c2]] -= 1
                            cover_count[balls[c3]] += 1
                            selected = [c for idx, c in enumerate(selected) if idx != i and idx != j]
                            selected.append(c3)
                            improved_2ex = True
                            break
            
            # Phase 2d: MILP-based refinement
            # Try to find a smaller subset of current codewords that still covers everything
            if len(selected) > 420 and time.time() < deadline - 5.0:
                try:
                    from scipy.optimize import milp, LinearConstraint, Bounds
                    import scipy.sparse as sp
                    
                    m = len(selected)
                    if m > 600:  # Too large for MILP, skip
                        pass
                    else:
                        # Build constraint matrix: A[w, c] = 1 if codeword c covers word w
                        # We want: for each word w, sum_{c: c covers w} x_c >= 1
                        # Minimize: sum x_c
                        # x_c in {0, 1}
                        
                        # Build sparse matrix A (total x m)
                        rows = []
                        cols = []
                        for ci, c in enumerate(selected):
                            for w in balls[c]:
                                rows.append(w)
                                cols.append(ci)
                        
                        A = sp.csr_matrix(
                            (np.ones(len(rows)), (rows, cols)),
                            shape=(total, m)
                        )
                        
                        # Constraints: A @ x >= 1 for all words
                        # Convert to standard form: -A @ x <= -1
                        A_neg = -A.tocsc()
                        
                        lb = np.full(total, 1.0)  # A @ x >= 1
                        ub = np.full(total, np.inf)
                        
                        constraint = LinearConstraint(A, lb, ub)
                        
                        # Bounds for x: 0 <= x <= 1, integer
                        res_bounds = Bounds(lb=np.zeros(m), ub=np.ones(m))
                        
                        # Objective: minimize sum(x)
                        c_obj = np.ones(m)
                        
                        # Solve MILP
                        result = milp(
                            c=c_obj,
                            constraints=constraint,
                            integrality=np.ones(m),  # All variables integer
                            bounds=res_bounds
                        )
                        
                        if result.status == 0:  # Success
                            x_opt = result.x
                            new_selected = [selected[i] for i in range(m) if x_opt[i] > 0.5]
                            
                            # Verify coverage
                            if len(new_selected) < len(selected):
                                verify_count = np.zeros(total, dtype=np.int32)
                                for c in new_selected:
                                    verify_count[balls[c]] += 1
                                if np.min(verify_count) >= 1:
                                    selected = new_selected
                                    cover_count = verify_count
                                    improved_2ex = True
                except Exception:
                    pass
            
            # Update best
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected[:]
                
                if best_size <= 486:
                    break

    if best_set is None:
        # Fallback: return a valid (large) code
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
