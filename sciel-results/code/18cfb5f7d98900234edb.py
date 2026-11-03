import time
from math import log, floor, sqrt, exp, log1p
import random
import numpy as np

def score_set(U):
    """Compute the exact score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def compute_sum_diff_counts(U):
    """Return (sum_count, diff_count, max_u) for a sorted set U containing 0."""
    n = len(U)
    sums = set()
    diffs = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    return len(sums), len(diffs), max(U)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
    return list(candidates)

def compute_marginal_gain(U, candidate):
    """
    Compute the marginal effect of adding 'candidate' to U on the score components.
    Returns (delta_sum_count, delta_diff_count, new_max_u).
    This is an approximation for guided selection.
    """
    U_set = set(U)
    U_sorted = sorted(U + [candidate])
    n = len(U_sorted)
    
    # Count new sums created
    new_sums = 0
    existing_sums = set()
    for i in range(n):
        for j in range(i, n):
            s = U_sorted[i] + U_sorted[j]
            existing_sums.add(s)
    
    # Count new diffs created
    new_diffs = 0
    existing_diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = U_sorted[j] - U_sorted[i]
            existing_diffs.add(d)
            existing_diffs.add(-d)
    existing_diffs.add(0)
    
    # Original counts
    orig_sums, orig_diffs, orig_max = compute_sum_diff_counts(U)
    
    delta_sums = len(existing_sums) - orig_sums
    delta_diffs = len(existing_diffs) - orig_diffs
    new_max = max(orig_max, candidate)
    
    return delta_sums, delta_diffs, new_max

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation and weighted greedy selection.
    Uses a smooth surrogate to guide candidate selection.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.15:
            current_max = max(U)
            new_limit = max(max_val_limit, int(current_max * 1.5) + 10)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.45:
            # Try to ADD an element - use weighted selection
            if len(U) < max_n_limit and candidates:
                # Score candidates by estimated marginal benefit
                scored_candidates = []
                for c in candidates[:20]:
                    if c in U:
                        continue
                    try:
                        ds, dd, nm = compute_marginal_gain(U, c)
                        # Smooth surrogate: prefer candidates that increase diff more than sum
                        # Use log1p for numerical stability
                        benefit = log1p(max(0, dd)) - 0.5 * log1p(max(0, ds))
                        # Penalize large max values
                        if nm > max(U) * 2:
                            benefit *= 0.7
                        scored_candidates.append((benefit, c))
                    except:
                        pass
                
                scored_candidates.sort(key=lambda x: -x[0])
                for _, c in scored_candidates[:5]:
                    new_U = sorted(U + [c])
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.65:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    # Try removing the element with least contribution
                    best_remove = None
                    best_remove_score = current_score
                    for c in random.sample(removable, min(5, len(removable))):
                        new_U = [x for x in U if x != c]
                        new_score = score_set(new_U)
                        if new_score > best_remove_score:
                            best_remove_score = new_score
                            best_remove = c
                    if best_remove is not None and best_remove_score > current_score:
                        U = [x for x in U if x != best_remove]
                        current_score = best_remove_score
                        improved = True
                        candidates = generate_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                for new_val in candidates[:10]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U = sorted(set(new_U))
                    if len(new_U) != len(U):
                        continue
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
        [0, 1, 5, 14, 30, 55, 91], # Pentatomial-ish
        [0, 1, 4, 10, 20, 35, 56, 84, 120], # Larger triangular
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Geometric Search
    b_candidates = [b_int / 10.0 for b_int in range(12, 80)]
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 50:
            break
        for m in range(2, 18):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                if s > best_score:
                    best_score = s
                    best_set = U

    # 3. Dynamic Local Search with multiple restarts
    seeds = [best_set]
    for cand in structured_candidates[:4]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 400
    max_n_limit = 28
    
    restarts = 0
    while time.time() < deadline - 20:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            if len(seeds) < 6:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 4. Final High-Precision Polish
    if time.time() < deadline - 3:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        polish_deadline = deadline - 1
        
        while improved and time.time() < polish_deadline and iteration < 800:
            improved = False
            iteration += 1
            n = len(current)
            
            # Phase 1: Fine-grained replacement (delta -15 to 15)
            for i in range(1, n):
                if time.time() > polish_deadline:
                    break
                old_val = current[i]
                for delta in range(-15, 16):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Phase 2: Add element near max or in gaps
            if not improved and n < 24 and time.time() < polish_deadline:
                max_val = max(current)
                # Try values near max and in gaps
                try_vals = set()
                for v in range(max(1, max_val - 20), max_val + 21):
                    try_vals.add(v)
                # Also try mid-range values
                for v in range(1, max_val, max(1, max_val // 20)):
                    try_vals.add(v)
                
                for new_val in sorted(try_vals):
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Phase 3: Remove element with least impact
            if not improved and n > 3 and time.time() < polish_deadline:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Phase 4: Pairwise swap (replace two elements)
            if not improved and n >= 4 and time.time() < polish_deadline:
                for i in range(1, n):
                    if time.time() > polish_deadline:
                        break
                    for j in range(i + 1, n):
                        if time.time() > polish_deadline:
                            break
                        old_i = current[i]
                        old_j = current[j]
                        # Try small perturbations to both
                        for di in range(-5, 6):
                            for dj in range(-5, 6):
                                if di == 0 and dj == 0:
                                    continue
                                new_i = old_i + di
                                new_j = old_j + dj
                                if new_i <= 0 or new_j <= 0:
                                    continue
                                if new_i == new_j:
                                    continue
                                new_set = list(current)
                                new_set[i] = new_i
                                new_set[j] = new_j
                                new_set = sorted(set(new_set))
                                if len(new_set) != n:
                                    continue
                                new_score = score_set(new_set)
                                if new_score > current_score:
                                    current = new_set
                                    current_score = new_score
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

# Ensure 0 is in the set and it's a list of nonnegative integers
# The function construct returns a list.
# The evaluator calls construct().

if __name__ == "__main__":
    # Quick test
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
