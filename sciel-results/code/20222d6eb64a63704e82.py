# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _precompute_balls(q, n, radius):
    """Precompute ball membership as a boolean matrix (total x total)."""
    total = q ** n
    balls = np.zeros((total, total), dtype=bool)
    # Precompute position indices for efficiency
    if radius >= 1:
        # For each position, precompute the indices of words that differ at that position
        # Actually, let's just iterate
        for idx in range(total):
            word = np.array(np.unravel_index(idx, (q,) * n), dtype=np.int8)
            balls[idx, idx] = True
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    w2 = word.copy()
                    w2[pos] = val
                    pidx = int(np.ravel_multi_index(tuple(w2), (q,) * n))
                    balls[idx, pidx] = True
    else:
        for idx in range(total):
            balls[idx, idx] = True
    return balls


def _greedy_cover_fast(balls, deadline):
    """Greedy set cover using precomputed balls."""
    total = balls.shape[0]
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    
    while uncovered.any() and time.time() < deadline:
        counts = balls[:, uncovered].sum(axis=1)
        best_idx = int(np.argmax(counts))
        if counts[best_idx] == 0:
            break
        codewords.append(best_idx)
        uncovered &= ~balls[best_idx]
    
    return codewords


def _simulated_annealing_swap(codewords, balls, q, n, radius, deadline, seed=42):
    """Simulated annealing with equal-size swaps to escape local minima."""
    random.seed(seed)
    total = balls.shape[0]
    
    # Current solution
    current = list(codewords)
    current_set = set(current)
    
    # Compute coverage
    covered = np.zeros(total, dtype=bool)
    for cw in current:
        covered |= balls[cw]
    
    if not covered.all():
        # Not a valid cover - fix by adding
        while not covered.all() and time.time() < deadline:
            uncovered_mask = ~covered
            counts = balls[:, uncovered_mask].sum(axis=1)
            best_idx = int(np.argmax(counts))
            if counts[best_idx] == 0:
                break
            current.append(best_idx)
            current_set.add(best_idx)
            covered |= balls[best_idx]
    
    best = list(current)
    best_len = len(best)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    n_steps = 5000
    
    step = 0
    T = T_start
    
    while time.time() < deadline and step < n_steps:
        step += 1
        # Cool down
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        # Try a random swap: remove one codeword, add another
        if len(current) <= 1:
            break
        
        i = random.randint(0, len(current) - 1)
        removed = current[i]
        
        # Compute what's covered without 'removed'
        covered_without = np.zeros(total, dtype=bool)
        for j, cw in enumerate(current):
            if j != i:
                covered_without |= balls[cw]
        
        # Points only covered by 'removed'
        must_cover = covered & ~covered_without
        
        # Find candidates that cover must_cover
        # A candidate c is valid if balls[c] covers must_cover
        if must_cover.any():
            must_cover_indices = np.where(must_cover)[0]
            # Candidate must cover all must_cover points
            # Check: for each must_cover point, is it in ball[c]?
            # balls[c, must_cover_indices] should all be True
            valid_mask = balls[:, must_cover_indices].all(axis=1)
            # Also exclude already-in-set candidates
            valid_mask[list(current_set)] = False
            candidates = np.where(valid_mask)[0]
            
            if len(candidates) == 0:
                continue
            
            new_cw = int(random.choice(candidates))
        else:
            # Removing 'removed' doesn't lose coverage, so any candidate works
            # Pick a random one not in the set
            all_candidates = np.array([x for x in range(total) if x not in current_set])
            if len(all_candidates) == 0:
                continue
            new_cw = int(random.choice(all_candidates))
        
        # Perform swap
        current[i] = new_cw
        current_set.discard(removed)
        current_set.add(new_cw)
        
        # Check validity
        new_covered = np.zeros(total, dtype=bool)
        for cw in current:
            new_covered |= balls[cw]
        
        if new_covered.all():
            # Valid swap - accept
            covered = new_covered
            if len(current) < best_len:
                best = list(current)
                best_len = len(best)
        else:
            # Invalid - revert with probability based on "damage"
            damage = (~new_covered).sum()
            # Accept with probability exp(-damage / T)
            if random.random() < np.exp(-damage / max(T, 1e-10)):
                covered = new_covered
            else:
                # Revert
                current[i] = removed
                current_set.discard(new_cw)
                current_set.add(removed)
    
    return best


def _local_search_remove(codewords, balls, deadline):
    """Try to remove redundant codewords."""
    total = balls.shape[0]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            # Check if removing codewords[i] still covers everything
            covered = np.zeros(total, dtype=bool)
            for j in range(len(codewords)):
                if j != i:
                    covered |= balls[codewords[j]]
            if covered.all():
                codewords.pop(i)
                improved = True
                break
    
    return codewords


def _swap_local_search(codewords, balls, q, n, radius, deadline):
    """Try swapping one codeword for another to reduce overlap."""
    total = balls.shape[0]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            current_cw = codewords[i]
            
            # Compute coverage without current_cw
            others_covered = np.zeros(total, dtype=bool)
            for j in range(len(codewords)):
                if j != i:
                    others_covered |= balls[codewords[j]]
            
            # Points that must be covered by the replacement
            must_cover = np.ones(total, dtype=bool) & ~others_covered
            
            if must_cover.any():
                must_cover_indices = np.where(must_cover)[0]
                valid_mask = balls[:, must_cover_indices].all(axis=1)
            else:
                valid_mask = np.ones(total, dtype=bool)
            
            # Exclude current set members
            valid_mask[codewords] = False
            
            candidates = np.where(valid_mask)[0]
            if len(candidates) == 0:
                continue
            
            # Among valid candidates, pick one that minimizes overlap with others
            best_overlap = len(balls[current_cw] & others_covered)
            best_cand = -1
            for cand in candidates:
                overlap = len(balls[cand] & others_covered)
                if overlap < best_overlap:
                    best_overlap = overlap
                    best_cand = cand
            
            if best_cand >= 0:
                codewords[i] = best_cand
                improved = True
                break
    
    return codewords


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115  # 115s safety margin out of 150s
    
    total = q ** n
    
    # Precompute balls
    balls = _precompute_balls(q, n, radius)
    
    # Greedy cover
    if time.time() < deadline - 30:
        greedy = _greedy_cover_fast(balls, deadline - 30)
    else:
        # Fallback: tetracode product
        import itertools
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            initial_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            initial_words = [list(w) for w in itertools.product(range(q), repeat=n)]
        greedy = []
        for w in initial_words:
            idx = int(np.ravel_multi_index(tuple(w), (q,) * n))
            greedy.append(idx)
    
    best = list(greedy)
    
    # Remove redundant
    if time.time() < deadline - 20:
        best = _local_search_remove(best, balls, deadline - 20)
    
    # Simulated annealing with multiple seeds
    remaining = deadline - start
    if remaining > 15:
        n_seeds = max(1, int(remaining / 15))
        for seed in range(n_seeds):
            if time.time() >= deadline - 5:
                break
            sa_result = _simulated_annealing_swap(best, balls, q, n, radius, deadline - 5, seed=seed + 42)
            if len(sa_result) < len(best):
                best = sa_result
    
    # Final swap local search
    if time.time() < deadline - 3:
        best = _swap_local_search(best, balls, q, n, radius, deadline - 3)
    
    # Final removal pass
    if time.time() < deadline - 1:
        best = _local_search_remove(best, balls, deadline - 1)
    
    # Convert indices back to words
    result = []
    for idx in best:
        word = list(np.unravel_index(idx, (q,) * n))
        result.append(word)
    
    return result
# EVOLVE-BLOCK-END
