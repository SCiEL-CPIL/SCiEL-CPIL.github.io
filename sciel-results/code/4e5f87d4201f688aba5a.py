# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _ball_indices(word, q, n, radius):
    """Return a frozenset of indices (integers 0..q^n-1) within Hamming distance radius of word."""
    word = np.array(word, dtype=np.int8)
    indices = [int(np.ravel_multi_index(tuple(word), (q,) * n))]
    if radius >= 1:
        for pos in range(n):
            for val in range(q):
                if val == word[pos]:
                    continue
                w2 = word.copy()
                w2[pos] = val
                indices.append(int(np.ravel_multi_index(tuple(w2), (q,) * n)))
    return frozenset(indices)


def _tetracode(q, n):
    """Product construction: ternary tetracode (9 words of length 4) x all words on remaining coords."""
    if n < 4:
        import itertools
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _greedy_cover_fast(q, n, radius, deadline):
    """Faster greedy using numpy for coverage counting."""
    total = q ** n
    # Build a matrix: rows = words, cols = points, 1 if point is in ball of word
    balls = np.zeros((total, total), dtype=bool)
    for idx in range(total):
        word = list(np.unravel_index(idx, (q,) * n))
        for p in _ball_indices(word, q, n, radius):
            balls[idx, p] = True
    
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    
    while uncovered.any() and time.time() < deadline:
        counts = balls[:, uncovered].sum(axis=1)
        best_idx = int(np.argmax(counts))
        if counts[best_idx] == 0:
            break
        word = list(np.unravel_index(best_idx, (q,) * n))
        codewords.append(word)
        uncovered &= ~balls[best_idx]
    
    return codewords


def _local_search_remove(codewords, q, n, radius, deadline):
    """Try to remove redundant codewords."""
    total = q ** n
    ball_sets = [_ball_indices(cw, q, n, radius) for cw in codewords]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            remaining = [j for j in range(len(codewords)) if j != i]
            covered = set()
            for j in remaining:
                covered |= ball_sets[j]
            if len(covered) == total:
                codewords.pop(i)
                ball_sets.pop(i)
                improved = True
                break
    
    return codewords


def _sa_cover(q, n, radius, deadline):
    """Simulated annealing based covering code optimization using numpy neighbors."""
    total = q ** n
    powers = [q ** p for p in range(n)]
    
    # Precompute neighbors as numpy array
    # Each word has 1 + n*(q-1) neighbors (itself + one-position changes)
    num_neighbors = 1 + n * (q - 1)
    neighbors = np.empty((total, num_neighbors), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % q for p in range(n)]
        neighbors[i, 0] = i
        k = 1
        for p in range(n):
            for alt in range(q):
                if alt != d[p]:
                    new_i = i + (alt - d[p]) * powers[p]
                    neighbors[i, k] = new_i
                    k += 1
    
    # Precompute neighbor sets for fast subset checks
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(total)]
    
    # Helper to convert index to list
    def idx_to_list(idx):
        return [(idx // powers[p]) % q for p in range(n)]
    
    # Initial Solution: Greedy Set Cover
    uncovered = np.ones(total, dtype=np.int8)
    codewords = []
    
    while uncovered.sum() > 0:
        if time.time() > deadline - 60:
            break
        counts = np.sum(uncovered[neighbors], axis=1)
        best_c = int(np.argmax(counts))
        if counts[best_c] == 0:
            break
        codewords.append(best_c)
        uncovered[neighbors[best_c]] = 0
        
    if uncovered.sum() != 0:
        while uncovered.sum() > 0:
            c = int(np.random.randint(total))
            codewords.append(c)
            uncovered[neighbors[c]] = 0
            
    # Initialize coverage counts
    coverage_counts = np.zeros(total, dtype=np.int32)
    for c in codewords:
        coverage_counts[neighbors[c]] += 1
        
    in_set = set(codewords)
    
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    random.seed(42)
    np.random.seed(42)
    
    T = 1.0
    T_min = 0.005
    cooling = 0.9995
    
    steps = 0
    while time.time() < deadline - 5:
        steps += 1
        
        # Find redundant words efficiently
        # A codeword c is redundant if for all w in neighbors[c], coverage_counts[w] >= 2
        redundant_list = []
        for idx, c in enumerate(codewords):
            if coverage_counts[neighbors[c]].min() >= 2:
                redundant_list.append(idx)
                
        if redundant_list:
            if len(redundant_list) > 1:
                slacks = [coverage_counts[neighbors[codewords[i]]].min() for i in redundant_list]
                max_slack = max(slacks)
                best_candidates = [i for i, s in zip(redundant_list, slacks) if s == max_slack]
                idx_to_remove = random.choice(best_candidates)
            else:
                idx_to_remove = redundant_list[0]
            
            c_to_remove = codewords[idx_to_remove]
            codewords.pop(idx_to_remove)
            coverage_counts[neighbors[c_to_remove]] -= 1
            in_set.discard(c_to_remove)
            
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            
            T = max(T * cooling, T_min)
            continue
            
        else:
            # No redundant words. Try swaps.
            found_swap = False
            num_tries = 3
            for _ in range(num_tries):
                idx_to_remove = random.randint(0, len(codewords) - 1)
                c_to_remove = codewords[idx_to_remove]
                
                crit_mask = (coverage_counts[neighbors[c_to_remove]] == 1)
                critical_words = neighbors[c_to_remove][crit_mask]
                
                if len(critical_words) == 0:
                    continue
                
                w0 = int(critical_words[0])
                candidates = neighbors[w0]
                
                valid_candidates = []
                for cand in candidates:
                    if cand == c_to_remove:
                        continue
                    if all(w in neighbor_sets[int(cand)] for w in critical_words):
                        valid_candidates.append(int(cand))
                
                if not valid_candidates:
                    continue
                
                c_to_add = random.choice(valid_candidates)
                
                codewords[idx_to_remove] = c_to_add
                coverage_counts[neighbors[c_to_remove]] -= 1
                coverage_counts[neighbors[c_to_add]] += 1
                in_set.discard(c_to_remove)
                in_set.add(c_to_add)
                
                found_swap = True
                break
            
            T = max(T * cooling, T_min)

    # Final cleanup
    improved = True
    while improved:
        improved = False
        for idx in range(len(codewords)):
            c = codewords[idx]
            if coverage_counts[neighbors[c]].min() >= 2:
                codewords.pop(idx)
                coverage_counts[neighbors[c]] -= 1
                improved = True
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                break

    if len(best_codewords) < len(codewords):
        codewords = best_codewords

    return [idx_to_list(c) for c in codewords]


def _swap_local_search(codewords, q, n, radius, deadline):
    """Try swapping one codeword for another to improve coverage."""
    total = q ** n
    ball_sets = [_ball_indices(cw, q, n, radius) for cw in codewords]
    
    all_covered = set()
    for bs in ball_sets:
        all_covered |= bs
    
    for i in range(len(codewords)):
        if time.time() > deadline:
            break
        current_ball = ball_sets[i]
        others_covered = set()
        for j in range(len(codewords)):
            if j != i:
                others_covered |= ball_sets[j]
        
        must_cover = all_covered - others_covered
        
        for idx in range(total):
            candidate_ball = _ball_indices(list(np.unravel_index(idx, (q,) * n)), q, n, radius)
            if must_cover <= candidate_ball:
                new_covered = others_covered | candidate_ball
                if len(new_covered) == total:
                    overlap = len(candidate_ball & others_covered)
                    if overlap < len(current_ball & others_covered):
                        codewords[i] = list(np.unravel_index(idx, (q,) * n))
                        ball_sets[i] = candidate_ball
                        all_covered = new_covered
                        break
    
    return codewords


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115  # 115s safety margin out of 150s
    
    # Start with tetracode product construction
    initial = _tetracode(q, n)
    best = [tuple(cw) for cw in initial]
    
    # Try SA-based cover (fast, uses numpy neighbors)
    if time.time() < deadline - 30:
        try:
            sa_result = _sa_cover(q, n, radius, deadline - 30)
            if len(sa_result) < len(best):
                best = [tuple(cw) for cw in sa_result]
        except Exception:
            pass
    
    # Local search to remove redundant codewords
    if time.time() < deadline - 10:
        best = _local_search_remove([list(cw) for cw in best], q, n, radius, deadline - 10)
        best = [tuple(cw) for cw in best]
    
    # Try swap local search for further improvement
    if time.time() < deadline - 3:
        best = _swap_local_search([list(cw) for cw in best], q, n, radius, deadline - 3)
        best = [tuple(cw) for cw in best]
    
    # Final removal pass
    if time.time() < deadline - 1:
        best = _local_search_remove([list(cw) for cw in best], q, n, radius, deadline - 1)
        best = [tuple(cw) for cw in best]
    
    return [list(cw) for cw in best]
# EVOLVE-BLOCK-END
