"""Greedy set cover + local search for K_q(n, radius)."""
import time
import itertools
import numpy as np
import random

_DEADLINE = 150.0


def _word_to_idx(word, q, n):
    """Convert a word (list of ints) to a linear index in [0, q**n)."""
    idx = 0
    for x in word:
        idx = idx * q + x
    return idx


def _idx_to_word(idx, q, n):
    """Convert a linear index to a word (list of ints)."""
    word = [0] * n
    for c in range(n - 1, -1, -1):
        word[c] = idx % q
        idx //= q
    return word


def _get_neighbors(cw_idx, q, n, radius):
    """Return list of indices of all words within Hamming distance <= radius of cw_idx."""
    if radius == 1:
        # Fast path for radius 1
        neighbors = [cw_idx]
        word = _idx_to_word(cw_idx, q, n)
        for pos in range(n):
            offset = q ** (n - 1 - pos)
            cur = word[pos]
            for v in range(q):
                if v != cur:
                    neighbors.append(cw_idx + (v - cur) * offset)
        return neighbors
    else:
        # Generic: enumerate all words and check distance
        total = q ** n
        word = _idx_to_word(cw_idx, q, n)
        neighbors = []
        for i in range(total):
            w = _idx_to_word(i, q, n)
            d = sum(1 for a, b in zip(word, w) if a != b)
            if d <= radius:
                neighbors.append(i)
        return neighbors


def _greedy_cover(q, n, radius, deadline, seed=42):
    """
    Greedy set cover: repeatedly pick the word that covers the most uncovered words.
    Returns a list of codeword indices.
    """
    rng = random.Random(seed)
    total = q ** n
    
    # Precompute all neighbor sets
    all_neighbors = {}
    for i in range(total):
        all_neighbors[i] = _get_neighbors(i, q, n, radius)
    
    # uncovered: set of uncovered word indices
    uncovered = set(range(total))
    # coverage_count[i] = number of codewords that cover word i
    coverage_count = np.zeros(total, dtype=np.int32)
    
    codewords = []
    
    while uncovered and time.time() < deadline:
        # Find the word that covers the most uncovered words
        best_cw = -1
        best_count = -1
        # To speed up, we can sample or use a heuristic
        # For now, scan all candidates (could be optimized with a priority queue)
        # But total is 6561, and we do this ~500 times, so 6561*500 = 3.3M operations, fine.
        
        # Optimization: only consider words that are themselves uncovered or have high coverage
        # Actually, we should consider all words as potential codewords
        # But we can skip words that are already covered and have low marginal gain
        
        # Sample-based approach for speed: consider a random subset + the best so far
        # Actually, let's do a full scan but with early termination if we find a perfect cover
        
        candidates = list(uncovered)  # Start with uncovered words as candidates
        if not candidates:
            break
        # Also add some random covered words that might have high marginal coverage
        if len(candidates) < total // 2:
            extra = rng.sample(range(total), min(total // 4, 1000))
            candidates = list(set(candidates + extra))
        
        for cw in candidates:
            if time.time() > deadline:
                break
            # Count how many uncovered words this codeword covers
            cnt = 0
            for nb in all_neighbors[cw]:
                if nb in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_cw = cw
                if best_count == len(uncovered):
                    break
        
        if best_cw == -1 or best_count <= 0:
            # No improvement possible, but we still need to cover all
            # Pick any uncovered word
            best_cw = next(iter(uncovered))
        
        # Add best_cw as a codeword
        codewords.append(best_cw)
        for nb in all_neighbors[best_cw]:
            if nb in uncovered:
                uncovered.discard(nb)
            coverage_count[nb] += 1
    
    return codewords


def _local_search(codewords, q, n, radius, deadline, seed=42):
    """
    Local search with vectorized redundancy checks, swap scoring with redundancy
    penalty, and periodic full sweeps to escape local minima.
    Returns improved list of codeword indices.
    """
    rng = random.Random(seed)
    total = q ** n
    
    # Precompute neighbor sets as numpy arrays for vectorized ops
    all_neighbors = {}
    for i in range(total):
        all_neighbors[i] = np.array(_get_neighbors(i, q, n, radius), dtype=np.int32)
    
    # coverage_count[i] = number of codewords covering word i
    coverage_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        coverage_count[all_neighbors[cw]] += 1
    
    cw_set = set(codewords)
    improved = True
    sweep_count = 0
    while improved and time.time() < deadline:
        improved = False
        sweep_count += 1
        
        # Periodic full redundancy sweep every 5 iterations to catch missed removals
        if sweep_count % 5 == 1:
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                cw = codewords[i]
                # Vectorized check: all neighbors have coverage >= 2?
                if np.min(coverage_count[all_neighbors[cw]]) >= 2:
                    coverage_count[all_neighbors[cw]] -= 1
                    codewords.pop(i)
                    cw_set.discard(cw)
                    improved = True
                    # Restart scan since indices shifted
                    break
            if improved:
                continue
        
        # Try removal: vectorized check
        for i, cw in enumerate(list(codewords)):
            if time.time() > deadline:
                break
            if np.min(coverage_count[all_neighbors[cw]]) >= 2:
                coverage_count[all_neighbors[cw]] -= 1
                codewords.pop(i)
                cw_set.discard(cw)
                improved = True
                break
        
        # Try swap with redundancy penalty scoring
        if not improved and len(codewords) > 1:
            remove_candidates = rng.sample(codewords, min(len(codewords), 25))
            for rem_cw in remove_candidates:
                if time.time() > deadline:
                    break
                # Vectorized: find words that would become uncovered
                rem_neighbors = all_neighbors[rem_cw]
                would_be_uncovered = rem_neighbors[coverage_count[rem_neighbors] == 1]
                if len(would_be_uncovered) == 0:
                    # Can just remove
                    coverage_count[rem_neighbors] -= 1
                    codewords.remove(rem_cw)
                    cw_set.discard(rem_cw)
                    improved = True
                    break
                
                # Build candidate replacements: words within distance 1 of any would-be-uncovered
                add_candidates = set()
                for w in would_be_uncovered:
                    word = _idx_to_word(w, q, n)
                    for pos in range(n):
                        offset = q ** (n - 1 - pos)
                        cur = word[pos]
                        for v in range(q):
                            if v != cur:
                                add_candidates.add(w + (v - cur) * offset)
                    add_candidates.add(w)
                
                add_candidates -= cw_set
                if not add_candidates:
                    continue
                
                # Score candidates: gain = covers would_be_uncovered - redundancy penalty
                best_add = -1
                best_score = -1
                for add_cw in list(add_candidates)[:60]:
                    add_neighbors = all_neighbors[add_cw]
                    # Gain: how many would_be_uncovered this covers
                    gain = np.sum(np.isin(would_be_uncovered, add_neighbors))
                    # Redundancy penalty: average coverage of add_neighbors (lower is better)
                    redundancy = np.mean(coverage_count[add_neighbors])
                    score = gain * 10.0 - redundancy
                    if score > best_score:
                        best_score = score
                        best_add = add_cw
                    if gain == len(would_be_uncovered):
                        break
                
                if best_add != -1 and np.sum(np.isin(would_be_uncovered, all_neighbors[best_add])) == len(would_be_uncovered):
                    # Perform swap
                    coverage_count[all_neighbors[rem_cw]] -= 1
                    coverage_count[all_neighbors[best_add]] += 1
                    codewords.remove(rem_cw)
                    cw_set.discard(rem_cw)
                    codewords.append(best_add)
                    cw_set.add(best_add)
                    improved = True
                    break
    
    return codewords


def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + _DEADLINE - 10.0  # safety margin
    
    total = q ** n
    
    # Phase 1: Greedy set cover
    greedy_result = _greedy_cover(q, n, radius, deadline, seed=42)
    
    # Phase 2: Local search to improve (use most of remaining time)
    ls_deadline = deadline - 5.0  # leave 5s for verification
    result = _local_search(greedy_result, q, n, radius, ls_deadline, seed=42)
    
    # Verify coverage
    all_neighbors = {}
    for cw in result:
        all_neighbors[cw] = _get_neighbors(cw, q, n, radius)
    
    covered = np.zeros(total, dtype=bool)
    for cw in result:
        for nb in all_neighbors[cw]:
            covered[nb] = True
    
    if not covered.all():
        # Fallback: direct sum construction
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            out = []
            for h in tetracode:
                for tail in itertools.product(range(q), repeat=n - 4):
                    out.append(h + list(tail))
            result = [_word_to_idx(w, q, n) for w in out]
        else:
            # Full space
            result = list(range(total))
    
    # Convert back to list of words
    return [_idx_to_word(idx, q, n) for idx in result]
