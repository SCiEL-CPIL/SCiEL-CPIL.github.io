"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _all_words(q, n):
    """Generate all q^n words as a numpy array of shape (q^n, n)."""
    total = q ** n
    words = np.zeros((total, n), dtype=np.int8)
    for i in range(total):
        v = i
        for j in range(n - 1, -1, -1):
            words[i, j] = v % q
            v //= q
    return words


def _neighbors_of_word(word, q, n):
    """Yield all words within Hamming distance 1 of the given word."""
    yield tuple(word)
    for pos in range(n):
        orig = word[pos]
        for val in range(q):
            if val == orig:
                continue
            new_word = list(word)
            new_word[pos] = val
            yield tuple(new_word)


def _tetracode_direct_sum(q, n):
    """Generate the tetracode direct sum construction."""
    if n < 4:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    result = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            result.append(h + list(tail))
    return result


def _is_valid_cover(codewords, q, n, radius=1):
    """Check if codewords form a valid cover using vectorized operations."""
    total = q ** n
    if not codewords:
        return False
    
    all_words = _all_words(q, n)
    cw_array = np.array(codewords, dtype=np.int8)
    
    # For each codeword, compute which words it covers (Hamming distance <= radius)
    # Use broadcasting: diff[i, j] = |cw[i] - all_words[j]|
    # This is memory intensive for large spaces, but 729*73*6 is manageable
    covered = np.zeros(total, dtype=bool)
    
    for cw in codewords:
        cw_arr = np.array(cw, dtype=np.int8)
        # Compute Hamming distance from this codeword to all words
        diff = np.abs(all_words - cw_arr)  # shape (total, n)
        dist = np.sum(diff > 0, axis=1)  # shape (total,)
        covered |= (dist <= radius)
    
    return np.all(covered)


def _is_valid_cover_fast(codewords, q, n, radius=1):
    """Faster validity check using precomputed coverage."""
    total = q ** n
    if not codewords:
        return False
    
    all_words = _all_words(q, n)
    covered = np.zeros(total, dtype=bool)
    
    for cw in codewords:
        cw_arr = np.array(cw, dtype=np.int8)
        diff = np.abs(all_words - cw_arr)
        dist = np.sum(diff > 0, axis=1)
        covered |= (dist <= radius)
        if np.all(covered):
            return True
    
    return np.all(covered)


def _greedy_cover(q, n, radius=1, time_limit=30, seed=None):
    """Greedy set cover with symmetry-aware tie-breaking."""
    if seed is not None:
        random.seed(seed)
    
    start_time = time.time()
    total = q ** n
    all_words = _all_words(q, n)
    
    # Precompute coverage for all possible codewords
    # coverage[cand_idx] = set of word indices covered
    all_candidates = list(itertools.product(range(q), repeat=n))
    n_candidates = len(all_candidates)
    
    # Map word to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute which words each candidate covers
    coverage_sets = []
    for cand in all_candidates:
        cand_arr = np.array(cand, dtype=np.int8)
        diff = np.abs(all_words - cand_arr)
        dist = np.sum(diff > 0, axis=1)
        covered_idx = np.where(dist <= radius)[0]
        coverage_sets.append(covered_idx)
    
    covered = np.zeros(total, dtype=bool)
    selected = []
    selected_set = set()
    
    while not np.all(covered):
        if time.time() - start_time > time_limit:
            break
        
        best_gain = 0
        best_candidates = []
        
        for i, cand in enumerate(all_candidates):
            if i in selected_set:
                continue
            gain = np.sum(~covered[coverage_sets[i]])
            if gain > best_gain:
                best_gain = gain
                best_candidates = [i]
            elif gain == best_gain:
                best_candidates.append(i)
        
        if best_gain == 0:
            break
        
        # Symmetry-aware tie-breaking: prefer candidates that are "central" 
        # (closer to the all-zeros word) to promote regularity
        if len(best_candidates) > 1:
            def _centrality(idx):
                word = all_candidates[idx]
                return sum(word)  # L1 norm as centrality measure
            best_candidates.sort(key=_centrality)
            chosen = best_candidates[0]
        else:
            chosen = best_candidates[0]
        
        selected.append(all_candidates[chosen])
        selected_set.add(chosen)
        covered[coverage_sets[chosen]] = True
    
    return selected


def _local_search_remove(codewords, q, n, radius=1, time_limit=20):
    """Local search: iteratively remove redundant codewords."""
    start_time = time.time()
    current = list(codewords)
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(len(current)):
            if time.time() - start_time > time_limit:
                break
            trial = current[:i] + current[i+1:]
            if _is_valid_cover_fast(trial, q, n, radius):
                current = trial
                improved = True
                break
        if not improved:
            break
    
    return current


def _local_search_swap(codewords, q, n, radius=1, time_limit=20):
    """Local search: try swapping codewords to improve coverage efficiency."""
    start_time = time.time()
    current = list(codewords)
    total = q ** n
    all_candidates = list(itertools.product(range(q), repeat=n))
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(len(current)):
            if time.time() - start_time > time_limit:
                break
            
            # Try removing codeword i and see if we can replace it
            without_i = current[:i] + current[i+1:]
            
            # Find words not covered by without_i
            if not _is_valid_cover_fast(without_i, q, n, radius):
                # Need to add a new codeword that covers the missing words
                all_words = _all_words(q, n)
                covered_by_without = np.zeros(total, dtype=bool)
                for cw in without_i:
                    cw_arr = np.array(cw, dtype=np.int8)
                    diff = np.abs(all_words - cw_arr)
                    dist = np.sum(diff > 0, axis=1)
                    covered_by_without |= (dist <= radius)
                
                uncovered_idx = np.where(~covered_by_without)[0]
                
                # Find candidate that covers most uncovered words
                best_gain = 0
                best_cand = None
                for cand in all_candidates:
                    cand_arr = np.array(cand, dtype=np.int8)
                    diff = np.abs(all_words - cand_arr)
                    dist = np.sum(diff > 0, axis=1)
                    gain = np.sum((dist <= radius) & (~covered_by_without))
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = cand
                
                if best_cand is not None and best_gain > 0:
                    trial = without_i + [list(best_cand)]
                    if _is_valid_cover_fast(trial, q, n, radius):
                        current = trial
                        improved = True
                        break
    
    return current


def _simulated_annealing(codewords, q, n, radius=1, time_limit=40):
    """Simulated annealing with smooth objective."""
    start_time = time.time()
    total = q ** n
    all_words = _all_words(q, n)
    all_candidates = list(itertools.product(range(q), repeat=n))
    n_candidates = len(all_candidates)
    
    # Precompute coverage
    coverage_sets = []
    for cand in all_candidates:
        cand_arr = np.array(cand, dtype=np.int8)
        diff = np.abs(all_words - cand_arr)
        dist = np.sum(diff > 0, axis=1)
        coverage_sets.append(np.where(dist <= radius)[0])
    
    # Current state
    current_included = np.zeros(n_candidates, dtype=bool)
    for cw in codewords:
        idx = all_candidates.index(tuple(cw))
        current_included[idx] = True
    
    cov = np.zeros(total, dtype=np.int32)
    for i in range(n_candidates):
        if current_included[i]:
            cov[coverage_sets[i]] += 1
    
    current_uncovered = np.sum(cov == 0)
    current_weight = np.sum(current_included)
    
    best_cw = list(codewords)
    best_score = current_weight
    
    T = 10.0
    cooling = 0.998
    
    while time.time() - start_time < time_limit:
        T *= cooling
        if T < 0.05:
            break
        
        # Random move: toggle a candidate
        i = random.randint(0, n_candidates - 1)
        old_included = current_included[i]
        
        if old_included:
            # Removing
            delta_weight = -1
            new_uncovered = np.sum((cov[coverage_sets[i]] == 1))
            delta_uncovered = new_uncovered
        else:
            # Adding
            delta_weight = 1
            new_uncovered = np.sum((cov[coverage_sets[i]] == 0))
            delta_uncovered = -new_uncovered
        
        # Smooth objective: weight + lambda * uncovered (with soft penalty)
        lambda_pen = 50.0
        delta_score = delta_weight + lambda_pen * delta_uncovered
        
        if delta_score <= 0 or random.random() < np.exp(-delta_score / T):
            current_included[i] = not old_included
            delta = 1 if not old_included else -1
            cov[coverage_sets[i]] += delta
            current_uncovered += delta_uncovered
            current_weight += delta_weight
            
            if current_uncovered == 0 and current_weight < best_score:
                best_score = current_weight
                best_cw = [list(all_candidates[i]) for i in range(n_candidates) if current_included[i]]
        
        # Occasionally try to remove redundant codewords
        if random.random() < 0.3:
            redundant = []
            for i in range(n_candidates):
                if current_included[i]:
                    if np.all(cov[coverage_sets[i]] > 1):
                        redundant.append(i)
            
            if redundant:
                i = random.choice(redundant)
                current_included[i] = False
                cov[coverage_sets[i]] -= 1
                current_weight -= 1
                
                if current_uncovered == 0 and current_weight < best_score:
                    best_score = current_weight
                    best_cw = [list(all_candidates[i]) for i in range(n_candidates) if current_included[i]]
    
    return best_cw


def construct(q, n, radius):
    """Construct a covering code using multiple strategies."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with tetracode direct sum
    codewords = _tetracode_direct_sum(q, n)
    best = list(codewords)
    best_len = len(best)
    
    # Phase 1: Greedy deletion with fast validity check
    for _ in range(5):
        if time.time() > deadline - 80:
            break
        changed = True
        while changed:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline - 80:
                    break
                trial = codewords[:i] + codewords[i+1:]
                if _is_valid_cover_fast(trial, q, n, radius):
                    codewords = trial
                    changed = True
                    break
        if len(codewords) < best_len:
            best = list(codewords)
            best_len = len(best)
    
    # Phase 2: Local search with swaps
    if time.time() < deadline - 40:
        result = _local_search_swap(codewords, q, n, radius, time_limit=25)
        if len(result) < best_len:
            best = result
            best_len = len(best)
            codewords = result
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        result = _simulated_annealing(codewords, q, n, radius, time_limit=min(remaining, 50))
        if len(result) < best_len:
            best = result
            best_len = len(best)
    
    # Final validation
    if not _is_valid_cover(best, q, n, radius):
        best = _tetracode_direct_sum(q, n)
    
    return best
# EVOLVE-BLOCK-END
