"""Structural rewrite: vectorized greedy set-cover with swap-based local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _word_to_idx(word, q):
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _idx_to_word(idx, q, n):
    word = []
    for _ in range(n):
        word.append(idx % q)
        idx //= q
    return word[::-1]


def _all_neighbors_indices(cw_idx, q, n, radius=1):
    """Return the set of indices within Hamming distance <= radius from cw_idx."""
    cw = _idx_to_word(cw_idx, q, n)
    covered = {cw_idx}
    if radius >= 1:
        for pos in range(n):
            for new_val in range(q):
                if new_val == cw[pos]:
                    continue
                new_word = cw[:]
                new_word[pos] = new_val
                covered.add(_word_to_idx(new_word, q))
    if radius >= 2:
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                for v1 in range(q):
                    if v1 == cw[pos1]:
                        continue
                    for v2 in range(q):
                        if v2 == cw[pos2]:
                            continue
                        new_word = cw[:]
                        new_word[pos1] = v1
                        new_word[pos2] = v2
                        covered.add(_word_to_idx(new_word, q))
    return covered


def _build_cover_matrix(codewords, q, n, radius=1):
    """Build a boolean matrix: rows = codewords, cols = all words in space.
    M[i, j] = True if codeword i covers word j."""
    total = q ** n
    m = len(codewords)
    # Precompute neighbor sets for each codeword
    neighbor_sets = []
    for cw in codewords:
        cw_idx = _word_to_idx(cw, q)
        neighbor_sets.append(_all_neighbors_indices(cw_idx, q, n, radius))
    
    # Build matrix using vectorized approach
    # For each codeword, mark which words it covers
    cover_matrix = np.zeros((m, total), dtype=bool)
    for i, ns in enumerate(neighbor_sets):
        for idx in ns:
            cover_matrix[i, idx] = True
    return cover_matrix


def _greedy_cover(cover_matrix, uncovered_mask):
    """Greedy: pick row with max coverage of currently uncovered words.
    Returns (best_row_idx, gain)."""
    # Number of uncovered words each row covers
    gains = cover_matrix[:, uncovered_mask].sum(axis=1)
    if gains.max() == 0:
        return -1, 0
    best = int(np.argmax(gains))
    return best, int(gains[best])


def _local_search_swap(code_idx_list, cover_matrix, uncovered_mask, q, n, radius, deadline, max_swaps=5000):
    """Try to reduce code size by removing codewords whose coverage is redundant,
    or swap a codeword for a better one."""
    m = len(code_idx_list)
    if m == 0:
        return code_idx_list, uncovered_mask
    
    current_code = list(code_idx_list)
    current_uncovered = uncovered_mask.copy()
    current_cover_matrix = cover_matrix
    
    # Try removals first
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Recompute coverage
        covered = np.zeros(q ** n, dtype=bool)
        for idx in current_code:
            covered |= current_cover_matrix[idx]
        current_uncovered = ~covered
        
        if not current_uncovered.any():
            break
        
        # Try removing each codeword
        for i in range(len(current_code)):
            if time.time() > deadline:
                return current_code, current_uncovered
            test_code = current_code[:i] + current_code[i+1:]
            if not test_code:
                continue
            test_covered = np.zeros(q ** n, dtype=bool)
            for idx in test_code:
                test_covered |= current_cover_matrix[idx]
            if test_covered.all():
                current_code = test_code
                improved = True
                break
    
    # Try swaps: replace one codeword with a non-code word that covers more
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        
        current_covered = np.zeros(q ** n, dtype=bool)
        for idx in current_code:
            current_covered |= current_cover_matrix[idx]
        current_uncovered = ~current_covered
        
        if not current_uncovered.any():
            break
        
        best_swap_gain = 0
        best_swap_remove = -1
        best_swap_add = -1
        
        # Sample a few positions to try removing
        n_remove = min(5, len(current_code))
        remove_candidates = random.sample(range(len(current_code)), n_remove)
        
        for ri in remove_candidates:
            if time.time() > deadline:
                break
            remove_idx = current_code[ri]
            
            # Coverage without this codeword
            without_covered = np.zeros(q ** n, dtype=bool)
            for j, idx in enumerate(current_code):
                if j != ri:
                    without_covered |= current_cover_matrix[idx]
            
            # Words that would become uncovered
            would_lose = ~without_covered
            
            if not would_lose.any():
                # Can just remove
                current_code = current_code[:ri] + current_code[ri+1:]
                break
            
            # Try to find a codeword that covers all of would_lose
            # and is not already in the code
            code_set = set(current_code)
            
            # Sample some candidates from uncovered words
            lose_indices = np.where(would_lose)[0]
            if len(lose_indices) > 30:
                sample = random.sample(lose_indices.tolist(), 30)
            else:
                sample = lose_indices.tolist()
            
            for cand_idx in sample:
                if cand_idx in code_set:
                    continue
                cand_cover = current_cover_matrix[cand_idx]
                # Check if cand covers all of would_lose
                if cand_cover[would_lose].all():
                    # Also check net gain: how many new words does cand cover beyond current
                    new_gain = (cand_cover & ~current_covered).sum()
                    if new_gain > best_swap_gain:
                        best_swap_gain = new_gain
                        best_swap_remove = ri
                        best_swap_add = cand_idx
        
        if best_swap_remove >= 0:
            current_code[best_swap_remove] = best_swap_add
        elif not current_uncovered.any():
            break
    
    return current_code, current_uncovered


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    # Special case: ternary, length 7, radius 1
    if q == 3 and n == 7 and radius == 1:
        total = 3 ** 7  # 2187
        
        # Start with direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=3)]
        
        # Convert to indices
        initial_idx = [_word_to_idx(w, q) for w in initial]
        unique_initial = list(set(initial_idx))
        
        # Build cover matrix for all possible codewords (2187 rows x 2187 cols)
        # This is 2187 * 22 = ~48k booleans per row, total ~2187*2187 = 4.78M booleans
        # Memory: ~4.78 MB, fine.
        
        # Precompute neighbor sets for ALL words
        all_neighbor_sets = []
        for idx in range(total):
            all_neighbor_sets.append(_all_neighbors_indices(idx, q, n, radius))
        
        # Build full cover matrix: cover_matrix[i, j] = True if word i covers word j
        cover_matrix = np.zeros((total, total), dtype=bool)
        for i in range(total):
            for j in all_neighbor_sets[i]:
                cover_matrix[i, j] = True
        
        # Check if initial set covers everything
        initial_covered = np.zeros(total, dtype=bool)
        for idx in unique_initial:
            initial_covered |= cover_matrix[idx]
        
        if initial_covered.all():
            return initial
        
        # Greedy: start with initial, try to add codewords to cover remaining
        current_code = list(unique_initial)
        current_covered = initial_covered.copy()
        
        # Greedy addition
        while True:
            if time.time() > deadline:
                break
            uncovered = ~current_covered
            if not uncovered.any():
                break
            
            # Find best codeword to add
            gains = cover_matrix[:, uncovered].sum(axis=1)
            # Exclude codewords already in the code
            code_set = set(current_code)
            for idx in code_set:
                gains[idx] = 0
            
            if gains.max() == 0:
                break
            
            best_idx = int(np.argmax(gains))
            current_code.append(best_idx)
            current_covered |= cover_matrix[best_idx]
        
        # Local search: try to reduce size
        if time.time() < deadline:
            current_code, current_uncovered = _local_search_swap(
                current_code, cover_matrix, ~current_covered, q, n, radius, deadline, max_swaps=3000
            )
        
        # Verify and return
        final_covered = np.zeros(total, dtype=bool)
        for idx in current_code:
            final_covered |= cover_matrix[idx]
        
        # Safety: if not fully covered, fall back to a complete cover
        if not final_covered.all():
            # Use greedy from scratch
            current_code = []
            current_covered = np.zeros(total, dtype=bool)
            while True:
                if time.time() > deadline:
                    break
                uncovered = ~current_covered
                if not uncovered.any():
                    break
                gains = cover_matrix[:, uncovered].sum(axis=1)
                for idx in current_code:
                    gains[idx] = 0
                if gains.max() == 0:
                    break
                best_idx = int(np.argmax(gains))
                current_code.append(best_idx)
                current_covered |= cover_matrix[best_idx]
        
        # Convert back to words
        result = [_idx_to_word(idx, q, n) for idx in current_code]
        # Ensure distinct
        seen = set()
        unique_result = []
        for w in result:
            key = tuple(w)
            if key not in seen:
                seen.add(key)
                unique_result.append(w)
        return unique_result
    
    # General case: direct sum construction
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return initial
    
    # Fallback: greedy from empty
    total = q ** n
    if total > 100000:
        # Too large, return a simple construction
        return [list(range(n)) for _ in range(1)]
    
    # Build cover matrix
    all_neighbor_sets = []
    for idx in range(total):
        all_neighbor_sets.append(_all_neighbors_indices(idx, q, n, radius))
    
    cover_matrix = np.zeros((total, total), dtype=bool)
    for i in range(total):
        for j in all_neighbor_sets[i]:
            cover_matrix[i, j] = True
    
    current_code = []
    current_covered = np.zeros(total, dtype=bool)
    
    while True:
        if time.time() > deadline:
            break
        uncovered = ~current_covered
        if not uncovered.any():
            break
        gains = cover_matrix[:, uncovered].sum(axis=1)
        for idx in current_code:
            gains[idx] = 0
        if gains.max() == 0:
            break
        best_idx = int(np.argmax(gains))
        current_code.append(best_idx)
        current_covered |= cover_matrix[best_idx]
    
    result = [_idx_to_word(idx, q, n) for idx in current_code]
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)
    return unique_result
# EVOLVE-BLOCK-END
