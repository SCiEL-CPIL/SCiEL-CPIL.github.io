"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    # Deadline guard: 150s wall-clock limit, keep 10s margin
    start_time = time.time()
    deadline = start_time + 140.0

    total_words = q ** n

    # If the space is small enough for a fast greedy, use it.
    # For q=3, n=6, radius=1: 729 words, sphere size 13. Very fast.
    # For larger spaces, fall back to the direct-sum construction.
    if total_words <= 100000:
        # --- Fast NumPy greedy set cover ---
        # Precompute neighbors for each index (radius-1 neighbors)
        if radius == 1:
            sphere_size = 1 + n * (q - 1)
            # Build neighbor index table: neighbors[i] = list of indices within dist 1 of i
            neighbors = np.empty((total_words, sphere_size), dtype=np.int32)
            for i in range(total_words):
                nbrs = [i]
                for pos in range(n):
                    pow_q = q ** (n - 1 - pos)
                    current_val = (i // pow_q) % q
                    for new_val in range(q):
                        if new_val != current_val:
                            nbrs.append(i + (new_val - current_val) * pow_q)
                neighbors[i] = nbrs

            covered = np.zeros(total_words, dtype=bool)
            code_indices = []

            while time.time() < deadline:
                # Gains: how many uncovered words each candidate covers
                gains = (~covered)[neighbors].sum(axis=1)
                best_idx = int(np.argmax(gains))
                if gains[best_idx] == 0:
                    break
                code_indices.append(best_idx)
                covered[neighbors[best_idx]] = True

            # Convert indices to codewords
            result = []
            for idx in code_indices:
                w = []
                temp = idx
                for _ in range(n):
                    w.append(temp % q)
                    temp //= q
                w.reverse()
                result.append(w)

            # Local improvement pass: try to remove redundant codewords
            # and swap to better positions to reduce total count
            if time.time() < deadline:
                result = _local_improve(result, q, n, radius, neighbors, deadline)

            # Verify coverage (safety check)
            if _verify_coverage(result, q, n, radius):
                return result
            # If verification fails (shouldn't happen), fall through to fallback

        else:
            # Generic radius: use a simpler greedy with itertools for small spaces
            if total_words <= 10000:
                return _greedy_generic(q, n, radius, deadline)
            # Otherwise fall through to direct sum

    # --- Fallback: Direct sum construction ---
    # For q=3, n>=4, radius=1: tetracode (9 words of length 4) × all tails
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_n = 4
        base_code = tetracode
    elif q >= 3 and n >= 4 and radius == 1:
        # Generic tetracode-like code over q
        base_code = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        base_n = 4
    else:
        # Last resort: use a simple covering
        base_n = min(n, 4)
        base_code = [list(range(base_n))]
        # Pad to ensure coverage isn't trivially broken; this is a weak fallback
        base_code = []
        for tail in itertools.product(range(q), repeat=base_n):
            base_code.append(list(tail))
        base_n = base_n

    if n < base_n:
        return [list(range(n))]

    tails = list(itertools.product(range(q), repeat=n - base_n))
    return [h + list(tail) for h in base_code for tail in tails]


def _local_improve(codewords, q, n, radius, neighbors, deadline):
    """Local improvement: remove redundant codewords and try swaps to reduce count."""
    if len(codewords) < 2:
        return codewords
    
    total_words = q ** n
    code_indices = []
    for cw in codewords:
        idx = 0
        for i, v in enumerate(cw):
            idx += v * (q ** (n - 1 - i))
        code_indices.append(idx)
    
    # Build a set of covered indices for current code
    covered = np.zeros(total_words, dtype=bool)
    for idx in code_indices:
        covered[neighbors[idx]] = True
    
    # Try to remove redundant codewords
    improved = True
    while improved:
        improved = False
        for i in range(len(code_indices)):
            if time.time() > deadline:
                break
            # Temporarily remove codeword i
            idx_i = code_indices[i]
            saved_covered = covered.copy()
            covered[neighbors[idx_i]] = False
            
            # Check if all words still covered
            if covered.all():
                # It was redundant, remove it
                code_indices.pop(i)
                improved = True
                break  # Restart the loop since indices changed
            else:
                # Restore
                covered = saved_covered
    
    # Try swaps: replace one codeword with another that covers more uncovered
    # This is a 2-opt style local search
    for _ in range(50):  # Limit iterations for time
        if time.time() > deadline:
            break
        best_swap = None
        best_gain = 0
        
        # Try removing each codeword and replacing with a better one
        for i in range(len(code_indices)):
            if time.time() > deadline:
                break
            idx_i = code_indices[i]
            # Words that would become uncovered if we remove idx_i
            # These are words where idx_i is the only cover
            saved_covered = covered.copy()
            covered[neighbors[idx_i]] = False
            
            if not covered.all():
                # Find uncovered words
                uncovered = np.where(~covered)[0]
                # Find a candidate that covers the most uncovered
                # Only check candidates not already in code
                in_code = set(code_indices)
                gains = (~covered)[neighbors].sum(axis=1)
                # Zero out gains for words already in code
                for idx in in_code:
                    gains[idx] = 0
                
                best_candidate = int(np.argmax(gains))
                candidate_gain = gains[best_candidate]
                
                if candidate_gain > best_gain and candidate_gain > 0:
                    best_gain = candidate_gain
                    best_swap = (i, best_candidate)
                
                covered = saved_covered
        
        if best_swap is not None:
            i, new_idx = best_swap
            code_indices[i] = new_idx
            # Update covered
            covered = np.zeros(total_words, dtype=bool)
            for idx in code_indices:
                covered[neighbors[idx]] = True
        else:
            break
    
    # Convert back to codewords
    result = []
    for idx in code_indices:
        w = []
        temp = idx
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
    
    return result


def _verify_coverage(codewords, q, n, radius):
    """Check that every word is within Hamming distance radius of some codeword."""
    total_words = q ** n
    # Build a set of covered indices
    covered = set()
    for cw in codewords:
        idx = 0
        for i, v in enumerate(cw):
            idx += v * (q ** (n - 1 - i))
        # Add all words within distance radius
        if radius == 1:
            covered.add(idx)
            for pos in range(n):
                pow_q = q ** (n - 1 - pos)
                cur = (idx // pow_q) % q
                for nv in range(q):
                    if nv != cur:
                        covered.add(idx + (nv - cur) * pow_q)
        else:
            # Generic: generate all words within distance radius
            for delta in _all_deltas(n, q, radius):
                covered.add(idx + delta)
    return len(covered) == total_words


def _all_deltas(n, q, radius):
    """Yield index deltas for all words within Hamming distance radius of the zero word."""
    # For each subset of positions of size <= radius, and each non-zero change
    deltas = set()
    deltas.add(0)
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            # For each position, choose a non-zero delta value
            base_options = []
            for pos in positions:
                pow_q = q ** (n - 1 - pos)
                base_options.append([d * pow_q for d in range(1, q)])
            for combo in itertools.product(*base_options):
                deltas.add(sum(combo))
    return deltas


def _greedy_generic(q, n, radius, deadline):
    """Generic greedy for small spaces and any radius."""
    total_words = q ** n
    # Precompute neighbors
    neighbors_list = []
    for i in range(total_words):
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        nbrs = _neighbors_of_word(w, q, n, radius)
        # Convert back to indices
        idx_nbrs = []
        for nb in nbrs:
            idx = 0
            for j, v in enumerate(nb):
                idx += v * (q ** (n - 1 - j))
            idx_nbrs.append(idx)
        neighbors_list.append(idx_nbrs)

    covered = [False] * total_words
    code_indices = []

    while time.time() < deadline:
        best_idx = -1
        best_gain = 0
        for i in range(total_words):
            gain = sum(1 for nb in neighbors_list[i] if not covered[nb])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        code_indices.append(best_idx)
        for nb in neighbors_list[best_idx]:
            covered[nb] = True

    result = []
    for idx in code_indices:
        w = []
        temp = idx
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
    return result


def _neighbors_of_word(word, q, n, radius):
    """Return all words within Hamming distance <= radius of the given word."""
    result = [list(word)]
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            base = list(word)
            for combo in itertools.product(*[range(q)] * r):
                for pos, val in zip(positions, combo):
                    if val != base[pos]:
                        base[pos] = val
                # Only include if different from original
                if base != list(word):
                    result.append(list(base))
            # Reset base for next combination of positions
            base = list(word)
    return result
# EVOLVE-BLOCK-END
