import itertools
import time
import numpy as np
import random

def construct(q, n, radius):
    # Deadline guard: 150s wall-clock limit, keep 10s margin
    start_time = time.time()
    deadline = start_time + 140.0

    total_words = q ** n

    # If the space is small enough for a fast greedy, use it.
    if total_words <= 100000:
        if radius == 1:
            result = _fast_greedy_q1(q, n, deadline)
            if result is not None:
                return result
        else:
            if total_words <= 10000:
                result = _greedy_generic(q, n, radius, deadline)
                if result is not None:
                    return result

    # --- Fallback: Direct sum construction ---
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_n = 4
        base_code = tetracode
    elif q >= 3 and n >= 4 and radius == 1:
        base_code = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        base_n = 4
    else:
        base_n = min(n, 4)
        base_code = []
        for tail in itertools.product(range(q), repeat=base_n):
            base_code.append(list(tail))

    if n < base_n:
        return [list(range(n))]

    tails = list(itertools.product(range(q), repeat=n - base_n))
    return [h + list(tail) for h in base_code for tail in tails]

def _fast_greedy_q1(q, n, deadline):
    """Optimized greedy + local search with simulated annealing swaps for q-ary, radius=1 covering."""
    total_words = q ** n
    sphere_size = 1 + n * (q - 1)

    # Build neighbor table
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

    # Greedy set cover
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []

    while time.time() < deadline:
        # Calculate gain for each word: how many uncovered neighbors it has
        # We can optimize this by maintaining uncovered set, but for 729 words, simple numpy is fine
        gains = (~covered)[neighbors].sum(axis=1)
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        code_indices.append(best_idx)
        covered[neighbors[best_idx]] = True

    # Convert to list
    code_indices = list(code_indices)
    
    # --- Local Search Optimization ---
    # State: code_indices (list of int), cover_count (np.array of int)
    
    def compute_coverage_count(codewords):
        """Return array where cov_count[w] = number of codewords covering word w."""
        cov_count = np.zeros(total_words, dtype=np.int32)
        for ci in codewords:
            cov_count[neighbors[ci]] += 1
        return cov_count

    def try_remove(codewords, cov_count):
        """Try to remove any redundant codewords. Returns (new_codewords, new_cov_count, changed)"""
        changed = False
        for i in range(len(codewords) - 1, -1, -1):
            ci = codewords[i]
            # Check if any word in neighbors[ci] is covered only by ci
            # If min(cover_count[neighbors[ci]]) >= 2, we can remove
            if np.min(cov_count[neighbors[ci]]) >= 2:
                codewords.pop(i)
                cov_count[neighbors[ci]] -= 1
                changed = True
        return codewords, cov_count, changed

    def local_search(codewords, cov_count, time_limit):
        """Perform local search with swaps."""
        while time.time() < time_limit:
            # First, remove all redundant codewords
            codewords, cov_count, changed = try_remove(codewords, cov_count)
            
            if not changed:
                # Try swaps
                swap_made = False
                # Sort codewords to make iteration deterministic or just iterate
                # We want to find a swap (i, j) such that removing i and adding j is valid.
                # Condition: UniqueCoverers(i) subset of neighbors(j)
                
                # Precompute unique coverers for each codeword in the current set?
                # That's expensive. Instead, for each i, check if it can be removed if replaced by j.
                
                # Optimization: 
                # 1. Identify "critical" codewords? No, just try swaps.
                # 2. To speed up, we can check if codeword i is "essential" (cannot be removed even with replacement by any j? No, that's the goal).
                
                # Let's try random swaps or structured swaps.
                # Structured: For each i in codewords, find best j.
                # This is O(N^2 * SphereSize). N~80, SphereSize=13. 80^2 * 13 = 83200 ops. Very fast.
                
                for i in range(len(codewords)):
                    ci = codewords[i]
                    # Words that are uniquely covered by ci
                    # We need to check if there exists j not in codewords such that
                    # {w | cover_count[w] == 1 and w in neighbors[ci]} subset of neighbors[j]
                    
                    # Let U_i = {w | w in neighbors[ci] and cover_count[w] == 1}
                    # If U_i is empty, we can just remove ci (handled by try_remove).
                    # So U_i is not empty.
                    
                    # We need to find j such that U_i subset of neighbors[j].
                    # This means for all w in U_i, j must be a neighbor of w.
                    # i.e., j must be in intersection of neighbors[w] for all w in U_i.
                    
                    # Let's compute the intersection of neighbors for all w in U_i.
                    # If U_i is large, the intersection might be empty.
                    
                    # Efficient check:
                    # For a candidate j, is U_i subset of neighbors[j]?
                    # Equivalent to: min_{w in U_i} (1 if w in neighbors[j] else 0) == 1.
                    # Or: sum_{w in U_i} (w in neighbors[j]) == |U_i|.
                    
                    # Let's iterate over all possible j (0..total_words-1) not in codewords.
                    # To speed up, we can filter j.
                    # j must be a neighbor of at least one w in U_i? No, j must be neighbor of ALL w in U_i.
                    # So j must be in intersection_{w in U_i} neighbors[w].
                    
                    # Let's compute the set of valid j's.
                    # Start with all j's.
                    # For each w in U_i, valid_j = valid_j intersection neighbors[w].
                    
                    # Since |U_i| is small (usually 1 or 2), this is fast.
                    
                    # Get U_i indices
                    u_i_indices = [w for w in neighbors[ci] if cov_count[w] == 1]
                    
                    if not u_i_indices:
                        # Should have been removed by try_remove
                        continue
                    
                    # Find intersection of neighbors for w in u_i_indices
                    # Initialize valid_j_mask or set
                    # Using a boolean array for speed? Or just a set.
                    # Set intersection is fast enough for small sets.
                    
                    valid_j_set = None
                    for w in u_i_indices:
                        nbrs_w = set(neighbors[w].tolist())
                        if valid_j_set is None:
                            valid_j_set = nbrs_w
                        else:
                            valid_j_set = valid_j_set.intersection(nbrs_w)
                    
                    if not valid_j_set:
                        continue
                    
                    # Remove current codewords from valid_j_set
                    current_code_set = set(codewords)
                    valid_j_set -= current_code_set
                    
                    if not valid_j_set:
                        continue
                    
                    # Pick a j from valid_j_set
                    # Heuristic: Pick j that maximizes new coverage? Or just any?
                    # Any valid j maintains validity. We want to enable future removals.
                    # Let's pick the one that covers the most "weakly covered" words?
                    # Or just random.
                    # Let's try to pick the one that minimizes the number of unique coverers it creates?
                    # For now, pick the first one.
                    j = next(iter(valid_j_set))
                    
                    # Perform swap
                    # Remove i, add j
                    # Update cov_count
                    cov_count[neighbors[ci]] -= 1
                    cov_count[neighbors[j]] += 1
                    codewords[i] = j
                    swap_made = True
                    break # Restart the loop with the new state
                
                if not swap_made:
                    # No swap found, try random perturbation or break
                    # If we are stuck, we might need to add a word and remove two? (2-opt)
                    # For now, break to avoid infinite loop if no swaps are possible.
                    # But we might be in a local minimum.
                    # Let's try a random restart of the local search part?
                    # Or just accept the current solution.
                    break
        
        return codewords, cov_count

    # Run initial removal pass
    cov_count = compute_coverage_count(code_indices)
    code_indices, cov_count, _ = try_remove(code_indices, cov_count)
    
    # Run local search
    local_search_deadline = deadline - 5
    code_indices, cov_count = local_search(code_indices, cov_count, local_search_deadline)
    
    # Final removal pass
    code_indices, cov_count, _ = try_remove(code_indices, cov_count)
    
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

    # Verify coverage (safety check)
    if _verify_coverage(result, q, n, 1):
        return result
    return None


def _verify_coverage(codewords, q, n, radius):
    """Check that every word is within Hamming distance radius of some codeword."""
    total_words = q ** n
    if radius == 1:
        covered = np.zeros(total_words, dtype=bool)
        for cw in codewords:
            idx = 0
            for i, v in enumerate(cw):
                idx += v * (q ** (n - 1 - i))
            covered[idx] = True
            for pos in range(n):
                pow_q = q ** (n - 1 - pos)
                cur = (idx // pow_q) % q
                for nv in range(q):
                    if nv != cur:
                        covered[idx + (nv - cur) * pow_q] = True
        return covered.all()
    else:
        # Generic: generate all words within distance radius
        covered = set()
        for cw in codewords:
            idx = 0
            for i, v in enumerate(cw):
                idx += v * (q ** (n - 1 - i))
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

