import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Handle general case fallback
    def get_fallback():
        if n == 4 and q == 3 and radius == 1:
            # Tetracode
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n > 4 and q == 3 and radius == 1:
            # Direct sum with tetracode
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # General greedy for other params
        return _general_greedy(q, n, radius, deadline)

    if q != 3 or n != 8 or radius != 1:
        return get_fallback()

    total_words = 3 ** 8
    num_neighbors = 1 + n * (q - 1)  # 17

    # Precompute neighbors
    def int_to_digits(num, n, q):
        res = [0] * n
        for i in range(n):
            res[i] = num % q
            num //= q
        return res

    # Build neighbor array: neighbors[w] = list of 17 words within distance 1 of w
    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)
    for w in range(total_words):
        digits = int_to_digits(w, n, q)
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[w, idx] = w + (new_val - orig_val) * (q ** pos)
                idx += 1

    # For fast set operations, we'll use a different approach:
    # Represent the code as a boolean array of size total_words
    # And use vectorized operations for coverage

    best_code = None
    best_size = total_words

    # Strategy: Iterated Local Search (ILS) with perturbation
    # 1. Greedy construction
    # 2. Local search to minimize size
    # 3. Perturb (remove some codewords, re-add greedily)
    # 4. Local search again
    # 5. Keep best, repeat

    def greedy_construct(uncovered_mask, max_codewords):
        """Greedy construction starting from uncovered_mask (boolean array where True = uncovered)."""
        selected = []
        covered = np.zeros(total_words, dtype=np.uint8)
        
        for _ in range(max_codewords):
            if time.time() > deadline - 5.0:
                break
            if not np.any(uncovered_mask):
                break
            
            # Compute gain for each word: number of currently uncovered neighbors
            # Vectorized: for each word w, gain[w] = sum(uncovered_mask[neighbors[w]])
            gains = uncovered_mask[neighbors].sum(axis=1)
            
            # Find word with max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Pick from top candidates with some randomness
            top_candidates = np.where(gains == max_gain)[0]
            if len(top_candidates) > 1:
                # Weighted random: prefer words that cover more uncovered
                # But for speed, just pick randomly from top
                best_idx = top_candidates[random.randint(0, len(top_candidates) - 1)]
            else:
                best_idx = top_candidates[0]
            
            selected.append(best_idx)
            # Update uncovered
            uncovered_mask[neighbors[best_idx]] = False
        
        return selected

    def local_search_reduce(selected, covered_counts):
        """Iteratively remove redundant codewords and try swaps to reduce size."""
        selected = list(selected)
        covered_counts = np.array(covered_counts, dtype=np.int32, copy=True)
        
        # Phase 1: Remove redundant codewords
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > deadline - 3.0:
                    break
                idx = selected[i]
                # Check if all neighbors of idx are covered by others
                if np.all(covered_counts[neighbors[idx]] > 1):
                    covered_counts[neighbors[idx]] -= 1
                    selected.pop(i)
                    changed = True
        
        # Phase 2: Try swaps - replace one codeword with another to allow further removals
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            # Try random swaps
            for _ in range(100):
                if time.time() > deadline - 3.0:
                    break
                if len(selected) < 2:
                    break
                
                # Pick a random codeword to try replacing
                i = random.randint(0, len(selected) - 1)
                idx = selected[i]
                
                # Find critical words (covered only by idx)
                crit_mask = (covered_counts[neighbors[idx]] == 1)
                crit_words = neighbors[idx][crit_mask]
                
                if len(crit_words) == 0:
                    # Can remove
                    covered_counts[neighbors[idx]] -= 1
                    selected.pop(i)
                    improved = True
                    break
                
                # Find a replacement that covers all critical words
                # A word c covers crit_words if all crit_words are in neighbors[c]
                # i.e., for all v in crit_words, c is in neighbors[v]
                # Equivalently, c is in the intersection of neighbors[v] for all v in crit_words
                
                # Efficient check: c must be a neighbor of each critical word
                # Let's find candidates: neighbors of the first critical word, minus idx
                first_crit = crit_words[0]
                candidates = neighbors[first_crit]
                
                valid_candidates = []
                for c in candidates:
                    if c == idx:
                        continue
                    # Check if c covers all critical words
                    # c covers v if v is in neighbors[c], i.e., c is in neighbors[v]
                    # So check: for all v in crit_words, c is in neighbors[v]
                    # Equivalent: neighbors[c] contains all crit_words
                    # Or: for each v in crit_words, v is in neighbors[c]
                    # Which means: c is in the "pre-image" - let's just check directly
                    ok = True
                    for v in crit_words:
                        # Check if v is in neighbors[c]
                        if not np.any(neighbors[c] == v):
                            ok = False
                            break
                    if ok:
                        valid_candidates.append(c)
                
                if valid_candidates:
                    c_prime = random.choice(valid_candidates)
                    # Perform swap
                    covered_counts[neighbors[idx]] -= 1
                    covered_counts[neighbors[c_prime]] += 1
                    selected[i] = c_prime
                    improved = True
                    
                    # After swap, try to remove more
                    for j in range(len(selected) - 1, -1, -1):
                        if time.time() > deadline - 3.0:
                            break
                        idx2 = selected[j]
                        if np.all(covered_counts[neighbors[idx2]] > 1):
                            covered_counts[neighbors[idx2]] -= 1
                            selected.pop(j)
                            improved = True
                    break
        
        return selected, covered_counts

    def compute_coverage(selected):
        """Compute coverage counts for a given list of codewords."""
        covered = np.zeros(total_words, dtype=np.int32)
        for idx in selected:
            covered[neighbors[idx]] += 1
        return covered

    # Main ILS loop
    seed = 42
    num_restarts = 0
    while time.time() < deadline - 5.0:
        num_restarts += 1
        random.seed(seed)
        seed += 1
        
        # Greedy construction
        uncovered_mask = np.ones(total_words, dtype=bool)
        selected = greedy_construct(uncovered_mask, best_size - 1 if best_code else total_words)
        
        if not selected:
            continue
        
        # Compute coverage
        covered_counts = compute_coverage(selected)
        
        # Verify coverage
        if not np.all(covered_counts > 0):
            # Some words uncovered, add more greedily
            uncovered_idx = np.where(covered_counts == 0)[0]
            while len(uncovered_idx) > 0 and time.time() < deadline - 5.0:
                # Pick word covering most uncovered
                gains = (covered_counts == 0)[neighbors].sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
                selected.append(best_idx)
                covered_counts[neighbors[best_idx]] += 1
                uncovered_idx = np.where(covered_counts == 0)[0]
        
        if len(selected) >= best_size:
            continue
        
        # Local search
        selected, covered_counts = local_search_reduce(selected, covered_counts)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]
        
        # Perturbation for ILS: remove some random codewords and re-add
        if time.time() < deadline - 10.0 and best_code:
            perturbed = list(best_code)
            # Remove 2-5 random codewords
            num_remove = random.randint(2, min(5, len(perturbed) // 10 + 2))
            for _ in range(num_remove):
                if perturbed:
                    i = random.randint(0, len(perturbed) - 1)
                    perturbed.pop(i)
            
            # Recompute coverage
            covered_counts = compute_coverage(perturbed)
            
            # Add back greedily until fully covered
            uncovered_idx = np.where(covered_counts == 0)[0]
            while len(uncovered_idx) > 0 and time.time() < deadline - 5.0:
                gains = (covered_counts == 0)[neighbors].sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
                perturbed.append(best_idx)
                covered_counts[neighbors[best_idx]] += 1
                uncovered_idx = np.where(covered_counts == 0)[0]
            
            # Local search on perturbed
            perturbed, covered_counts = local_search_reduce(perturbed, covered_counts)
            
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed[:]

    if best_code is None:
        return get_fallback()

    return [list(int_to_digits(i, n, q)) for i in best_code]

def _general_greedy(q, n, radius, deadline):
    total_words = q ** n
    neighbors = []
    for w in range(total_words):
        w_list = []
        rem = w
        for _ in range(n):
            w_list.append(rem % q)
            rem //= q
        w_list.reverse()
        neigh = [w]
        for pos in range(n):
            orig = w_list[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                new_list = w_list[:]
                new_list[pos] = nv
                new_w = 0
                for i in range(n):
                    new_w = new_w * q + new_list[i]
                neigh.append(new_w)
        neighbors.append(neigh)
    
    uncovered = [1] * total_words
    selected = []
    remaining = total_words
    
    while remaining > 0:
        if time.time() > deadline:
            break
        max_gain = -1
        best_idx = -1
        for i in range(total_words):
            if uncovered[i] == 0:
                continue
            gain = sum(uncovered[nb] for nb in neighbors[i])
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        selected.append(best_idx)
        for nb in neighbors[best_idx]:
            if uncovered[nb] > 0:
                uncovered[nb] -= 1
                if uncovered[nb] == 0:
                    remaining -= 1
    
    # Prune
    coverage = [0] * total_words
    for idx in selected:
        for nb in neighbors[idx]:
            coverage[nb] += 1
    
    changed = True
    while changed:
        changed = False
        new_sel = []
        for idx in selected:
            removable = True
            for nb in neighbors[idx]:
                if coverage[nb] == 1:
                    removable = False
                    break
            if removable:
                for nb in neighbors[idx]:
                    coverage[nb] -= 1
                changed = True
            else:
                new_sel.append(idx)
        selected = new_sel
    
    return [[(i // (q ** (n - 1 - j))) % q for j in range(n)] for i in selected]
