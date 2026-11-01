import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q != 3 or n != 6 or radius != 1:
        return _greedy_set_cover(q, n, radius, deadline)

    space_size = q ** n  # 729

    # Precompute index <-> word conversions
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors (distance <= radius) for every word in the space
    # neighbors[i] is a set of indices at distance <= 1 from i
    neighbors = []
    for i in range(space_size):
        w = idx_to_word(i)
        nbs = {i}  # include self
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    nbs.add(word_to_idx(w))
            w[pos] = orig
        neighbors.append(nbs)

    # Also build a list version for iteration
    neighbors_list = [list(nb) for nb in neighbors]

    # Tetracode direct-sum initial construction: 9 * 3^(6-4) = 81 words
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    initial_codewords = [h + list(tail) for h in tetracode for tail in tails]
    
    # Convert to indices
    initial_indices = [word_to_idx(w) for w in initial_codewords]

    def build_counts(indices):
        """Build coverage count array from a list of codeword indices."""
        counts = np.zeros(space_size, dtype=np.int32)
        for ci in indices:
            counts[neighbors_list[ci]] += 1
        return counts

    def greedy_remove(cw_indices, counts, deadline):
        """Iteratively remove redundant codewords. Returns (list, counts)."""
        current = list(cw_indices)
        cnt = counts.copy()
        current_set = set(current)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    break
                ci = current[i]
                can_remove = True
                for nb in neighbors_list[ci]:
                    if cnt[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors_list[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    current_set.discard(ci)
                    changed = True
        return current, cnt

    def local_search_swap(cw_indices, counts, deadline, max_iterations=50000):
        """
        Local search: try swaps to reduce size.
        We iterate over codewords and try to replace them with a random other word
        if it allows for a net reduction or enables further removals.
        Actually, a simpler and effective heuristic for covering codes is:
        1. Try to remove any redundant word.
        2. If no word is redundant, try a 2-opt swap: remove word A, add word B.
           Accept if it leads to a state where more words can be removed.
        """
        current = list(cw_indices)
        cnt = counts.copy()
        current_set = set(current)
        
        # We will use a randomized local search
        # Objective: Minimize |current|
        # Move: Remove i, Add j (if j not in set). 
        # Constraint: Coverage must remain valid (all counts >= 1).
        
        # To make this fast, we only consider moves that keep the solution valid.
        # A swap (remove i, add j) is valid if:
        # 1. For all nb in neighbors[i]: cnt[nb] > 1 OR nb in neighbors[j]
        # 2. For all nb in neighbors[j]: cnt[nb] >= 0 (always true)
        #    Actually, if we remove i, cnt[nb] decreases. If cnt[nb] becomes 0, it's invalid unless nb is covered by j.
        
        rng = random.Random(42)
        improved = True
        iterations = 0
        
        while improved and time.time() < deadline and iterations < max_iterations:
            improved = False
            iterations += 1
            # Shuffle to break symmetry and avoid local minima
            if iterations % 100 == 0:
                rng.shuffle(current)
                # Rebuild set if we shuffled? No, set is independent of order.
                
            # Phase 1: Greedy Removal
            removal_made = False
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    return current, cnt
                ci = current[i]
                can_remove = True
                for nb in neighbors_list[ci]:
                    if cnt[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors_list[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    current_set.discard(ci)
                    improved = True
                    removal_made = True
                    break # Restart scan after removal
            
            if removal_made:
                continue

            # Phase 2: Try Swaps
            # Pick a random codeword to try to replace
            if not current:
                break
                
            idx_to_remove = rng.randint(0, len(current) - 1)
            ci = current[idx_to_remove]
            
            # Find a candidate to add
            # We want a word that covers the "essential" neighbors of ci
            # Essential neighbors: nb in neighbors[ci] where cnt[nb] == 1
            essential = [nb for nb in neighbors_list[ci] if cnt[nb] == 1]
            
            if not essential:
                # If no essential neighbors, it should have been removed in Phase 1.
                # But due to random order or timing, maybe not. Try to remove it.
                for nb in neighbors_list[ci]:
                    cnt[nb] -= 1
                current.pop(idx_to_remove)
                current_set.discard(ci)
                improved = True
                continue

            # Try to find a word j that covers all essential neighbors
            # And does not uncover any other word.
            # A word j is a valid replacement if:
            # 1. j not in current_set
            # 2. For all e in essential: e in neighbors[j]
            # 3. For all nb in neighbors[ci] \ essential: cnt[nb] > 1 OR nb in neighbors[j]
            
            found_swap = False
            # Try a few random candidates
            for _ in range(20):
                if time.time() > deadline:
                    return current, cnt
                cand = rng.randint(0, space_size - 1)
                if cand in current_set:
                    continue
                
                cand_nbs = neighbors_list[cand]
                cand_set = neighbors[cand] # Use set for fast lookup
                
                # Check essential coverage
                if not all(e in cand_set for e in essential):
                    continue
                
                # Check no new holes
                valid = True
                for nb in neighbors_list[ci]:
                    if cnt[nb] == 1 and nb not in cand_set:
                        valid = False
                        break
                    # If cnt[nb] > 1, it's fine even if not in cand_set
                    # If cnt[nb] == 1, it MUST be in cand_set (handled above for essential, 
                    # but what if nb is not essential but cnt==1? 
                    # Wait, essential IS the set of nb where cnt==1. 
                    # So if nb is not in essential, cnt[nb] > 1. So it's safe to lose coverage from ci.
                
                if valid:
                    # Perform swap
                    for nb in neighbors_list[ci]:
                        cnt[nb] -= 1
                    for nb in cand_nbs:
                        cnt[nb] += 1
                    
                    current[idx_to_remove] = cand
                    current_set.discard(ci)
                    current_set.add(cand)
                    improved = True
                    found_swap = True
                    break
            
            if not found_swap and not improved:
                # Stuck in local minimum? 
                # We can try a random perturbation: remove a random word, add a random word, 
                # even if it temporarily increases size? 
                # No, we must maintain validity. 
                # Let's just break and rely on restarts.
                break

        return current, cnt

    # Main search loop
    best_size = float('inf')
    best_code = None
    
    # 1. Initial greedy reduction
    counts = build_counts(initial_indices)
    code, counts = greedy_remove(initial_indices, counts, deadline - 60)
    
    # 2. Local search with swaps
    code, counts = local_search_swap(code, counts, deadline - 20, max_iterations=100000)
    
    if len(code) < best_size:
        best_size = len(code)
        best_code = code.copy()

    # 3. Randomized restarts with different initial orderings or perturbations
    # We will try to improve the best solution found so far
    if time.time() < deadline - 10 and best_size > 73:
        # Try to improve best_code
        current_best = best_code.copy()
        current_counts = build_counts(current_best)
        
        # Randomize the order of removals/swaps
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Perturb: randomly swap a few codewords
            perturbed = current_best.copy()
            perturbed_counts = current_counts.copy()
            
            # Try a few random swaps to escape local minimum
            rng = random.Random(restart)
            for _ in range(5):
                if len(perturbed) < 2:
                    break
                i = rng.randint(0, len(perturbed) - 1)
                ci = perturbed[i]
                # Try to replace ci with a random word
                for _ in range(10):
                    cand = rng.randint(0, space_size - 1)
                    if cand in set(perturbed):
                        continue
                    cand_set = neighbors[cand]
                    essential = [nb for nb in neighbors_list[ci] if perturbed_counts[nb] == 1]
                    if not all(e in cand_set for e in essential):
                        continue
                    valid = True
                    for nb in neighbors_list[ci]:
                        if perturbed_counts[nb] == 1 and nb not in cand_set:
                            valid = False
                            break
                    if valid:
                        # Apply swap
                        for nb in neighbors_list[ci]:
                            perturbed_counts[nb] -= 1
                        for nb in neighbors_list[cand]:
                            perturbed_counts[nb] += 1
                        perturbed[i] = cand
                        break
            
            # Run local search on perturbed state
            improved_code, improved_counts = local_search_swap(perturbed, perturbed_counts, deadline - 5, max_iterations=20000)
            
            if len(improved_code) < best_size:
                best_size = len(improved_code)
                best_code = improved_code.copy()
                current_best = best_code.copy()
                current_counts = build_counts(current_best)

    # Final polish
    if best_code is not None and time.time() < deadline - 2:
        final_counts = build_counts(best_code)
        final_code, final_counts = greedy_remove(best_code, final_counts, deadline - 1)
        final_code, final_counts = local_search_swap(final_code, final_counts, deadline, max_iterations=5000)
        if len(final_code) < best_size:
            best_code = final_code
            best_size = len(final_code)

    # Convert back to list of lists
    result = [idx_to_word(ci) for ci in best_code]
    return result

def _greedy_set_cover(q, n, radius, deadline):
    """Greedy set cover for general parameters."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n

    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    # Precompute coverage sets
    coverage = {}
    for i, w in enumerate(all_words):
        covered = set()
        for j, v in enumerate(all_words):
            if hamming_dist(w, v) <= radius:
                covered.add(j)
        coverage[i] = covered

    uncovered = set(range(total))
    code = []

    while uncovered and time.time() < deadline - 5:
        best_i = -1
        best_count = 0
        for i in range(total):
            count = len(uncovered & coverage[i])
            if count > best_count:
                best_count = count
                best_i = i
            if time.time() > deadline - 5:
                break
        if best_i == -1:
            break
        code.append(list(all_words[best_i]))
        uncovered -= coverage[best_i]

    return code
