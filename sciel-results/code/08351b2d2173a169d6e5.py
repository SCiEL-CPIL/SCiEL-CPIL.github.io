import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        # Simple greedy for other cases
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        neighbors = [set() for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_word = list(w)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    neighbors[i].add(word_to_idx[new_word])
        
        uncovered = set(range(N))
        solution = []
        while uncovered:
            best_c = -1
            best_gain = -1
            for c in range(N):
                gain = len(neighbors[c] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            solution.append(best_c)
            uncovered -= neighbors[best_c]
        return [list(all_words[i]) for i in solution]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbors as lists of indices
    neighbors = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors.append(nb)
    
    # Convert to numpy arrays for faster indexing if needed, but lists are fine for small N
    # Actually, for fast "is subset" checks, sets are better, but we only have 15 neighbors.
    # Let's keep them as lists for memory efficiency and simple iteration.

    def greedy_initial():
        """Greedy Set Cover to get a good starting point."""
        uncovered_mask = np.ones(N, dtype=bool)
        code = []
        while uncovered_mask.any():
            best_c = -1
            best_count = 0
            # Optimization: Only check candidates that cover at least one uncovered word?
            # For N=2187, scanning all is ~2187 * 15 ops = ~32k ops. Very fast.
            for c in range(N):
                if not uncovered_mask[c]: # If c is already covered, it's still a valid candidate to add
                    # Actually, adding a covered word is fine, it might cover others.
                    pass
                # Count how many uncovered words c covers
                # neighbors[c] is a list of 15 indices
                # We need sum(uncovered_mask[nb] for nb in neighbors[c])
                # Using numpy slice is fast
                cnt = np.sum(uncovered_mask[neighbors[c]])
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                if best_count == np.sum(uncovered_mask):
                    break
            code.append(best_c)
            uncovered_mask[neighbors[best_c]] = False
        return code

    def build_coverage(code):
        coverage = np.zeros(N, dtype=np.int32)
        for c in code:
            coverage[neighbors[c]] += 1
        return coverage

    def prune(code, coverage):
        """Remove redundant codewords."""
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                c = code[i]
                # Check if c is redundant: all neighbors have coverage > 1
                # neighbors[c] has 15 elements
                if np.min(coverage[neighbors[c]]) > 1:
                    code.pop(i)
                    coverage[neighbors[c]] -= 1
                    changed = True
                    break
        return code

    def try_1_kick(code, coverage, deadline):
        """Try to remove one codeword and replace with one if necessary."""
        if not code:
            return False
        idx = random.randint(0, len(code) - 1)
        c_remove = code[idx]
        
        # Temporarily remove
        code.pop(idx)
        coverage[neighbors[c_remove]] -= 1
        
        # Check if valid
        # Find uncovered words
        # We can check if any neighbor of c_remove has coverage 0?
        # No, removing c_remove might uncover words that were ONLY covered by c_remove.
        # These words are in neighbors[c_remove].
        # But it's possible that a word NOT in neighbors[c_remove] was covered by c_remove?
        # No, coverage is defined by neighbors.
        # So only words in neighbors[c_remove] are affected.
        
        uncovered_in_neighbors = []
        for nb in neighbors[c_remove]:
            if coverage[nb] == 0:
                uncovered_in_neighbors.append(nb)
        
        if not uncovered_in_neighbors:
            # Valid removal, size decreased by 1
            return True
        
        # Need to repair. Find a codeword c_add that covers all uncovered_in_neighbors
        # And ideally covers some others to be efficient, but primarily must cover these.
        best_c_add = -1
        best_extra = -1
        
        for c_add in range(N):
            if c_add in code:
                continue
            # Check if c_add covers all uncovered_in_neighbors
            # uncovered_in_neighbors is a list of indices
            # We need to check if set(uncovered_in_neighbors) is subset of set(neighbors[c_add])
            # Since len(uncovered_in_neighbors) is small (<= 15), and neighbors[c_add] is 15.
            # We can do:
            nb_set = set(neighbors[c_add])
            if all(u in nb_set for u in uncovered_in_neighbors):
                # It covers the required ones.
                # Count how many OTHER uncovered words it covers (to be greedy)
                # Actually, we just want to restore validity.
                # To improve, we prefer one that covers more currently uncovered words (if any exist elsewhere? No, only these were uncovered).
                # Wait, if we removed c_remove, only its neighbors could become uncovered.
                # So we just need to cover these.
                # Let's pick the first one or the one with max coverage of these?
                # They all cover all of them.
                best_c_add = c_add
                break
        
        if best_c_add != -1:
            # Add it back
            code.append(best_c_add)
            coverage[neighbors[best_c_add]] += 1
            # Size is same as before kick. We didn't improve size, but we might have changed the structure.
            # For "kick" to be useful in local search, we usually accept if it leads to further improvements.
            # However, standard kick accepts if valid. If size didn't decrease, it's a neutral move.
            # Let's accept neutral moves to escape local minima.
            return True
        else:
            # Rollback
            code.append(c_remove)
            coverage[neighbors[c_remove]] += 1
            return False

    def try_2_kick(code, coverage, deadline):
        """Try to remove two codewords and replace with one or two."""
        if len(code) < 2:
            return False
        i1 = random.randint(0, len(code) - 1)
        i2 = random.randint(0, len(code) - 1)
        if i1 == i2:
            i2 = (i2 + 1) % len(code)
            
        c1 = code[i1]
        c2 = code[i2]
        
        # Remove both
        # Be careful with indices if we pop
        if i1 > i2:
            i1, i2 = i2, i1
        code.pop(i1)
        code.pop(i2)
        
        coverage[neighbors[c1]] -= 1
        coverage[neighbors[c2]] -= 1
        
        # Find uncovered words
        # Affected words are in neighbors[c1] U neighbors[c2]
        affected = set(neighbors[c1]) | set(neighbors[c2])
        uncovered = [w for w in affected if coverage[w] == 0]
        
        if not uncovered:
            return True # Size decreased by 2
        
        # Try to repair with 1 codeword
        best_c1 = -1
        for c_add in range(N):
            if c_add in code: continue
            nb_set = set(neighbors[c_add])
            if all(u in nb_set for u in uncovered):
                best_c1 = c_add
                break
        
        if best_c1 != -1:
            code.append(best_c1)
            coverage[neighbors[best_c1]] += 1
            # Size decreased by 1 (removed 2, added 1)
            return True
            
        # Try to repair with 2 codewords
        # Greedy: pick best first, then best second
        # This is expensive. Let's do a limited search.
        # If we can't find a 1-swap, maybe we can find a 2-swap that reduces size?
        # Removed 2, need to add 2. Net change 0.
        # Only accept if we can find 2 codewords that cover all uncovered.
        # And ideally, this new pair is "better" (hard to define).
        # For now, just try to restore validity with 2 codewords.
        # Greedy repair for 2
        remaining = set(uncovered)
        added = []
        for _ in range(2):
            if not remaining: break
            best_c = -1
            best_gain = 0
            for c_add in range(N):
                if c_add in code or c_add in added: continue
                nb_set = set(neighbors[c_add])
                gain = len(remaining & nb_set)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c_add
            if best_c == -1: break
            added.append(best_c)
            remaining -= set(neighbors[best_c])
            
        if not remaining:
            for c in added:
                code.append(c)
                coverage[neighbors[c]] += 1
            # Size same. Accept to explore.
            return True
        else:
            # Rollback
            code.append(c1)
            code.append(c2)
            coverage[neighbors[c1]] += 1
            coverage[neighbors[c2]] += 1
            return False

    def local_search(code, coverage, time_limit):
        """Main local search loop."""
        start = time.time()
        no_improve_count = 0
        best_size = len(code)
        
        while time.time() - start < time_limit:
            improved = False
            
            # 1. Prune
            old_len = len(code)
            code = prune(code, coverage)
            if len(code) < old_len:
                improved = True
                best_size = len(code)
                no_improve_count = 0
                
            # 2. Try moves
            move_type = random.random()
            
            if move_type < 0.5:
                if try_1_kick(code, coverage, deadline):
                    # Check if size improved
                    if len(code) < best_size:
                        best_size = len(code)
                        no_improve_count = 0
                        improved = True
                    else:
                        no_improve_count += 1
            elif move_type < 0.8:
                if try_2_kick(code, coverage, deadline):
                    if len(code) < best_size:
                        best_size = len(code)
                        no_improve_count = 0
                        improved = True
                    else:
                        no_improve_count += 1
            else:
                # Exchange move: Remove one, add one (different from kick which allows same size)
                # Actually 1-kick already handles remove+add.
                # Let's do a specific "swap" where we force a change.
                if code:
                    idx = random.randint(0, len(code) - 1)
                    c_remove = code[idx]
                    code.pop(idx)
                    coverage[neighbors[c_remove]] -= 1
                    
                    # Find best replacement
                    best_c_add = -1
                    best_gain = -1
                    # We want to cover the holes left by c_remove, but also maximize coverage of others?
                    # Holes are words with coverage 0.
                    holes = [nb for nb in neighbors[c_remove] if coverage[nb] == 0]
                    
                    if not holes:
                        # We can just remove c_remove?
                        # That's a 1-kick success.
                        pass
                    else:
                        for c_add in range(N):
                            if c_add in code: continue
                            nb_set = set(neighbors[c_add])
                            if all(h in nb_set for h in holes):
                                # Valid replacement
                                # Score: how many total words does it cover? (All 15)
                                # Or how many were already covered? (Waste)
                                # Let's just pick the first valid one.
                                best_c_add = c_add
                                break
                        
                        if best_c_add != -1:
                            code.append(best_c_add)
                            coverage[neighbors[best_c_add]] += 1
                            # Size same.
                            no_improve_count += 1
                        else:
                            # Rollback
                            code.append(c_remove)
                            coverage[neighbors[c_remove]] += 1
                            no_improve_count += 1
                    # If no holes, we effectively removed a codeword.
                    # Check if size improved
                    if len(code) < best_size:
                        best_size = len(code)
                        no_improve_count = 0
                        improved = True
                    else:
                        no_improve_count += 1

            if not improved and no_improve_count > 50:
                # Restart or Perturb
                # Perturb: remove 5 random codewords and repair
                for _ in range(5):
                    if code:
                        idx = random.randint(0, len(code) - 1)
                        c = code.pop(idx)
                        coverage[neighbors[c]] -= 1
                # Repair
                while True:
                    uncovered = np.where(coverage == 0)[0]
                    if len(uncovered) == 0: break
                    # Greedy add
                    best_c = -1
                    best_gain = 0
                    for c in range(N):
                        if c in code: continue
                        cnt = np.sum(coverage[neighbors[c]] == 0) # Count uncovered covered by c
                        if cnt > best_gain:
                            best_gain = cnt
                            best_c = c
                    if best_c == -1: break
                    code.append(best_c)
                    coverage[neighbors[best_c]] += 1
                no_improve_count = 0

        return code, coverage

    # Main Execution
    # Multi-start
    best_code = []
    best_coverage = None
    best_len = float('inf')
    
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        if r == 0:
            code = greedy_initial()
        else:
            # Perturb best known
            if not best_code:
                code = greedy_initial()
            else:
                code = best_code[:]
                # Remove some random
                for _ in range(random.randint(2, 5)):
                    if code:
                        idx = random.randint(0, len(code) - 1)
                        code.pop(idx)
                # Rebuild coverage
                coverage = build_coverage(code)
                # Repair
                while True:
                    uncovered = np.where(coverage == 0)[0]
                    if len(uncovered) == 0: break
                    best_c = -1
                    best_gain = 0
                    for c in range(N):
                        if c in code: continue
                        cnt = np.sum(coverage[neighbors[c]] == 0)
                        if cnt > best_gain:
                            best_gain = cnt
                            best_c = c
                    if best_c == -1: break
                    code.append(best_c)
                    coverage[neighbors[best_c]] += 1
        coverage = build_coverage(code)
        code = prune(code, coverage)
        
        # Local Search
        code, coverage = local_search(code, coverage, time_per_restart)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
            best_coverage = coverage.copy()
            
    # Final Prune
    if best_code:
        best_code = prune(best_code, best_coverage)
        
    return [idx_to_tuple(c) for c in best_code]
