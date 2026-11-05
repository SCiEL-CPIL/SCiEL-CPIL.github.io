import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if not (q == 3 and n == 8 and radius == 1):
        # Fallback
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0
    
    total_points = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute neighbors
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        neighbors[idx, 0] = idx
        nb = 1
        for pos in range(n):
            orig = word[pos]
            pw = 1
            for j in range(pos):
                pw *= q
            # +1
            nd = (orig + 1) % q
            delta = -2 * pw if nd == 0 else pw
            neighbors[idx, nb] = idx + delta
            nb += 1
            # -1
            nd = (orig - 1) % q
            delta = 2 * pw if nd == 2 else -pw
            neighbors[idx, nb] = idx + delta
            nb += 1

    # Precompute digit powers for fast index conversion
    digit_powers = np.array([q**i for i in range(n)], dtype=np.int64)
    
    def idx_to_word(idx):
        temp = idx
        return [int(temp // p % q) for p in digit_powers]
    
    def idxs_to_words(idxs):
        idxs = np.asarray(idxs, dtype=np.int64)
        if len(idxs) == 0:
            return []
        words = np.zeros((len(idxs), n), dtype=np.int64)
        for i in range(n):
            words[:, i] = (idxs // digit_powers[i]) % q
        return [list(row) for row in words]

    def get_counts(code):
        counts = np.zeros(total_points, dtype=np.int32)
        for idx in code:
            counts[neighbors[idx]] += 1
        return counts

    def reduce_code(code, counts):
        current_code = list(code)
        current_counts = counts.copy()
        changed = True
        while changed:
            changed = False
            next_code = []
            for c in current_code:
                if np.all(current_counts[neighbors[c]] > 1):
                    current_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    next_code.append(c)
            current_code = next_code
        return current_code, current_counts

    def fast_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        potential = np.full(total_points, ball_size, dtype=np.int32)
        selected = []
        while True:
            if time.time() > deadline - 40.0:
                break
            maxp = potential.max()
            if maxp <= 0:
                break
            cands = np.where(potential == maxp)[0]
            c = int(cands[rng.integers(len(cands))])
            selected.append(c)
            potential[c] = 0
            for x in neighbors[c]:
                if not is_covered[x]:
                    is_covered[x] = True
                    potential[neighbors[x]] -= 1
                    np.maximum(potential, 0, out=potential)
            if is_covered.all():
                break
        return selected

    def local_search(code, counts, deadline):
        code = list(code)
        counts = counts.copy()
        code_set = set(code)
        best_len = len(code)
        best_code = code[:]
        rng = np.random.default_rng(42)
        no_improve = 0
        max_no_improve = 5000
        
        # Precompute neighbors for all points as a numpy array for vectorized operations
        # neighbors is already (total_points, ball_size)
        
        while time.time() < deadline and no_improve < max_no_improve:
            improved = False
            
            # Pass 1: Vectorized removals
            if code:
                code_arr = np.array(code, dtype=np.int64)
                # Get neighbors for all codewords: shape (len(code), ball_size)
                code_neighbors = neighbors[code_arr]  # (len(code), ball_size)
                # Get counts for all these neighbor positions
                code_counts = counts[code_neighbors]  # (len(code), ball_size)
                # A codeword can be removed if all its neighbors have count > 1
                removable_mask = np.all(code_counts > 1, axis=1)  # (len(code),)
                removable_indices = np.where(removable_mask)[0]
                
                if len(removable_indices) > 0:
                    # Remove one (or a few) removable codewords
                    # Pick the one with the highest minimum count (most redundant)
                    min_counts = code_counts[removable_indices].min(axis=1)
                    best_idx = removable_indices[np.argmax(min_counts)]
                    c = code[best_idx]
                    counts[neighbors[c]] -= 1
                    code.pop(best_idx)
                    code_set.discard(c)
                    improved = True
                    if len(code) < best_len:
                        best_len = len(code)
                        best_code = code[:]

            # Pass 2: Moves - find a codeword that can be moved to cover its risky points
            if not improved and code:
                code_arr = np.array(code, dtype=np.int64)
                code_neighbors = neighbors[code_arr]
                code_counts = counts[code_neighbors]
                
                # Find codewords with risky points (count == 1)
                risky_mask = np.any(code_counts == 1, axis=1)
                risky_indices = np.where(risky_mask)[0]
                
                if len(risky_indices) > 0:
                    # Sample from risky codewords
                    sample_size = min(20, len(risky_indices))
                    sample = rng.choice(risky_indices, size=sample_size, replace=False)
                    
                    for s in sample:
                        c = code[s]
                        c_neighbors = neighbors[c]
                        risky_points = c_neighbors[counts[c_neighbors] == 1]
                        
                        if len(risky_points) == 0:
                            continue
                        
                        # Find candidates that cover all risky points
                        # A candidate x covers risky point r if r is in neighbors[x]
                        # i.e., x is in neighbors[r]
                        # So candidates = intersection of neighbors[r] for all r in risky_points
                        cand_mask = np.ones(total_points, dtype=bool)
                        for r in risky_points:
                            cand_mask[neighbors[r]] = True
                            # Actually, we want x such that r is in Ball(x), i.e., x is in Ball(r)
                            # So x must be in neighbors[r]
                            # Let me reconsider: we want x such that for all r in risky_points, r is in Ball(x)
                            # That means x is in Ball(r) for all r, so x is in intersection of Ball(r)
                            pass
                        
                        # Recompute: x is a valid candidate if for all r in risky_points, r in neighbors[x]
                        # Equivalent to: x in intersection of {x : r in neighbors[x]} for all r
                        # Which is: x in intersection of neighbors[r] (since neighbors is symmetric for radius 1)
                        # Wait, is neighbors symmetric? If r is in Ball(c), is c in Ball(r)?
                        # Yes, Hamming distance is symmetric.
                        # So candidates = intersection of neighbors[r] for r in risky_points
                        
                        cand_mask = np.ones(total_points, dtype=bool)
                        for r in risky_points:
                            cand_mask[~np.isin(cand_mask, neighbors[r])] = False
                            # Simpler: start with all True, then for each r, keep only those in neighbors[r]
                            # Let me use a different approach
                        
                        # Better approach:
                        cand_set = set(neighbors[risky_points[0]].tolist())
                        for r in risky_points[1:]:
                            cand_set &= set(neighbors[r].tolist())
                            if not cand_set:
                                break
                        
                        if not cand_set:
                            continue
                        
                        # Filter out existing codewords
                        cand_list = [x for x in cand_set if x not in code_set]
                        if not cand_list:
                            continue
                        
                        # Score candidates by how many additional points they cover with count > 1
                        # or just pick randomly
                        c_new = cand_list[rng.integers(len(cand_list))]
                        
                        # Validate: after removing c and adding c_new, all risky points of c are covered
                        # This is guaranteed by construction since c_new covers all risky points
                        # And other points in Ball(c) with count > 1 will still have count >= 1
                        
                        # Update
                        counts[neighbors[c]] -= 1
                        counts[neighbors[c_new]] += 1
                        code.pop(s)
                        code.append(c_new)
                        code_set.discard(c)
                        code_set.add(c_new)
                        
                        improved = True
                        if len(code) < best_len:
                            best_len = len(code)
                            best_code = code[:]
                        break

            if improved:
                no_improve = 0
            else:
                no_improve += 1
                # Periodic reduce
                if no_improve % 10 == 0:
                    new_code, new_counts = reduce_code(code, counts)
                    if len(new_code) < len(code):
                        code = new_code
                        counts = new_counts
                        code_set = set(code)
                        no_improve = 0
                        if len(code) < best_len:
                            best_len = len(code)
                            best_code = code[:]

        return best_code, get_counts(best_code)

    best_code = None
    best_len = float('inf')
    
    # Multiple Greedy + Local Search
    seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        sel = fast_greedy(seed)
        counts = get_counts(sel)
        sel, counts = reduce_code(sel, counts)
        
        # Local Search
        sel, counts = local_search(sel, counts, deadline)
        sel, counts = reduce_code(sel, counts)
        
        if len(sel) < best_len:
            best_len = len(sel)
            best_code = sel

    if best_code is None:
        sel = fast_greedy(1)
        best_code = sel

    result = idxs_to_words(best_code)
    return result
