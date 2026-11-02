import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # Constants for the specific task instance
    # q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other instances, though task implies specific
        # Just return a trivial valid code (all words) if not the target case
        # This is inefficient but valid
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    total = q ** n  # 729
    
    # Precompute neighbors for all 729 words
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return tuple(t)
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute balls as numpy arrays for speed
    neighbors = [None] * total
    neighbors_set = [None] * total  # For O(1) membership checks
    neighbors_list = [None] * total
    
    for idx in range(total):
        w = idx_to_tuple(idx)
        nbs = [idx]
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (new_w[i] + d) % q
                nbs.append(tuple_to_idx(tuple(new_w)))
        neighbors_list[idx] = nbs
        neighbors[idx] = np.array(nbs, dtype=np.int32)
        neighbors_set[idx] = set(nbs)

    # Greedy set-cover initialization
    def greedy_cover():
        uncovered = set(range(total))
        code = []
        # Precompute: for each candidate word, which uncovered points it covers
        while uncovered:
            best_cand = -1
            best_count = -1
            # Sample a subset of candidates for speed
            candidates = list(range(total))
            random.shuffle(candidates)
            for cand in candidates:
                if not uncovered:
                    break
                covered_here = len(neighbors_set[cand] & uncovered)
                if covered_here > best_count:
                    best_count = covered_here
                    best_cand = cand
                    if best_count == len(uncovered):
                        break
            if best_cand < 0:
                break
            code.append(best_cand)
            uncovered -= neighbors_set[best_cand]
        return code

    # Tetracode direct sum
    def tetracode_init():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                result.append(tuple_to_idx(tuple(h + list(tail))))
        return result

    def random_init(num_words):
        return random.sample(range(total), num_words)

    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill

    best_code = None
    best_size = float('inf')

    # Try multiple initialization strategies and local searches
    initializations = [
        tetracode_init(),
        greedy_cover(),
    ]
    
    # Add a few random inits of varying sizes
    for sz in [75, 80, 85, 90]:
        initializations.append(random_init(sz))

    random.shuffle(initializations)

    for init_code in initializations:
        if time.time() > deadline:
            break
        
        # Ensure validity: if init doesn't cover, add words until it does
        cover_count = np.zeros(total, dtype=np.int32)
        for c in init_code:
            cover_count[neighbors[c]] += 1
        
        # Check coverage
        if np.any(cover_count == 0):
            # Add words to cover remaining
            uncovered = set(np.where(cover_count == 0)[0].tolist())
            while uncovered:
                best_cand = -1
                best_count = -1
                for cand in list(uncovered):
                    covered_here = len(neighbors_set[cand] & uncovered)
                    if covered_here > best_count:
                        best_count = covered_here
                        best_cand = cand
                if best_cand < 0:
                    break
                init_code.append(best_cand)
                cover_count[neighbors[best_cand]] += 1
                uncovered -= neighbors_set[best_cand]

        # Remove duplicates
        init_code = list(dict.fromkeys(init_code))
        
        if len(init_code) < best_size:
            best_code = list(init_code)
            best_size = len(init_code)

        # Local search on this initialization
        code = list(init_code)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code:
            cover_count[neighbors[c]] += 1

        def try_remove_idx(c_idx):
            """Try removing codeword at index c_idx. Returns (success, uncovered_list)."""
            nb_arr = neighbors[c_idx]
            old_counts = cover_count[nb_arr].copy()
            cover_count[nb_arr] -= 1
            uncovered = []
            for nb in nb_arr:
                if cover_count[nb] == 0:
                    uncovered.append(nb)
            if uncovered:
                cover_count[nb_arr] = old_counts  # Restore
                return False, uncovered
            return True, []

        def add_idx(c_idx):
            cover_count[neighbors[c_idx]] += 1

        def find_swap_target(uncovered_indices):
            """Find a word that covers all uncovered indices."""
            if not uncovered_indices:
                return None
            u0 = uncovered_indices[0]
            candidates = list(neighbors_list[u0])
            random.shuffle(candidates)
            for c_idx in candidates:
                if all(u in neighbors_set[c_idx] for u in uncovered_indices):
                    return c_idx
            return None

        def find_double_swap(uncovered1, uncovered2):
            """Find a word that covers all of uncovered1 + uncovered2."""
            all_uncov = list(set(uncovered1) | set(uncovered2))
            if not all_uncov:
                return None
            u0 = all_uncov[0]
            candidates = list(neighbors_list[u0])
            random.shuffle(candidates)
            for c_idx in candidates:
                if all(u in neighbors_set[c_idx] for u in all_uncov):
                    return c_idx
            return None

        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(code)
            
            # Phase 1: Try removals
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                success, uncovered = try_remove_idx(code[i])
                if success:
                    code.pop(i)
                    improved = True
                    break
            
            if time.time() > deadline:
                break
            if improved:
                continue
            
            # Phase 2: Try single swaps
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                c = code[i]
                success, uncovered = try_remove_idx(c)
                if success:
                    code.pop(i)
                    improved = True
                    break
                if uncovered:
                    new_c = find_swap_target(uncovered)
                    if new_c is not None and new_c not in code:
                        add_idx(new_c)
                        code[i] = new_c
                        improved = True
                        break
                    elif new_c is not None:
                        # new_c already in code, skip
                        pass
                # If not successful, cover_count is already restored by try_remove_idx

            if time.time() > deadline:
                break
            if improved:
                continue

            # Phase 3: Try double swaps (remove 2, add 1)
            if len(code) >= 2:
                found_double = False
                # Sample pairs to limit computation
                max_pairs = min(50, len(code) * (len(code) - 1) // 2)
                pair_indices = list(itertools.combinations(range(len(code)), 2))
                random.shuffle(pair_indices)
                pair_indices = pair_indices[:max_pairs]
                
                for (i, j) in pair_indices:
                    if time.time() > deadline:
                        break
                    if i >= len(code) or j >= len(code):
                        continue
                    # Remove both
                    success1, unc1 = try_remove_idx(code[i])
                    if not success1:
                        continue
                    # code[i] is now removed from cover_count
                    # Remove code[j] (adjust index if i < j)
                    j_adj = j - 1 if i < j else j
                    if j_adj < 0 or j_adj >= len(code):
                        # Restore
                        add_idx(code[i])
                        break
                    success2, unc2 = try_remove_idx(code[j_adj])
                    if not success2:
                        add_idx(code[i])
                        break
                    # Both removed, find replacement
                    all_uncov = list(set(unc1) | set(unc2))
                    new_c = find_double_swap(unc1, unc2)
                    if new_c is not None and new_c not in code:
                        add_idx(new_c)
                        code[i] = new_c
                        code.pop(j_adj)
                        found_double = True
                        improved = True
                        break
                    else:
                        # Restore both
                        add_idx(code[j_adj])
                        add_idx(code[i])
                
                if found_double:
                    continue

        # Update best
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if best_size <= 73:  # Can't do better than known best
            break

    # Verify best_code is valid
    if best_code is None:
        best_code = tetracode_init()

    # Final verification and cleanup
    cover_count = np.zeros(total, dtype=np.int32)
    for c in best_code:
        cover_count[neighbors[c]] += 1
    if np.any(cover_count == 0):
        # Fallback: use full space
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    return [idx_to_tuple(c) for c in best_code]
