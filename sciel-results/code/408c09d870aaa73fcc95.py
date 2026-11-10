import numpy as np
import time

def construct(q, n, radius):
    deadline = time.time() + 115.0  # safety margin before 150s kill

    if q == 3 and n == 8 and radius == 1:
        N = 3 ** 8  # 6561
        powers = [3 ** i for i in range(n)]

        # Precompute neighbors for all words (1 + 2*n = 17 neighbors)
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                diff1 = ((d + 1) % q) - d
                neighbors[w, idx] = w + diff1 * powers[i]
                idx += 1
                diff2 = ((d + 2) % q) - d
                neighbors[w, idx] = w + diff2 * powers[i]
                idx += 1

        def greedy_construct(rng, alpha=1.0, beta=1.0):
            """Greedy set cover with randomized tie-breaking.
            alpha: threshold fraction of max gain for candidate pool
            beta: temperature for soft selection (1.0 = uniform, 0 = greedy)
            """
            uncovered = np.ones(N, dtype=np.bool_)
            code = []
            while np.any(uncovered) and time.time() < deadline - 8:
                gains = uncovered[neighbors].sum(axis=1).astype(np.float64)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                threshold = max_gain * alpha
                candidates = np.where(gains >= threshold)[0]
                
                if len(candidates) > 1 and beta < 1.0:
                    weights = gains[candidates] ** beta
                    w = rng.choice(candidates, p=weights / weights.sum())
                else:
                    w = rng.choice(candidates)
                
                code.append(w)
                uncovered[neighbors[w, :]] = False
            return code

        def reduce_code(code_set):
            """Remove redundant codewords via iterative coverage counting."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                code_list = list(code_set)
                for c in code_list:
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        code_set.remove(c)
                        coverage[neighbors[c, :]] -= 1
                        changed = True
            return code_set

        def compute_essentiality(coverage, code_list):
            """Compute how many words each codeword exclusively covers."""
            if not code_list:
                return np.array([], dtype=np.float64)
            code_arr = np.array(code_list, dtype=np.int32)
            ess = np.zeros(len(code_arr), dtype=np.float64)
            for i, c in enumerate(code_arr):
                ess[i] = np.sum(coverage[neighbors[c, :]] == 1)
            return ess

        def add_remove_search(code_set, max_iter=200):
            """
            Add-Remove local search targeting singly-covered words.
            Uses essentiality to guide which codewords to remove.
            """
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            rng = np.random.default_rng()
            code_list = list(code_set)
            
            for _ in range(max_iter):
                if time.time() > deadline - 8:
                    break

                # Find singly-covered words (bottlenecks)
                single_mask = (coverage == 1)
                if not np.any(single_mask):
                    break

                single_covered = np.where(single_mask)[0]
                
                # Compute gain for all potential new codewords
                gain = np.zeros(N, dtype=np.int32)
                for s in single_covered:
                    gain[neighbors[s, :]] += 1

                # Filter out words already in code
                code_arr = np.array(code_list, dtype=np.int32)
                gain[code_arr] = 0

                max_gain = np.max(gain)
                if max_gain == 0:
                    break

                # Select among top candidates
                threshold = max_gain * 0.8
                candidates = np.where(gain >= threshold)[0]
                if len(candidates) == 0:
                    candidates = np.where(gain == max_gain)[0]
                
                new_c = rng.choice(candidates)

                # Add new_c
                code_set.add(new_c)
                coverage[neighbors[new_c, :]] += 1
                code_list.append(new_c)
                
                # Try to remove redundant codewords, prioritizing low essentiality
                code_arr = np.array(code_list, dtype=np.int32)
                mins = np.min(coverage[neighbors[code_arr]], axis=1)
                redundant_mask = mins > 1
                redundant_indices = np.where(redundant_mask)[0]
                
                if len(redundant_indices) > 0:
                    # Among redundant, pick one with lowest essentiality
                    ess = compute_essentiality(coverage, code_list)
                    redundant_ess = ess[redundant_indices]
                    best_idx = redundant_indices[np.argmin(redundant_ess)]
                    c_to_remove = code_arr[best_idx]
                    
                    code_set.remove(c_to_remove)
                    coverage[neighbors[c_to_remove, :]] -= 1
                    code_list.remove(c_to_remove)
                else:
                    # Nothing removable. Revert the addition.
                    code_set.remove(new_c)
                    coverage[neighbors[new_c, :]] -= 1
                    code_list.remove(new_c)
                    
            return code_set

        def try_swaps(code_set, max_iter=50):
            """Try 1-for-1 swaps to reduce code size, using essentiality weighting."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            rng = np.random.default_rng()
            code_list = list(code_set)

            for _ in range(max_iter):
                if time.time() > deadline - 8:
                    break
                
                code_list = list(code_set)
                if not code_list:
                    break
                
                # Compute essentiality for all codewords
                ess = compute_essentiality(coverage, code_list)
                
                # Pick a codeword to try swapping, weighted toward lower essentiality
                weights = 1.0 / (ess + 1.0)
                weights /= weights.sum()
                idx = rng.choice(len(code_list), p=weights)
                c = code_list[idx]
                c_neigh = neighbors[c, :]

                # Words only covered by c
                essential = c_neigh[coverage[c_neigh] == 1]

                if len(essential) == 0:
                    # c is redundant
                    code_set.remove(c)
                    coverage[neighbors[c, :]] -= 1
                    continue

                # Try to find a single replacement d that covers all essential words
                if len(essential) == 1:
                    candidates = neighbors[essential[0], :]
                else:
                    cand_set = set(neighbors[essential[0], :].tolist())
                    for ew in essential[1:]:
                        cand_set &= set(neighbors[ew, :].tolist())
                    if not cand_set:
                        continue
                    candidates = np.array(list(cand_set), dtype=np.int32)

                # Filter: d not already in code
                found = False
                for d in candidates:
                    if d in code_set:
                        continue
                    d_neigh_set = set(neighbors[d, :].tolist())
                    bad = False
                    for w in essential:
                        if w not in d_neigh_set:
                            bad = True
                            break
                    if bad:
                        continue
                    # Accept the swap
                    code_set.remove(c)
                    code_set.add(d)
                    coverage[neighbors[c, :]] -= 1
                    coverage[neighbors[d, :]] += 1
                    found = True
                    break

            return code_set

        def two_for_one_search(code_set, max_iter=30):
            """Try to replace two codewords with one (2-for-1 reduction)."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            rng = np.random.default_rng()
            code_list = list(code_set)

            for _ in range(max_iter):
                if time.time() > deadline - 8:
                    break
                
                code_list = list(code_set)
                if len(code_list) < 2:
                    break

                # Pick two codewords at random
                idx1, idx2 = rng.choice(len(code_list), size=2, replace=False)
                c1, c2 = code_list[idx1], code_list[idx2]
                
                n1 = set(neighbors[c1, :].tolist())
                n2 = set(neighbors[c2, :].tolist())
                
                # All words that would be uncovered if we remove both c1 and c2
                # and need to be covered by a single replacement
                all_ess = set()
                for w in n1:
                    if coverage[w] <= 2 and w not in n2:
                        all_ess.add(w)
                for w in n2:
                    if coverage[w] <= 2 and w not in n1:
                        all_ess.add(w)
                
                if len(all_ess) == 0:
                    # Both are redundant
                    code_set.remove(c1)
                    code_set.remove(c2)
                    coverage[neighbors[c1, :]] -= 1
                    coverage[neighbors[c2, :]] -= 1
                    continue

                # Find a single word d that covers all_ess
                if len(all_ess) == 1:
                    candidates = neighbors[list(all_ess)[0], :]
                elif len(all_ess) <= 4:
                    cand_set = set(neighbors[list(all_ess)[0], :].tolist())
                    for ew in list(all_ess)[1:]:
                        cand_set &= set(neighbors[ew, :].tolist())
                    if not cand_set:
                        continue
                    candidates = np.array(list(cand_set), dtype=np.int32)
                else:
                    continue

                # Filter: d not already in code
                found = False
                for d in candidates:
                    if d in code_set:
                        continue
                    d_neigh_set = set(neighbors[d, :].tolist())
                    if all_ess.issubset(d_neigh_set):
                        # Check that removing c1 and c2 and adding d doesn't leave gaps
                        lost = (n1 | n2) - d_neigh_set
                        ok = True
                        for w in lost:
                            if coverage[w] <= 2:
                                ok = False
                                break
                        if ok:
                            code_set.remove(c1)
                            code_set.remove(c2)
                            code_set.add(d)
                            coverage[neighbors[c1, :]] -= 1
                            coverage[neighbors[c2, :]] -= 1
                            coverage[neighbors[d, :]] += 1
                            found = True
                            break

            return code_set

        # Run multiple greedy constructions with different parameters
        best_code = None
        best_size = float('inf')

        configs = [
            (42, 1.0, 1.0), (123, 1.0, 1.0), (456, 0.95, 0.8), (789, 0.9, 0.7),
            (101, 1.0, 1.0), (202, 0.85, 0.6), (303, 0.95, 0.8), (404, 1.0, 1.0),
            (505, 0.9, 0.7), (606, 0.8, 0.5), (707, 1.0, 1.0), (808, 0.95, 0.8),
            (909, 0.85, 0.6), (111, 1.0, 1.0), (222, 0.9, 0.7), (333, 0.8, 0.5),
            (444, 1.0, 1.0), (555, 0.9, 0.7), (666, 1.0, 1.0), (777, 0.85, 0.6),
            (888, 0.9, 0.7), (999, 1.0, 1.0), (1111, 0.95, 0.8), (2222, 0.85, 0.6),
        ]

        for seed, alpha, beta in configs:
            if time.time() > deadline - 15:
                break

            rng = np.random.default_rng(seed)
            code_list = greedy_construct(rng, alpha, beta)
            code_set = set(code_list)

            # Reduction phase
            code_set = reduce_code(code_set)
            
            # Swap phase
            code_set = try_swaps(code_set, max_iter=25)
            
            # Add-Remove phase
            code_set = add_remove_search(code_set, max_iter=120)
            
            # 2-for-1 search
            code_set = two_for_one_search(code_set, max_iter=20)
            
            # Second swap phase after add-remove
            code_set = try_swaps(code_set, max_iter=25)
            
            # Final reduction
            code_set = reduce_code(code_set)

            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = code_set

            if best_size <= 486:
                break

        if best_code is None:
            # Fallback
            rng = np.random.default_rng(0)
            code_list = greedy_construct(rng, 1.0, 1.0)
            best_code = set(reduce_code(code_list))

        # Final polish on the best code found
        if best_code:
            best_code = try_swaps(best_code, max_iter=30)
            best_code = add_remove_search(best_code, max_iter=80)
            best_code = two_for_one_search(best_code, max_iter=15)
            best_code = try_swaps(best_code, max_iter=30)
            best_code = reduce_code(best_code)

        # Convert to list of digit lists
        final_code = []
        for w in best_code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)

        return final_code

    else:
        # Generic fallback for other (q, n, radius)
        N = q ** n
        powers = [q ** i for i in range(n)]
        neigh_count = 1 + n * (q - 1)
        neighbors = np.empty((N, neigh_count), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    diff = new_d - d
                    neighbors[w, idx] = w + diff * powers[i]
                    idx += 1

        rng = np.random.default_rng()
        uncovered = np.ones(N, dtype=np.bool_)
        code = set()
        while np.any(uncovered) and time.time() < deadline:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.add(w)
            uncovered[neighbors[w, :]] = False

        final_code = []
        for w in code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        return final_code
