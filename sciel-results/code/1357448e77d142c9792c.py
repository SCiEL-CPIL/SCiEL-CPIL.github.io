import numpy as np
import time

def construct(q, n, radius):
    deadline = time.time() + 115.0  # 150s wall, safety margin

    if q == 3 and n == 8 and radius == 1:
        N = 3 ** 8  # 6561
        powers = [3 ** i for i in range(n)]

        # Precompute neighbors for all words (17 neighbors: self + 16 at distance 1)
        neighbors = np.empty((N, 17), dtype=np.int32)
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

        # Build reverse neighbor map: for each word, which codeword positions cover it
        # Actually, let's precompute: for each word w, the list of words at distance <= 1
        # That's just neighbors[w]. We need the reverse: for each codeword c, which words does it cover.
        # That's neighbors[c]. Good.

        def greedy_construct(rng, alpha=1.0):
            """Greedy set cover with randomized tie-breaking."""
            uncovered = np.ones(N, dtype=np.bool_)
            code = []
            while np.any(uncovered) and time.time() < deadline - 5:
                gains = uncovered[neighbors].sum(axis=1).astype(np.float64)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains >= max_gain * alpha)[0]
                if alpha < 1.0 and len(candidates) > 1:
                    w = rng.choice(candidates, p=gains[candidates] / gains[candidates].sum())
                else:
                    w = rng.choice(candidates)
                code.append(w)
                uncovered[neighbors[w, :]] = False
            return code

        def reduce_code(code_list):
            """Remove redundant codewords via iterative coverage counting."""
            code_set = set(code_list)
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            changed = True
            while changed and time.time() < deadline - 3:
                changed = False
                for c in list(code_set):
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        code_set.remove(c)
                        coverage[neighbors[c, :]] -= 1
                        changed = True
            return code_set

        def try_swaps(code_set, max_iter=50):
            """Try 1-for-1 swaps to reduce code size."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            rng = np.random.default_rng()
            code_list = list(code_set)

            for _ in range(max_iter):
                if time.time() > deadline - 3:
                    break
                idx = rng.integers(0, len(code_list))
                c = code_list[idx]
                c_neigh = neighbors[c, :]

                essential = c_neigh[coverage[c_neigh] == 1]

                if len(essential) == 0:
                    code_list.pop(idx)
                    coverage[neighbors[c, :]] -= 1
                    continue

                if len(essential) == 1:
                    candidates = neighbors[essential[0], :]
                else:
                    cand_set = set(neighbors[essential[0], :].tolist())
                    for ew in essential[1:]:
                        cand_set &= set(neighbors[ew, :].tolist())
                    if not cand_set:
                        continue
                    candidates = np.array(list(cand_set), dtype=np.int32)

                for d in candidates:
                    if d in code_set:
                        continue
                    bad = False
                    for w in c_neigh:
                        if coverage[w] == 1 and w not in neighbors[d, :]:
                            bad = True
                            break
                    if bad:
                        continue
                    code_set.remove(c)
                    code_set.add(d)
                    coverage[neighbors[c, :]] -= 1
                    coverage[neighbors[d, :]] += 1
                    code_list = list(code_set)
                    break

            return code_set

        def add_reduce_search(code_set, max_iter=100):
            """Add-Reduce local search: temporarily add a codeword, then reduce."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            rng = np.random.default_rng()
            code_list = list(code_set)

            for _ in range(max_iter):
                if time.time() > deadline - 3:
                    break

                # Try to find a codeword to remove
                removable = [c for c in code_list if np.min(coverage[neighbors[c, :]]) > 1]
                if not removable:
                    # No removable codeword; try adding one to enable removal
                    # Find a word that is covered by many codewords (redundant coverage)
                    # or find a gap
                    if np.any(coverage == 0):
                        # There's an uncovered word (shouldn't happen if valid, but just in case)
                        uncovered_words = np.where(coverage == 0)[0]
                        uw = rng.choice(uncovered_words)
                        # Find best codeword to cover it
                        gains = coverage[neighbors[uw, :]]
                        # Actually we need to find a codeword that covers uw
                        # uw is at distance 0 or 1 from its neighbors
                        candidates = neighbors[uw, :]
                        # Pick the one that covers the most uncovered words
                        # But coverage is 0 for uw, so we need to add a codeword
                        # Pick randomly from candidates
                        new_c = rng.choice(candidates)
                        if new_c not in code_set:
                            code_set.add(new_c)
                            coverage[neighbors[new_c, :]] += 1
                            code_list.append(new_c)
                    else:
                        # All covered, try to add a random codeword to create redundancy
                        new_c = rng.integers(0, N)
                        if new_c not in code_set:
                            code_set.add(new_c)
                            coverage[neighbors[new_c, :]] += 1
                            code_list.append(new_c)
                    continue

                # Remove a random removable codeword
                c = rng.choice(removable)
                code_set.remove(c)
                coverage[neighbors[c, :]] -= 1
                code_list.remove(c)

            return code_set

        # Run multiple greedy constructions with different parameters
        best_code = None
        best_size = float('inf')

        configs = [
            (42, 1.0), (123, 1.0), (456, 0.95), (789, 0.9),
            (101, 1.0), (202, 0.85), (303, 0.95), (404, 1.0),
            (505, 0.9), (606, 0.8), (707, 1.0), (808, 0.95),
            (909, 0.85), (111, 1.0), (222, 0.9), (333, 0.8),
        ]

        for seed, alpha in configs:
            if time.time() > deadline - 15:
                break

            rng = np.random.default_rng(seed)
            code_list = greedy_construct(rng, alpha)
            code_set = set(code_list)

            # Reduction phase
            code_set = reduce_code(list(code_set))

            # Swap phase
            code_set = try_swaps(code_set, max_iter=40)

            # Add-reduce phase
            code_set = add_reduce_search(code_set, max_iter=50)

            # Final reduction
            code_set = reduce_code(list(code_set))

            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = code_set

            if best_size <= 486:
                break

        # Final reduction on best code
        if best_code is not None:
            best_code = reduce_code(list(best_code))
            best_code = try_swaps(best_code, max_iter=30)
            best_code = add_reduce_search(best_code, max_iter=30)
            best_code = reduce_code(list(best_code))

        if best_code is None:
            rng = np.random.default_rng(0)
            code_list = greedy_construct(rng, 1.0)
            best_code = set(reduce_code(code_list))

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
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        powers = [q ** i for i in range(n)]
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
