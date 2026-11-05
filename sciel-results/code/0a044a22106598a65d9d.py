import time
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        total_words = q ** n

        def word_to_idx(w):
            idx = 0
            for val in w:
                idx = idx * q + val
            return idx

        def idx_to_word(idx):
            w = [0] * n
            temp = idx
            for i in range(n - 1, -1, -1):
                w[i] = temp % q
                temp //= q
            return list(w)

        neighbors = [None] * total_words
        for idx in range(total_words):
            w = idx_to_word(idx)
            nbrs = [idx]
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        nw = list(w)
                        nw[i] = val
                        nbrs.append(word_to_idx(tuple(nw)))
            neighbors[idx] = nbrs

        uncovered = [True] * total_words
        code = []
        while any(uncovered):
            best_idx = -1
            best_count = -1
            for idx in range(total_words):
                if idx in code:
                    continue
                count = sum(1 for nbr in neighbors[idx] if uncovered[nbr])
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1:
                break
            code.append(best_idx)
            for nbr in neighbors[best_idx]:
                uncovered[nbr] = False
        return [list(idx_to_word(idx)) for idx in code]

    # Specific optimization for q=3, n=8, radius=1
    total_words = 3 ** 8
    deadline = time.time() + 145.0

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx

    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return list(w)

    # Precompute neighbors
    neighbors = [None] * total_words
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(8):
            for val in range(3):
                if val != w[i]:
                    nw = list(w)
                    nw[i] = val
                    nbrs.append(word_to_idx(tuple(nw)))
        neighbors[idx] = nbrs

    # Build a bitmask of all neighbors for each word
    # For a word with value at position i, the neighbors are obtained by changing position i to the other two values
    # This is a structured graph.

    # Precompute neighbor lists as numpy arrays for vectorized operations
    neighbor_arrays = [np.array(nbrs, dtype=np.int32) for nbrs in neighbors]
    neighbor_sets = [set(nbrs) for nbrs in neighbors]

    # Phase 1: Greedy construction
    uncovered = np.ones(total_words, dtype=bool)
    code_list = []
    code_set = set()
    
    start_time = time.time()
    greedy_deadline = start_time + 30.0

    while np.any(uncovered) and time.time() < greedy_deadline:
        # Compute coverage of each candidate
        best_idx = -1
        best_gain = -1
        # Sample a subset of candidates to speed up
        remaining = [i for i in range(total_words) if i not in code_set]
        if len(remaining) > 1000:
            candidates = random.sample(remaining, 1000)
        else:
            candidates = remaining

        for i in candidates:
            nbrs = neighbor_arrays[i]
            gain = np.sum(uncovered[nbrs])
            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx == -1 or best_gain == 0:
            break

        code_set.add(best_idx)
        code_list.append(best_idx)
        for nbr in neighbors[best_idx]:
            uncovered[nbr] = False

    # Phase 2: Local search with kicks
    # Compute cover counts
    cover_count = np.zeros(total_words, dtype=int)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1

    def try_remove_redundant():
        removed = False
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                redundant = True
                for nbr in neighbors[c]:
                    if cover_count[nbr] <= 1:
                        redundant = False
                        break
                if redundant:
                    code_list.pop(i)
                    for nbr in neighbors[c]:
                        cover_count[nbr] -= 1
                    removed = True
                    changed = True
                    break
        return removed

    try_remove_redundant()

    # Phase 3: Iterated Local Search (ILS)
    # Strategy: kick (random swap) -> prune (remove redundant) -> accept if better
    # We use a smoother surrogate for the objective: instead of just minimizing size,
    # we also minimize the "tightness" of the cover, measured by the sum of (cover_count - 1)^2
    # This creates a smoother landscape for the search.

    def cover_tightness():
        cc = cover_count[code_list]  # This is wrong, we need cover_count over all words
        # Actually, tightness is sum over all words of (max(0, cover_count[w] - 1))
        # But that's not quite right. Let's use sum of (cover_count[w] - 1) for w with cover_count[w] >= 2
        # Actually, a good smooth surrogate is the sum of cover_count[w]^2
        # Since sum of cover_count[w] is fixed (sum of |N(c)| for c in code_list),
        # minimizing sum of squares is equivalent to minimizing variance, which promotes balanced coverage.
        cc = cover_count
        return np.sum(cc * cc)

    best_size = len(code_list)
    best_tightness = cover_tightness()
    best_code = list(code_list)
    
    temperature = 0.3
    min_temperature = 0.01
    cooling = 0.997

    search_deadline = deadline - 5.0

    while time.time() < search_deadline:
        # Kick: random swap
        if len(code_list) == 0:
            break

        # Try a random swap
        idx_c = random.randint(0, len(code_list) - 1)
        c = code_list[idx_c]

        # Temporarily remove c
        for nbr in neighbors[c]:
            cover_count[nbr] -= 1

        # Find a replacement
        uncovered_now = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]

        if not uncovered_now:
            # c was redundant, just remove it
            code_list.pop(idx_c)
        else:
            # Find candidates that cover all uncovered_now
            candidates = set(neighbors[uncovered_now[0]])
            for u in uncovered_now[1:]:
                candidates.intersection_update(neighbor_sets[u])
                if not candidates:
                    break

            candidates.difference_update(code_list)

            if candidates:
                cand_list = list(candidates)
                if len(cand_list) > 50:
                    cand_list = random.sample(cand_list, 50)

                # Score candidates by tightness improvement
                best_cp = -1
                best_score = float('inf')

                for cp in cand_list:
                    # Estimate the new tightness
                    # Current tightness: sum of cc^2
                    # New tightness: (cc + delta_cp)^2 where delta_cp is +1 for neighbors of cp
                    # We can compute this efficiently
                    nbrs_cp = neighbor_arrays[cp]
                    # The change in sum of squares is sum over nbrs of (cc[nbr]+1)^2 - cc[nbr]^2 = sum(2*cc[nbr]+1)
                    # But we need to account for the fact that some nbrs might already be covered
                    # Actually, since we're adding cp, all its neighbors get +1
                    delta = np.sum(2 * cover_count[nbrs_cp] + 1)
                    score = best_tightness + delta  # This is an approximation

                    if score < best_score:
                        best_score = score
                        best_cp = cp

                if best_cp != -1:
                    code_list[idx_c] = best_cp
                    for nbr in neighbors[best_cp]:
                        cover_count[nbr] += 1
                else:
                    # Revert
                    for nbr in neighbors[c]:
                        cover_count[nbr] += 1
            else:
                # No candidate, revert
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1

        # Prune redundant words
        try_remove_redundant()

        # Evaluate
        current_size = len(code_list)
        current_tightness = cover_tightness()

        if current_size < best_size or (current_size == best_size and current_tightness < best_tightness):
            best_size = current_size
            best_tightness = current_tightness
            best_code = list(code_list)
        else:
            # Accept with some probability if worse (simulated annealing)
            if current_size == best_size:
                delta_t = current_tightness - best_tightness
                if delta_t > 0:
                    prob = np.exp(-delta_t / (temperature * best_tightness))
                    if random.random() < prob:
                        best_size = current_size
                        best_tightness = current_tightness
                        best_code = list(code_list)

        # Cool down
        temperature *= cooling
        if temperature < min_temperature:
            temperature = min_temperature

    # Final polish
    code_list = best_code
    # Recompute cover_count
    cover_count = np.zeros(total_words, dtype=int)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1

    # Final removal pass
    for i in range(len(code_list) - 1, -1, -1):
        c = code_list[i]
        redundant = True
        for nbr in neighbors[c]:
            if cover_count[nbr] <= 1:
                redundant = False
                break
        if redundant:
            code_list.pop(i)

    return [idx_to_word(c) for c in code_list]
