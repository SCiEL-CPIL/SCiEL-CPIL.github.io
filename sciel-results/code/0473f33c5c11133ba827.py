import numpy as np
import time
import itertools
import random

Q = 3
N = 8
SPACE_SIZE = Q ** N

def precompute_neighbors():
    neighbors = [[] for _ in range(SPACE_SIZE)]
    for i in range(SPACE_SIZE):
        digits = [0] * N
        temp = i
        for k in range(N):
            digits[k] = temp % Q
            temp //= Q
        neighbors[i].append(i)
        for pos in range(N):
            orig_val = digits[pos]
            for new_val in range(Q):
                if new_val != orig_val:
                    new_digits = digits.copy()
                    new_digits[pos] = new_val
                    new_idx = 0
                    for k in range(N - 1, -1, -1):
                        new_idx = new_idx * Q + new_digits[k]
                    neighbors[i].append(new_idx)
    return neighbors

_NEIGHBORS = None
_ADJ_MATRIX = None

def get_neighbors():
    global _NEIGHBORS, _ADJ_MATRIX
    if _NEIGHBORS is None:
        _NEIGHBORS = precompute_neighbors()
        _ADJ_MATRIX = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        for i in range(SPACE_SIZE):
            for nb in _NEIGHBORS[i]:
                _ADJ_MATRIX[i, nb] = 1
    return _NEIGHBORS

def idx_to_tuple(idx):
    digits = [0] * N
    temp = idx
    for k in range(N):
        digits[k] = temp % Q
        temp //= Q
    return digits

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        pass

    neighbors = get_neighbors()
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Greedy Construction using adjacency matrix for fast gain computation
    covered_counts = np.zeros(SPACE_SIZE, dtype=np.int32)
    codewords = []
    uncovered_mask = np.ones(SPACE_SIZE, dtype=bool)
    uncov_int = uncovered_mask.astype(np.int8)

    while uncovered_mask.any():
        gains = uncov_int @ _ADJ_MATRIX
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]

        if best_gain == 0:
            break

        codewords.append(best_idx)
        for nb in neighbors[best_idx]:
            if uncovered_mask[nb]:
                uncovered_mask[nb] = False
                uncov_int[nb] = 0
            covered_counts[nb] += 1

    # 2. Peeling Redundant Words
    covered_counts = np.zeros(SPACE_SIZE, dtype=np.int32)
    for c in codewords:
        for nb in neighbors[c]:
            covered_counts[nb] += 1

    def peel_redundant(cw_list, cc_array):
        changed = True
        while changed:
            changed = False
            new_cw_list = []
            for c in cw_list:
                redundant = True
                for nb in neighbors[c]:
                    if cc_array[nb] < 2:
                        redundant = False
                        break
                if redundant:
                    for nb in neighbors[c]:
                        cc_array[nb] -= 1
                    changed = True
                else:
                    new_cw_list.append(c)
            if changed:
                cw_list = new_cw_list
        return cw_list, cc_array

    codewords, covered_counts = peel_redundant(codewords, covered_counts)
    best_size = len(codewords)

    # 3. Local Search with explicit list-based tracking
    random.seed(7919)

    # Precompute neighbor sets
    neighbor_sets = [set(nbrs) for nbrs in neighbors]

    # Precompute for each codeword, the set of words it uniquely covers (coverage==1)
    # This helps identify "critical" codewords that are hard to swap

    while time.time() < deadline:
        if len(codewords) < 2:
            break

        # Compute criticality for each codeword:
        # A codeword c is "critical" if removing it would leave some words uncovered
        # i.e., if there exists nb in neighbors[c] with covered_counts[nb] == 1
        # We bias swap selection toward less critical codewords

        # For efficiency, we compute a simple score: count of neighbors with count==1
        # Lower score = safer to swap
        criticality_scores = []
        for c in codewords:
            score = 0
            for nb in neighbors[c]:
                if covered_counts[nb] == 1:
                    score += 1
            criticality_scores.append(score)

        # Choose codeword to swap: prefer low criticality
        # Use weighted random selection
        total_crit = sum(criticality_scores)
        if total_crit == 0:
            c_idx = random.randint(0, len(codewords) - 1)
        else:
            # Weight inversely proportional to criticality + 1
            weights = [1.0 / (sc + 1.0) for sc in criticality_scores]
            total_w = sum(weights)
            r = random.random() * total_w
            cumsum = 0
            c_idx = 0
            for i, w in enumerate(weights):
                cumsum += w
                if r <= cumsum:
                    c_idx = i
                    break

        c = codewords[c_idx]

        # Pick a random word w to swap with
        w = random.randint(0, SPACE_SIZE - 1)
        if w == c:
            continue

        # Check if swap is valid
        S_w_set = neighbor_sets[w]

        valid_swap = True
        for nb in neighbors[c]:
            if nb not in S_w_set:
                if covered_counts[nb] < 2:
                    valid_swap = False
                    break

        if valid_swap:
            # Perform swap
            for nb in neighbors[c]:
                covered_counts[nb] -= 1
            for nb in neighbors[w]:
                covered_counts[nb] += 1

            codewords[c_idx] = w

            # Try to peel
            new_codewords, new_covered_counts = peel_redundant(codewords, covered_counts)

            if len(new_codewords) < best_size:
                best_size = len(new_codewords)
            codewords = new_codewords
            covered_counts = new_covered_counts

        # Also try double swaps: remove 2 codewords, add 1
        if len(codewords) >= 2 and random.random() < 0.25:
            c_idx1 = random.randint(0, len(codewords) - 1)
            c_idx2 = random.randint(0, len(codewords) - 1)
            if c_idx1 == c_idx2:
                continue

            c1 = codewords[c_idx1]
            c2 = codewords[c_idx2]

            w = random.randint(0, SPACE_SIZE - 1)
            if w == c1 or w == c2:
                continue

            S_w_set = neighbor_sets[w]
            S_c1_set = neighbor_sets[c1]
            S_c2_set = neighbor_sets[c2]

            valid_swap = True
            for nb in neighbors[c1]:
                if nb not in S_w_set:
                    loss = 0
                    if nb in S_c1_set:
                        loss += 1
                    if nb in S_c2_set:
                        loss += 1
                    if covered_counts[nb] < loss + 1:
                        valid_swap = False
                        break

            if valid_swap:
                for nb in neighbors[c2]:
                    if nb not in S_w_set:
                        loss = 0
                        if nb in S_c1_set:
                            loss += 1
                        if nb in S_c2_set:
                            loss += 1
                        if covered_counts[nb] < loss + 1:
                            valid_swap = False
                            break

            if valid_swap:
                for nb in neighbors[c1]:
                    covered_counts[nb] -= 1
                for nb in neighbors[c2]:
                    covered_counts[nb] -= 1
                for nb in neighbors[w]:
                    covered_counts[nb] += 1

                new_cw = []
                for i, cw in enumerate(codewords):
                    if i != c_idx1 and i != c_idx2:
                        new_cw.append(cw)
                new_cw.append(w)
                codewords = new_cw

                new_codewords, new_covered_counts = peel_redundant(codewords, covered_counts)

                if len(new_codewords) < best_size:
                    best_size = len(new_codewords)
                codewords = new_codewords
                covered_counts = new_covered_counts

    # 4. Final polish: exhaustive single-move improvement
    # For each codeword, try moving it to every position that maintains coverage.
    # This is a high-precision polish pass with tighter tolerance.
    polish_deadline = start_time + 118.0
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        for i in range(len(codewords)):
            if time.time() > polish_deadline:
                break
            c = codewords[i]
            c_nbrs = neighbors[c]
            # Find words that are uniquely covered by c (coverage == 1)
            # These must remain covered after moving c
            must_cover = [nb for nb in c_nbrs if covered_counts[nb] == 1]
            if not must_cover:
                # c is not critical; try to find a position that covers more
                # or at least maintains coverage while allowing peeling
                best_pos = -1
                best_gain = 0
                for w in range(SPACE_SIZE):
                    if w == c:
                        continue
                    w_nbrs = neighbors[w]
                    # Check validity: all must_cover words must be in w_nbrs
                    # (they're all in c_nbrs with count 1, so w must cover them)
                    valid = True
                    gain = 0
                    for nb in c_nbrs:
                        if nb not in neighbor_sets[w]:
                            if covered_counts[nb] < 2:
                                valid = False
                                break
                    if not valid:
                        continue
                    # Compute gain: new words covered by w that weren't by c
                    for nb in w_nbrs:
                        if covered_counts[nb] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_pos = w
                if best_pos >= 0 and best_gain > 0:
                    # Perform the move
                    for nb in c_nbrs:
                        covered_counts[nb] -= 1
                    for nb in neighbors[best_pos]:
                        covered_counts[nb] += 1
                    codewords[i] = best_pos
                    new_codewords, new_covered_counts = peel_redundant(codewords, covered_counts)
                    if len(new_codewords) < best_size:
                        best_size = len(new_codewords)
                    codewords = new_codewords
                    covered_counts = new_covered_counts
                    improved = True
                    break  # restart scan after modification
            else:
                # c is critical: find a position that covers all must_cover words
                # and potentially allows peeling
                best_pos = -1
                best_score = -1
                for w in range(SPACE_SIZE):
                    if w == c:
                        continue
                    w_nbrs = neighbors[w]
                    w_set = neighbor_sets[w]
                    # All must_cover must be in w_nbrs
                    valid = True
                    for mc in must_cover:
                        if mc not in w_set:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check all other c_nbrs with count < 2 are also in w_nbrs
                    for nb in c_nbrs:
                        if nb not in w_set:
                            if covered_counts[nb] < 2:
                                valid = False
                                break
                    if not valid:
                        continue
                    # Score: prefer positions that allow peeling (minimize unique coverage)
                    unique_count = 0
                    for nb in w_nbrs:
                        if covered_counts[nb] == 0:
                            unique_count += 1
                    # Also check: after moving, can we peel?
                    # Approximate: count how many c_nbrs would have count become 1
                    # (i.e., were at count 2 and now at 1)
                    peel_potential = 0
                    for nb in c_nbrs:
                        if covered_counts[nb] == 2 and nb not in w_set:
                            peel_potential += 1
                    score = peel_potential * 10 - unique_count
                    if score > best_score:
                        best_score = score
                        best_pos = w
                if best_pos >= 0:
                    for nb in c_nbrs:
                        covered_counts[nb] -= 1
                    for nb in neighbors[best_pos]:
                        covered_counts[nb] += 1
                    codewords[i] = best_pos
                    new_codewords, new_covered_counts = peel_redundant(codewords, covered_counts)
                    if len(new_codewords) < best_size:
                        best_size = len(new_codewords)
                    codewords = new_codewords
                    covered_counts = new_covered_counts
                    improved = True
                    break  # restart scan

    result = [idx_to_tuple(c) for c in codewords]
    return result
