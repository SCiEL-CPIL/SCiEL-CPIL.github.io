import time
import numpy as np

def construct(q, n, radius):
    t0 = time.time()
    DEADLINE = 110.0
    MARGIN = 10.0

    total = q ** n
    # Precompute all words as digit arrays and index map
    words = np.zeros((total, n), dtype=np.int8)
    for i in range(total):
        x = i
        for j in range(n - 1, -1, -1):
            words[i, j] = x % q
            x //= q
    # index map: word tuple -> index
    idx_map = {}
    for i in range(total):
        idx_map[tuple(words[i].tolist())] = i

    # Precompute radius-1 ball offsets: for each codeword index, list of indices within radius
    # radius=1 ball: the word itself + all words differing in exactly 1 position
    # For speed, precompute neighbor offsets per position
    # ball_offsets[cw_idx] = list of indices in ball(cw)
    # We'll compute on the fly using digit manipulation for speed

    def ball_indices(cw_idx):
        """Return list of indices within Hamming distance <= radius of words[cw_idx]."""
        cw = words[cw_idx]
        result = [cw_idx]
        if radius >= 1:
            for pos in range(n):
                for v in range(q):
                    if v == cw[pos]:
                        continue
                    # compute index of word with pos changed to v
                    new_word = cw.tolist()
                    new_word[pos] = v
                    result.append(idx_map[tuple(new_word)])
        return result

    # Precompute balls for all possible codewords
    balls = []
    for i in range(total):
        balls.append(ball_indices(i))

    # Coverage array
    coverage = np.zeros(total, dtype=np.int32)
    # Codeword indices selected
    selected = []
    # For tie-breaking: distance to nearest selected codeword
    # We'll compute on the fly

    # Greedy selection
    best_score = None
    best_selected = None

    while True:
        elapsed = time.time() - t0
        if elapsed > DEADLINE - MARGIN:
            break

        # Find best candidate: max coverage, tie-break by min distance to existing
        # For efficiency, find max coverage first, then among ties pick best distance
        max_cov = coverage.max()
        if max_cov <= 0:
            break
        # Find all indices with max coverage
        candidates = np.where(coverage == max_cov)[0]
        if len(candidates) == 0:
            break

        if len(selected) == 0:
            # Pick first candidate (arbitrary, all equivalent)
            best_cand = int(candidates[0])
        else:
            # Among candidates with max coverage, pick one with max min-distance to selected
            # Compute distances to all selected codewords for each candidate
            # For speed with small candidate sets, do direct computation
            best_cand = int(candidates[0])
            best_min_dist = -1
            sel_arr = np.array(selected, dtype=np.int64)
            for cand in candidates:
                # Compute Hamming distance to each selected codeword
                cand_word = words[cand]
                sel_words = words[sel_arr]
                # Hamming distance: number of positions that differ
                diffs = (sel_words != cand_word).sum(axis=1)
                min_dist = int(diffs.min())
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_cand = int(cand)
                    if best_min_dist >= n:  # max possible
                        break

        # Add best_cand to selected
        selected.append(best_cand)
        # Update coverage
        for bidx in balls[best_cand]:
            coverage[bidx] += 1

        # Check if all covered
        if coverage.min() > 0:
            break

        # Check time
        if time.time() - t0 > DEADLINE - MARGIN:
            break

    # Polish phase: try to remove or swap codewords
    # First, try to remove codewords that are redundant
    if time.time() - t0 < DEADLINE - MARGIN:
        # Compute coverage contribution of each selected codeword
        # coverage_from[i] = how many words are ONLY covered by selected[i]
        # Actually, we need: for each word, which selected codewords cover it
        # Simpler: for each selected codeword, count words where coverage would drop to 0 if removed
        # i.e., words in ball(selected[i]) that are not in ball of any other selected

        # Build a set of covered words per selected codeword
        # For efficiency, use the coverage array and temporarily decrement
        improved = True
        while improved and time.time() - t0 < DEADLINE - MARGIN:
            improved = False
            for si in range(len(selected)):
                if time.time() - t0 > DEADLINE - MARGIN:
                    break
                # Tentatively remove selected[si]
                bidxs = balls[selected[si]]
                # Check if any word in this ball has coverage == 1 (only covered by this codeword)
                # We need to check coverage before removal
                # If coverage[bidx] == 1 for any bidx in bidxs, then removing would leave it uncovered
                # But coverage includes this codeword's contribution
                # So if coverage[bidx] == 1, this codeword is the only one covering it
                min_cov_in_ball = coverage[bidxs].min()
                if min_cov_in_ball > 1:
                    # Safe to remove
                    coverage[bidxs] -= 1
                    selected.pop(si)
                    improved = True
                    break  # restart scan from beginning

    # Final polish: try swaps - replace one codeword with another that allows further removal
    # This is complex; skip for now if time is tight

    # Ensure we have a valid covering
    # If not fully covered (shouldn't happen with greedy), add more
    if coverage.min() <= 0 and time.time() - t0 < DEADLINE - MARGIN:
        while coverage.min() <= 0:
            # Find uncovered words
            uncovered = np.where(coverage == 0)[0]
            # For each uncovered word, find which candidate codeword covers it and has max coverage
            # Pick the candidate that covers the most uncovered words
            best_cand = -1
            best_count = -1
            for uidx in uncovered:
                # Find all codewords within radius of uidx
                # This is the reverse ball: which codewords cover uidx
                # For radius 1, uidx is covered by codewords at distance <= 1
                # Compute reverse ball
                uword = words[uidx]
                reverse_ball = [uidx]
                for pos in range(n):
                    for v in range(q):
                        if v == uword[pos]:
                            continue
                        new_word = uword.tolist()
                        new_word[pos] = v
                        reverse_ball.append(idx_map[tuple(new_word)])
                # For each candidate in reverse_ball, count how many uncovered words it covers
                for cand in reverse_ball:
                    if cand in selected:
                        continue
                    # Count uncovered words in ball(cand)
                    bidxs = balls[cand]
                    count = coverage[bidxs][coverage[bidxs] == 0].shape[0]
                    if count > best_count:
                        best_count = count
                        best_cand = cand
            if best_cand == -1:
                break
            selected.append(best_cand)
            for bidx in balls[best_cand]:
                coverage[bidx] += 1

    # Convert selected indices to codeword lists
    result = [words[i].tolist() for i in selected]

    # Ensure distinctness (should be by construction, but verify)
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)

    return unique_result
