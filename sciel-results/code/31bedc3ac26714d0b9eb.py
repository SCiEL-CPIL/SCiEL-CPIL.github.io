"""Covering code K_3(8,1) via greedy set cover + aggressive local search."""
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 17 for q=3, n=8

    # Words as (total, n) int32 array
    words = np.array(list(np.array(np.unravel_index(np.arange(total), (q,)*n)).T), dtype=np.int32)

    # Precompute cover_matrix: cover_matrix[c, w] = True iff w within distance <= radius of c
    # For radius=1: 17 entries per row.
    cover_matrix = np.zeros((total, total), dtype=bool)
    for c_idx in range(total):
        cover_matrix[c_idx, c_idx] = True
        c_word = words[c_idx]
        for i in range(n):
            for v in range(q):
                if v == c_word[i]:
                    continue
                new_word = c_word.copy()
                new_word[i] = v
                w_idx = 0
                for j in range(n):
                    w_idx = w_idx * q + new_word[j]
                cover_matrix[c_idx, w_idx] = True

    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    selected = []

    while uncovered.any():
        if time.time() - start_time > deadline:
            break
        coverage_count = cover_matrix[:, uncovered].sum(axis=1)
        best_c = int(np.argmax(coverage_count))
        if coverage_count[best_c] == 0:
            break
        selected.append(best_c)
        uncovered[cover_matrix[best_c, :]] = False

    def compute_coverage_counts(sel_list):
        counts = np.zeros(total, dtype=np.int32)
        for c in sel_list:
            counts[cover_matrix[c, :]] += 1
        return counts

    counts = compute_coverage_counts(selected)

    # Phase 1: Remove redundant codewords (1-move: delete)
    # A codeword c is removable iff no word has counts==1 that c covers.
    # i.e., for all w in ball(c), counts[w] >= 2.
    # Equivalently: min(counts[ball(c)]) >= 2.
    changed = True
    while changed and (time.time() - start_time) < deadline - 15:
        changed = False
        # Vectorized: for each selected c, check if any word in its ball has count == 1
        # Build a (len(selected), total) slice? Too big. Instead loop but with fast check.
        for idx in range(len(selected)):
            if time.time() - start_time > deadline - 15:
                break
            c = selected[idx]
            # Fast: check if any word in c's ball has count == 1
            ball_mask = cover_matrix[c, :]
            if not (ball_mask & (counts == 1)).any():
                selected.pop(idx)
                counts[ball_mask] -= 1
                changed = True
                break  # restart scan

    # Phase 2: 2-move local search: remove c1, add c2 (net zero or negative)
    # Actually we want to try: remove c1 AND c2, add c3? That's 3-move.
    # Simpler: try to replace one codeword c with another c' such that:
    #   - c' is not already selected
    #   - c' covers all words that only c covers (i.e., words w where counts[w]==1 and c covers w)
    #   - This keeps coverage valid and same size. But we want to REDUCE size.
    # So the real goal: find c1, c2 in selected such that removing both and adding c3 (not selected)
    # still covers everything. That reduces size by 1.
    # That's expensive. Instead, do iterative 1-remove then try to add a "better" codeword
    # that allows future removals.

    # Better approach: iterate the remove phase, and between passes, try a "swap" that
    # replaces c with c' where c' covers the exclusive set of c AND c' helps cover words
    # that are currently covered by only 2 codewords (making them covered by 1 after swap... no).

    # Let's do: for each c in selected, compute its exclusive set E(c) = {w : c covers w and counts[w]==1}.
    # If E(c) is empty, c is removable (handled in phase 1).
    # If E(c) is non-empty, we need to find c' not in selected such that c' covers all of E(c).
    # Then swap c -> c'. This doesn't reduce size but may enable future removals.
    # After swapping, re-run phase 1.

    # To make this efficient: precompute for each word w, the set of codewords that cover it.
    # cover_by_word[w] = list of c such that cover_matrix[c, w] = True.
    # For each w, this is a ball around w, size 17.
    # Then E(c) = intersection over... no, E(c) = {w in ball(c) : counts[w]==1}.
    # For c' to cover all of E(c), we need: for all w in E(c), c' in cover_by_word[w].
    # i.e., c' in intersection_{w in E(c)} cover_by_word[w].
    # The intersection of 17-element sets can be computed quickly.

    # Precompute cover_by_word as a list of frozensets
    # cover_by_word[w] = frozenset of c indices that cover w
    # This is the transpose of cover_matrix.
    # total=6561, each set has 17 elements. 6561*17 = 111k elements total. Fine.
    cover_by_word = [None] * total
    for w in range(total):
        # Words that cover w are those at distance <=1 from w, i.e., w and its neighbors
        # cover_matrix[c, w] = True iff c is within distance 1 of w iff w is within distance 1 of c.
        # So cover_by_word[w] = {c : cover_matrix[c, w]} = ball around w.
        # We can compute this from cover_matrix[:, w]
        cover_by_word[w] = frozenset(np.where(cover_matrix[:, w])[0])

    # Phase 2: Swap to enable removals
    # Iterate: for each c, if E(c) non-empty, find c' in intersection of cover_by_word[w] for w in E(c),
    # c' not in selected, and c' != c. Swap. Then re-run phase 1.
    sel_set = set(selected)

    max_swap_rounds = 50
    for round_num in range(max_swap_rounds):
        if time.time() - start_time > deadline - 15:
            break

        # Phase 1: remove redundant
        any_removed = False
        changed = True
        while changed:
            changed = False
            for idx in range(len(selected)):
                if time.time() - start_time > deadline - 15:
                    break
                c = selected[idx]
                ball_mask = cover_matrix[c, :]
                if not (ball_mask & (counts == 1)).any():
                    selected.pop(idx)
                    sel_set.discard(c)
                    counts[ball_mask] -= 1
                    any_removed = True
                    changed = True
                    break
            if time.time() - start_time > deadline - 15:
                break

        if any_removed:
            continue  # re-scan

        # Phase 2: try swaps
        any_swapped = False
        for idx in range(len(selected)):
            if time.time() - start_time > deadline - 15:
                break
            c = selected[idx]
            ball_mask = cover_matrix[c, :]
            only_c = ball_mask & (counts == 1)
            if not only_c.any():
                continue  # would have been removed
            only_c_indices = np.where(only_c)[0]
            # Find c' in intersection of cover_by_word[w] for w in only_c_indices
            # Start with cover_by_word[only_c_indices[0]], intersect with rest
            candidate_set = cover_by_word[only_c_indices[0]]
            for w in only_c_indices[1:]:
                candidate_set = candidate_set & cover_by_word[w]
            # Remove already-selected codewords
            candidate_set = candidate_set - sel_set
            if not candidate_set:
                continue
            # Pick a candidate: prefer one that covers the most words with count==2 (to enable future removals)
            # Or just pick any. Let's pick the one that maximizes overlap with count==2 words.
            best_c2 = None
            best_score = -1
            for c2 in candidate_set:
                # Score: how many words in ball(c2) have count == 2
                score = (cover_matrix[c2, :] & (counts == 2)).sum()
                if score > best_score:
                    best_score = score
                    best_c2 = c2
            if best_c2 is None:
                continue
            # Perform swap
            selected[idx] = best_c2
            sel_set.discard(c)
            sel_set.add(best_c2)
            counts[cover_matrix[c, :]] -= 1
            counts[cover_matrix[best_c2, :]] += 1
            any_swapped = True
            break  # restart scan

        if not any_swapped and not any_removed:
            break  # converged

    codewords = [words[c].tolist() for c in selected]
    return codewords
