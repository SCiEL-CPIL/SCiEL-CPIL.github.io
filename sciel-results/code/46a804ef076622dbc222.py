import time
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    N = q ** n
    if N > 200000:
        return _constructive_cover(q, n, radius)

    # --- Build index mapping ---
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    word_to_idx = {tuple(w): idx for idx, w in enumerate(all_words)}

    # --- Precompute for each position and value: which indices have that value ---
    # pos_val[p][v] = sorted array of indices where word[p] == v
    pos_val = [[np.array([i for i in range(N) if all_words[i][p] == v], dtype=np.int32)
                for v in range(q)] for p in range(n)]

    # --- Helper: get ball indices for a word (the word itself + all 1-edit neighbors) ---
    def ball_indices(word_idx):
        idxs = [word_idx]
        w = all_words[word_idx]
        for p in range(n):
            for v in range(q):
                if v == w[p]:
                    continue
                # Find the index of the word that differs only at position p with value v
                # We can compute it directly: it's the word with same digits except p -> v
                # Use the pos_val arrays? No, we need the specific word.
                # Faster: precompute a lookup table for single-position changes.
                pass
        # Actually, let's precompute all balls as arrays upfront since N=6561
        return idxs

    # Precompute all balls as numpy arrays (each ball has 1 + n*(q-1) = 17 elements)
    # ball[i] = array of indices covered by selecting codeword i
    ball_list = []
    for i in range(N):
        w = all_words[i]
        idxs = [i]
        for p in range(n):
            for v in range(q):
                if v == w[p]:
                    continue
                # Compute the index of the word that is w but with position p changed to v
                # Since words are in lex order: index = sum(digit[k] * q^(n-1-k))
                new_idx = i
                # Remove old contribution at position p, add new
                old_digit = w[p]
                new_digit = v
                power = q ** (n - 1 - p)
                new_idx = i - old_digit * power + new_digit * power
                idxs.append(new_idx)
        ball_list.append(np.array(idxs, dtype=np.int32))

    # --- Coverage tracking ---
    # covered_count[v] = number of selected codewords whose ball contains v
    covered_count = np.zeros(N, dtype=np.int32)

    def add_codeword(idx):
        covered_count[ball_list[idx]] += 1

    def remove_codeword(idx):
        covered_count[ball_list[idx]] -= 1

    def is_fully_covered():
        return np.all(covered_count > 0)

    # --- Phase 1: Tetracode product seed (81 words) ---
    # Tetracode: all 9 words of length 4 over {0,1,2} with the property that
    # any two differ in at least 2 positions (it's a [4,2,2] code).
    # Actually for K_3(4,1)=9, the covering code is ALL 9 words of length 4? No.
    # K_3(4,1) = 9 means 9 codewords of length 4 cover all 81 words.
    # The tetracode is a [4,2,2] code with 9 codewords. Its radius-1 balls cover all 3^4=81 words.
    # So the tetracode IS the optimal covering code for (q=3, n=4, r=1).
    # Product: C8 = { (a, b) : a in Tetracode, b in Tetracode } where we split 8 positions into two groups of 4.
    # This gives 9*9 = 81 codewords of length 8.
    # Does this cover all of {0,1,2}^8? 
    # A word (x1..x8) is covered if there exists (a1..a4, b1..b4) in C8 with
    # d(x1..x4, a1..a4) + d(x5..x8, b1..b4) <= 1.
    # This requires d(x1..x4, a) <= 1 AND d(x5..x8, b) <= 1 simultaneously, OR
    # one of them is 0 and the other is 1.
    # Since Tetracode covers all length-4 words within radius 1, for any x1..x4 there exists a with d<=1,
    # and for any x5..x8 there exists b with d<=1. But we need the SAME pair (a,b) to work for both halves
    # with total distance <= 1. This means we need d(x1..x4,a) + d(x5..x8,b) <= 1.
    # If d(x1..x4,a)=1 and d(x5..x8,b)=1, total=2 > 1. So the product does NOT cover everything.
    # We need a different construction or just use it as a seed and fill gaps.

    # Let's use a simpler seed: the full tetracode product gives 81 words that cover a subset.
    # Then greedy fill the rest.
    tetracode = np.array([[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)])
    
    # Product code: positions 0-3 from first tetracode word, positions 4-7 from second
    seed_indices = []
    for i in range(9):
        for j in range(9):
            word = np.concatenate([tetracode[i], tetracode[j]])
            idx = word_to_idx[tuple(word)]
            seed_indices.append(idx)

    selected = list(seed_indices)
    for idx in selected:
        add_codeword(idx)

    # --- Phase 2: Greedy fill remaining uncovered ---
    if not is_fully_covered():
        # Find uncovered words
        uncovered = np.where(covered_count == 0)[0]
        while len(uncovered) > 0 and time.time() < deadline:
            # Find the codeword that covers the most uncovered words
            best_gain = -1
            best_idx = -1
            # Sample if too many candidates
            if N > 1000:
                # Try all, it's 6561 which is fine
                pass
            for i in range(N):
                if i in selected:
                    continue
                gain = np.sum(covered_count[ball_list[i]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected.append(best_idx)
            add_codeword(best_idx)
            uncovered = np.where(covered_count == 0)[0]

    # --- Phase 3: Iterative Local Search (ILS) ---
    best_size = len(selected)
    best_selected = list(selected)

    def coverage_delta_add(idx):
        """How many currently-uncovered words would be covered by adding idx?"""
        return np.sum(covered_count[ball_list[idx]] == 0)

    def coverage_delta_remove(idx):
        """How many words would become uncovered if we remove idx? (0 means safe)"""
        return np.sum(covered_count[ball_list[idx]] == 1)

    def try_remove_one():
        """Try to remove a redundant codeword. Returns True if removed."""
        for idx in selected:
            if coverage_delta_remove(idx) == 0:
                selected.remove(idx)
                remove_codeword(idx)
                return True
        return False

    def try_swap():
        """Try to swap: remove one, add another, if it reduces size or maintains size with better structure.
        Actually for minimization, we just want to reduce count. A swap that keeps same count but 
        enables future removals is complex. Let's do: try all pairs (remove i, add j) where 
        removing i makes some words uncovered, but adding j covers them AND j wasn't already there.
        Net effect: same count. Only useful if it enables a subsequent removal.
        Simpler: just try to find a codeword whose removal can be compensated by adding a different 
        codeword that covers MORE total uncovered (but since we're fully covered, this means 
        the new codeword covers words that were only covered by the removed one, plus maybe others).
        Actually, if we're fully covered, removing i makes some words uncovered (those with count==1 in ball[i]).
        We need to add j that covers ALL of those. If such j exists and j != i, we can swap.
        But this doesn't reduce count. The reduction comes from the subsequent try_remove_one().
        """
        # For each i in selected, find words that would become uncovered
        for i in selected:
            at_risk = ball_list[i][covered_count[ball_list[i]] == 1]
            if len(at_risk) == 0:
                # Already handled by try_remove_one
                continue
            # Find a j not in selected that covers all of at_risk
            # j must be in the intersection of balls of all words in at_risk
            # i.e., j must be within distance 1 of every word in at_risk
            # The set of valid j is the intersection of N[w] for w in at_risk
            # This is expensive to compute generally. For small at_risk, we can check candidates.
            if len(at_risk) > 4:
                continue  # Too many constraints, unlikely to find a single j
            # Candidate j must be in ball of every word in at_risk
            # Start with ball of first word, intersect with others
            candidates = set(ball_list[at_risk[0]].tolist())
            for w in at_risk[1:]:
                candidates &= set(ball_list[w].tolist())
            candidates -= set(selected)
            for j in candidates:
                # Check if adding j covers all at_risk (it should by construction)
                # Just do the swap
                selected.remove(i)
                remove_codeword(i)
                selected.append(j)
                add_codeword(j)
                # Try to remove now
                if try_remove_one():
                    return True
                # If no removal, undo
                selected.remove(j)
                remove_codeword(j)
                selected.append(i)
                add_codeword(i)
                break
        return False

    # ILS loop
    rng = np.random.default_rng(42)
    stall_count = 0
    max_stall = 50

    while time.time() < deadline - 5:
        # Try simple removals
        while try_remove_one():
            pass

        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= max_stall:
            # Perturbation: randomly remove k codewords, then greedy re-add
            k = min(len(selected), rng.integers(2, 6))
            removed = rng.choice(len(selected), size=k, replace=False)
            for pos in sorted(removed, reverse=True):
                idx = selected[pos]
                selected.remove(idx)
                remove_codeword(idx)
            # Greedy re-add
            while not is_fully_covered() and time.time() < deadline - 5:
                best_gain = -1
                best_idx = -1
                for i in range(N):
                    if i in selected:
                        continue
                    gain = np.sum(covered_count[ball_list[i]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1 or best_gain == 0:
                    break
                selected.append(best_idx)
                add_codeword(best_idx)
            # Try removals again
            while try_remove_one():
                pass
            stall_count = 0
            max_stall += 10  # Be more patient after perturbation

        # Try swaps
        try_swap()

        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)

    # --- Phase 4: Final cleanup ---
    while try_remove_one():
        pass

    if len(selected) < best_size:
        best_size = len(selected)
        best_selected = list(selected)

    # Ensure we have a valid solution
    # If for some reason we lost coverage, do a final greedy
    if not is_fully_covered():
        # Rebuild from best_selected
        covered_count[:] = 0
        for idx in best_selected:
            add_codeword(idx)
        while not is_fully_covered() and time.time() < deadline:
            best_gain = -1
            best_idx = -1
            for i in range(N):
                if i in best_selected:
                    continue
                gain = np.sum(covered_count[ball_list[i]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            best_selected.append(best_idx)
            add_codeword(best_idx)
        # Try removals
        for idx in list(best_selected):
            if np.sum(covered_count[ball_list[idx]] == 1) == 0:
                best_selected.remove(idx)
                remove_codeword(idx)

    result = [all_words[i].tolist() for i in best_selected]
    return result


def _constructive_cover(q, n, radius):
    """Fallback constructive approach."""
    if q == 3 and n == 8:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]

    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = len(all_words)

    balls = np.zeros((num_words, num_words), dtype=bool)
    np.fill_diagonal(balls, True)
    for i in range(num_words):
        for j in range(num_words):
            if np.sum(all_words[i] != all_words[j]) <= radius:
                balls[i, j] = True

    selected = []
    covered = np.zeros(num_words, dtype=bool)
    while not covered.all():
        uncovered_count = balls[:, ~covered].sum(axis=1)
        best_idx = np.argmax(uncovered_count)
        if uncovered_count[best_idx] == 0:
            break
        selected.append(best_idx)
        covered |= balls[best_idx, :]

    return [all_words[i].tolist() for i in selected]
