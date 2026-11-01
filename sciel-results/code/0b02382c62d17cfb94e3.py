import time
import random
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110

    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i

    # Symbol insertion construction: build superpermutation for n-1, then expand to n
    # Start with n-1 symbols
    word = list(range(1, n))  # identity permutation of 1..n-1

    for m in range(n - 1, n):
        # We have a superpermutation of 1..m, want to expand to 1..m+1
        new_sym = m + 1
        # Find all (m-1)-permutations as consecutive windows
        # For a superpermutation of 1..m, every permutation of 1..m appears as m consecutive symbols
        # We need to insert new_sym so that every permutation of 1..m+1 appears

        # Standard insertion: for each permutation of 1..m in the word,
        # insert new_sym at each position 0..m to get m+1 permutations
        # But we want to do this efficiently with maximum overlap

        # The classic insertion method:
        # Take the word for m symbols. For each occurrence of each permutation of 1..m,
        # we need to add new_sym in all m+1 positions.
        # A systematic way: process the word left to right, and for each (m-1)-length 
        # unique window, insert the new symbol in a specific pattern.

        # Better approach: the "insertion" construction
        # For m=2: "12" -> insert 3: "13212" (covers 132,321,213,123,231)
        # Actually let me use the standard recursive insertion:
        # S(1) = "1"
        # S(k) from S(k-1): for each permutation p in S(k-1) (as windows of length k-1),
        #   insert k at position 0, then shift: the pattern is
        #   p[0] k p[1..k-2] p[0..k-2] ... no, let me think more carefully.

        # The standard "insertion" superpermutation:
        # S(k) is built from S(k-1) by taking each (k-1)-permutation window in S(k-1),
        # and for the i-th such window (0-indexed), insert k at position i in that window,
        # then append the remaining k-1 symbols.

        # Actually, the well-known construction:
        # S(1) = "1"
        # S(2) = "121"
        # S(3) = "1321231"  (length 7 = 1!+2!+3!-1... no, 1+2+6=9? Let me check)
        #   Actually S(3) should have length 3! + 2! + 1! = 9? No.
        #   The standard formula: |S(n)| = n! + (n-1)! + ... + 1! - 1
        #   S(1) = "1", len=1
        #   S(2) = "121", len=3 = 2!+1!-1 = 2+1-1=2? No, 2!+1! = 3, minus 1 = 2? That's wrong.
        #   Let me just use the known result: |S(n)| = sum_{k=1}^{n} k!
        #   S(1)=1, S(2)=3, S(3)=9, S(4)=33, S(5)=153, S(6)=873, S(7)=5913

        # The insertion algorithm:
        # Given S(k-1) as a list, to get S(k):
        # For each position i in 0..len(S(k-1))-1 where we find a (k-1)-permutation window,
        # we insert k at a specific position.

        # Let me implement the standard "rotational insertion":
        # S(k) from S(k-1):
        #   result = []
        #   For each (k-1)-length window in S(k-1) that is a permutation:
        #     Let the window be w = [w0, w1, ..., w_{k-2}]
        #     Insert k at position (index_of_this_window) mod k in the window
        #     Then append the rest of the window after k

        # Actually, the cleanest formulation of the insertion construction:
        # S(k) is obtained from S(k-1) by:
        #   For each occurrence of a (k-1)-permutation at position i in S(k-1):
        #     replace that (k-1)-window with: w[i%k] ... k ... remaining
        #   This is getting complicated. Let me use a simpler known algorithm.

        # Simple insertion algorithm (from literature):
        # S(k) from S(k-1):
        #   new_word = []
        #   For each i from 0 to len(S(k-1)) - (k-1):
        #     window = S(k-1)[i:i+k-1]
        #     if window is a permutation of 1..k-1:
        #       # Insert k at position (i mod k) in this window
        #       insert_pos = i % k
        #       new_window = window[:insert_pos] + [k] + window[insert_pos:]
        #       # But we need to handle overlaps...

        # This is getting too complex to get right without testing. Let me use a different
        # well-known approach: the "gap insertion" method.

        # Gap insertion method:
        # S(k) from S(k-1):
        #   Take S(k-1). For each of the k gaps (before first, between each pair, after last),
        #   we insert k and a copy of the preceding (k-1)-permutation in a rotating pattern.

        # Actually, let me just implement the standard recursive construction properly:
        # The known formula gives S(n) of length sum(k! for k=1..n).
        # For n=7: 1+2+6+24+120+720+5040 = 5913

        # The construction:
        # S(1) = [1]
        # S(k) from S(k-1):
        #   For each (k-1)-permutation window w at position i in S(k-1):
        #     Create: w[0], k, w[1], w[0], k, w[1], ... no.

        # Let me use the most standard formulation I know:
        # S(k) is built by taking S(k-1) and for each (k-1)-perm window at index i,
        # inserting k at position (i mod k), and the window becomes length k.
        # The new word is formed by overlapping these expanded windows.

        # I'll implement it step by step:
        if m == n - 1:
            # We have word for symbols 1..m (which is n-1)
            # Expand to 1..m+1 (which is n)
            new_sym = m + 1

            # Find all (m)-length windows that are permutations of 1..m
            # and their positions
            windows = []
            for i in range(len(word) - m + 1):
                w = tuple(word[i:i+m])
                if len(set(w)) == m and all(1 <= x <= m for x in w):
                    windows.append((i, w))

            # Build new word using insertion
            # For the i-th permutation window (0-indexed in order of appearance),
            # insert new_sym at position (i mod (m+1)) in the window
            # Then the expanded window has length m+1
            # Overlap consecutive expanded windows by m symbols

            new_word = []
            for idx, (pos, w) in enumerate(windows):
                insert_pos = idx % (m + 1)
                # Insert new_sym at insert_pos in w
                expanded = list(w[:insert_pos]) + [new_sym] + list(w[insert_pos:])

                if not new_word:
                    new_word = expanded
                else:
                    # Try to overlap: find maximum overlap between end of new_word and start of expanded
                    max_ov = min(len(new_word), len(expanded) - 1)
                    overlap = 0
                    for o in range(max_ov, -1, -1):
                        if o == 0 or new_word[len(new_word)-o:] == expanded[:o]:
                            overlap = o
                            break
                    new_word += expanded[overlap:]

            word = new_word

    best_word = word[:]

    # Verify and compute counts
    all_perms = list(itertools.permutations(range(1, n + 1)))
    all_perms_set = set(all_perms)

    def compute_counts(w):
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            t = tuple(w[i:i+n])
            if t in all_perms_set:
                counts[t] += 1
        return counts

    def is_valid_from_counts(counts):
        for p in all_perms:
            if counts[p] < 1:
                return False
        return True

    current_len = len(best_word)
    current_counts = compute_counts(best_word)

    if not is_valid_from_counts(current_counts):
        # Fallback to naive construction if insertion failed
        word = [1]
        for m2 in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m2 + 2):
                window = tuple(word[i:i + m2 - 1])
                if len(set(window)) == m2 - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m2] + list(p)
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        best_word = word[:]
        current_len = len(best_word)
        current_counts = compute_counts(best_word)
        if not is_valid_from_counts(current_counts):
            return "".join(map(str, best_word))

    # Main search loop: deletion-based optimization
    while time.time() < deadline:
        found_improvement = False

        # Phase 1: Try to find redundant positions to remove
        for idx in range(len(best_word)):
            if len(best_word) <= n:
                break
            start_lo = max(0, idx - n + 1)
            start_hi = min(idx, len(best_word) - n)

            can_remove = True
            for s in range(start_lo, start_hi + 1):
                t = tuple(best_word[s:s+n])
                if t in all_perms_set and current_counts[t] == 1:
                    can_remove = False
                    break

            if can_remove:
                candidate = best_word[:]
                del candidate[idx]
                new_counts = compute_counts(candidate)
                if is_valid_from_counts(new_counts):
                    best_word = candidate
                    current_len = len(best_word)
                    current_counts = new_counts
                    found_improvement = True
                    break

        # Phase 2: If no deletion found, try random moves
        if not found_improvement:
            for _ in range(2000):
                if time.time() > deadline:
                    break
                if len(best_word) <= n:
                    break
                idx = random.randint(0, len(best_word) - 1)
                new_idx = random.randint(0, len(best_word) - 1)
                if idx == new_idx:
                    continue
                candidate = best_word[:]
                val = candidate.pop(idx)
                candidate.insert(new_idx, val)

                if len(candidate) < current_len:
                    new_counts = compute_counts(candidate)
                    if is_valid_from_counts(new_counts):
                        best_word = candidate
                        current_len = len(best_word)
                        current_counts = new_counts
                        break

    return "".join(map(str, best_word))
