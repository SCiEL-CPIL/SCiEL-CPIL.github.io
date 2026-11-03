"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START

def construct(n):
    """Return shortest superpermutation found for n symbols within time budget."""
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431214321342132431213243214321"

    start_time = time.time()
    TIME_LIMIT = 140  # Safety margin before 150s limit

    # Build initial superpermutation using recursive construction
    word = build_initial_superperm(n)

    # For n >= 5, try to shorten using structural compression
    if n >= 5 and time.time() - start_time < TIME_LIMIT:
        word = compress_superperm(word, n, start_time, TIME_LIMIT)

    return "".join(map(str, word))

def build_initial_superperm(n):
    """Classic recursive construction: length = 1! + 2! + ... + n!"""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def compress_superperm(word, n, start_time, time_limit):
    """
    Compress a superpermutation using incremental count tracking.
    
    Key idea: maintain a dict of permutation -> count.
    When removing position pos, only the n-1 windows that were affected
    need to be updated. A removal is safe iff no permutation's count drops to 0.
    """
    current = list(word)
    L = len(current)

    target = 1
    for i in range(2, n + 1):
        target *= i

    # Build initial permutation counts
    perm_counts = {}
    for i in range(L - n + 1):
        window = tuple(current[i:i + n])
        if len(set(window)) == n:
            perm_counts[window] = perm_counts.get(window, 0) + 1

    covered = len(perm_counts)
    if covered != target:
        # Fallback: return as-is
        return current

    # Helper: get the set of permutation windows affected by removing position pos
    # These are windows starting at pos-n+1 through pos (inclusive), i.e., n windows
    # but only those that are valid permutations matter
    def get_affected_windows(w, pos, n):
        """Return list of (window_tuple, old_count) for windows containing position pos."""
        affected = []
        for start in range(max(0, pos - n + 1), min(pos + 1, len(w) - n + 1)):
            window = tuple(w[start:start + n])
            if len(set(window)) == n:
                old_count = perm_counts.get(window, 0)
                affected.append((window, old_count))
        return affected

    # Multi-pass greedy removal with incremental updates
    # Pass 1: Remove non-essential characters (those not in any unique permutation)
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        L = len(current)
        if L < n:
            break

        # Identify which positions are "essential" (part of at least one permutation
        # that appears exactly once)
        essential = [False] * L
        for perm, count in perm_counts.items():
            if count == 1:
                # Find where this permutation occurs
                # We need to search, but we can optimize by only checking windows
                # Actually, let's use a different approach: mark all positions that
                # are in windows with count==1
                pass

        # Better approach: for each position, check if removing it would cause
        # any permutation count to drop to 0
        # Try positions from right to left (less disruptive to earlier structure)
        for pos in range(L - 1, -1, -1):
            if time.time() - start_time > time_limit:
                return current
            if L - 1 < n:
                break

            # Get affected windows
            affected = get_affected_windows(current, pos, n)

            # Check if any affected window has count == 1 (would be lost)
            safe = True
            for window, old_count in affected:
                if old_count == 1:
                    safe = False
                    break

            if not safe:
                continue

            # Removal is safe: perform it
            # Update perm_counts
            for window, old_count in affected:
                if old_count == 1:
                    del perm_counts[window]
                else:
                    perm_counts[window] = old_count - 1

            # Remove the character
            current = current[:pos] + current[pos + 1:]
            L -= 1
            improved = True
            break  # Restart scan from right

        # If no improvement in this full pass, try a different strategy

    # Pass 2: Try removing pairs of adjacent characters
    if time.time() - start_time < time_limit:
        L = len(current)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            if L < n:
                break

            for pos in range(L - 1, 0, -1):
                if time.time() - start_time > time_limit:
                    return current
                if L - 2 < n:
                    break

                # Try removing positions pos and pos-1
                # Affected windows: those containing pos-1 or pos
                affected = []
                seen_windows = set()
                for p in [pos - 1, pos]:
                    for start in range(max(0, p - n + 1), min(p + 1, L - n + 1)):
                        window = tuple(current[start:start + n])
                        if len(set(window)) == n and window not in seen_windows:
                            seen_windows.add(window)
                            old_count = perm_counts.get(window, 0)
                            affected.append((window, old_count))

                safe = True
                for window, old_count in affected:
                    if old_count <= 1:
                        safe = False
                        break

                if not safe:
                    continue

                # Perform removal
                for window, old_count in affected:
                    if old_count == 1:
                        del perm_counts[window]
                    else:
                        perm_counts[window] = old_count - 1

                current = current[:pos - 1] + current[pos + 1:]
                L -= 2
                improved = True
                break

    # Pass 3: More aggressive - try removing any single character with a different
    # heuristic order (middle of word first, to maintain balance)
    if time.time() - start_time < time_limit:
        L = len(current)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            if L < n:
                break

            # Try positions in order of "centrality" (middle first)
            mid = L // 2
            positions = sorted(range(L), key=lambda x: abs(x - mid))

            for pos in positions:
                if time.time() - start_time > time_limit:
                    return current
                if L - 1 < n:
                    break

                affected = get_affected_windows(current, pos, n)

                safe = True
                for window, old_count in affected:
                    if old_count == 1:
                        safe = False
                        break

                if not safe:
                    continue

                for window, old_count in affected:
                    if old_count == 1:
                        del perm_counts[window]
                    else:
                        perm_counts[window] = old_count - 1

                current = current[:pos] + current[pos + 1:]
                L -= 1
                improved = True
                break

    return current

# EVOLVE-BLOCK-END
