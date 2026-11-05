import itertools
import time
import heapq
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    if n == 1:
        return "1"
    if n == 2:
        return "121"

    # Step 1: Build standard superpermutation (length = sum(i! for i=1..n))
    # For n=7: 1+2+6+24+120+720+5040 = 5913
    def build_standard(n):
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

    word = build_standard(n)
    L = len(word)

    # Build frequency map of all n-windows
    window_freq = defaultdict(int)
    for i in range(L - n + 1):
        p = tuple(word[i:i + n])
        window_freq[p] += 1

    # Verify all permutations present
    required = set(itertools.permutations(range(1, n + 1)))
    present = set(window_freq.keys())
    if present != required:
        return "".join(map(str, word))

    # Helper to compute min frequency of affected windows for a position
    def min_freq_at(pos):
        """Return the minimum frequency among windows affected by deleting pos, or 0 if not deletable."""
        if pos <= 0 or pos >= len(word) - 1:
            return 0
        lo = max(0, pos - n + 1)
        hi = min(pos, len(word) - n)
        if lo > hi:
            return 0
        min_f = float('inf')
        for w in range(lo, hi + 1):
            p = tuple(word[w:w + n])
            f = window_freq[p]
            if f <= 1:
                return 0
            if f < min_f:
                min_f = f
        return min_f

    # Build initial priority queue: (negative_min_freq, position)
    # We want to delete positions with highest min_freq first (easiest)
    heap = []
    for pos in range(1, len(word) - 1):
        mf = min_freq_at(pos)
        if mf > 0:
            heapq.heappush(heap, (-mf, pos))

    def do_delete(pos):
        """Delete position pos. Returns True if successful."""
        nonlocal word, window_freq
        if pos <= 0 or pos >= len(word) - 1:
            return False

        lo = max(0, pos - n + 1)
        hi = min(pos, len(word) - n)

        if lo > hi:
            return False

        # Check: all affected windows must have freq > 1
        for w in range(lo, hi + 1):
            p = tuple(word[w:w + n])
            if window_freq[p] <= 1:
                return False

        # Save old affected windows
        old_perms = [tuple(word[w:w + n]) for w in range(lo, hi + 1)]

        # Perform deletion
        word = word[:pos] + word[pos + 1:]

        # New affected region
        new_lo = max(0, pos - n)
        new_hi = min(pos, len(word) - n)

        # Decrement old
        for p in old_perms:
            window_freq[p] -= 1
            if window_freq[p] == 0:
                del window_freq[p]

        # Increment new
        for w in range(new_lo, new_hi + 1):
            p = tuple(word[w:w + n])
            window_freq[p] += 1

        return True

    # Phase 1: Heap-based deletion
    total_deletions = 0
    while time.time() < deadline - 30 and heap:
        # Pop until we find a valid entry
        deleted = False
        while heap and not deleted:
            neg_mf, pos = heapq.heappop(heap)
            # Check if this position is still valid
            if pos <= 0 or pos >= len(word) - 1:
                continue
            # Verify the min_freq is still accurate
            mf_now = min_freq_at(pos)
            if mf_now <= 0:
                continue
            if -neg_mf != mf_now:
                # Stale entry, re-push with correct value
                heapq.heappush(heap, (-mf_now, pos))
                continue
            # Try to delete
            if do_delete(pos):
                total_deletions += 1
                deleted = True
                # After deletion, update heap for positions in the affected neighborhood
                new_lo = max(0, pos - n)
                new_hi = min(pos, len(word) - n)
                update_lo = max(1, new_lo - (n - 1))
                update_hi = min(len(word) - 2, new_hi + (n - 1))
                for p in range(update_lo, update_hi + 1):
                    mf = min_freq_at(p)
                    if mf > 0:
                        heapq.heappush(heap, (-mf, p))
                break

        if not deleted and not heap:
            break

    # Phase 2: End trimming
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        L = len(word)
        if L > n:
            first_perm = tuple(word[:n])
            if window_freq[first_perm] > 1:
                window_freq[first_perm] -= 1
                if window_freq[first_perm] == 0:
                    del window_freq[first_perm]
                word = word[1:]
                L = len(word)
                new_perm = tuple(word[:n])
                window_freq[new_perm] += 1
                changed = True

        if L > n:
            last_perm = tuple(word[-n:])
            if window_freq[last_perm] > 1:
                window_freq[last_perm] -= 1
                if window_freq[last_perm] == 0:
                    del window_freq[last_perm]
                word = word[:-1]
                L = len(word)
                new_perm = tuple(word[-n:])
                window_freq[new_perm] += 1
                changed = True

    # Phase 3: Local search - try all positions in order
    if time.time() < deadline - 5:
        # Try sequential deletion passes
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            any_deleted = False
            L = len(word)
            for pos in range(1, L - 1):
                if time.time() > deadline - 5:
                    break
                if pos <= 0 or pos >= len(word) - 1:
                    continue
                lo = max(0, pos - n + 1)
                hi = min(pos, len(word) - n)
                if lo > hi:
                    continue
                can_delete = True
                for w in range(lo, hi + 1):
                    p = tuple(word[w:w + n])
                    if window_freq[p] <= 1:
                        can_delete = False
                        break
                if can_delete:
                    old_perms = [tuple(word[w:w + n]) for w in range(lo, hi + 1)]
                    word = word[:pos] + word[pos + 1:]
                    L = len(word)
                    new_lo = max(0, pos - n)
                    new_hi = min(pos, L - n)
                    for p in old_perms:
                        window_freq[p] -= 1
                        if window_freq[p] == 0:
                            del window_freq[p]
                    for w in range(new_lo, new_hi + 1):
                        p = tuple(word[w:w + n])
                        window_freq[p] += 1
                    any_deleted = True
                    # After deletion, positions shifted, restart scan
                    break
            if not any_deleted:
                break

    # Final verification
    final_perms = set()
    for i in range(len(word) - n + 1):
        final_perms.add(tuple(word[i:i + n]))

    if len(final_perms) != len(required):
        # Fallback: return the standard construction
        word = build_standard(n)

    return "".join(map(str, word))
