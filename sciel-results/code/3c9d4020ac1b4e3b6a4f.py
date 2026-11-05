import itertools
import time
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

    # Helper to delete a position and update window_freq
    def do_delete(pos):
        """Delete position pos. Returns True if successful."""
        nonlocal word, window_freq, L
        if pos <= 0 or pos >= L - 1:
            return False

        lo = max(0, pos - n + 1)
        hi = min(pos, L - n)

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
        L = len(word)

        # New affected region
        new_lo = max(0, pos - n)
        new_hi = min(pos, L - n)

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

    # Iterative deletion: multiple rounds, forward and backward passes
    max_rounds = 30
    for rnd in range(max_rounds):
        if time.time() > deadline - 15:
            break

        L = len(word)
        if L <= n:
            break

        made_change = False

        # Forward pass
        pos = 1
        while pos < L - 1:
            if time.time() > deadline - 15:
                break
            if do_delete(pos):
                made_change = True
                L = len(word)
                # Don't increment pos; check same position again
            else:
                pos += 1

        # Backward pass
        if time.time() > deadline - 15:
            break
        L = len(word)
        pos = L - 2
        while pos > 0:
            if time.time() > deadline - 15:
                break
            if do_delete(pos):
                made_change = True
                L = len(word)
                # Don't decrement pos; check same position again
            else:
                pos -= 1

        if not made_change:
            break

    # End trimming
    if time.time() < deadline - 10:
        changed = True
        while changed:
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

    # Second round of internal deletions after trimming
    if time.time() < deadline - 10:
        L = len(word)
        window_freq = defaultdict(int)
        for i in range(L - n + 1):
            p = tuple(word[i:i + n])
            window_freq[p] += 1

        for rnd in range(20):
            if time.time() > deadline - 10:
                break

            L = len(word)
            if L <= n:
                break

            made_change = False

            # Forward pass
            pos = 1
            while pos < L - 1:
                if time.time() > deadline - 10:
                    break
                if do_delete(pos):
                    made_change = True
                    L = len(word)
                else:
                    pos += 1

            # Backward pass
            if time.time() > deadline - 10:
                break
            L = len(word)
            pos = L - 2
            while pos > 0:
                if time.time() > deadline - 10:
                    break
                if do_delete(pos):
                    made_change = True
                    L = len(word)
                else:
                    pos -= 1

            if not made_change:
                break

    # Final verification
    final_perms = set()
    for i in range(len(word) - n + 1):
        final_perms.add(tuple(word[i:i + n]))

    if len(final_perms) != len(required):
        # Fallback: return the standard construction
        word = build_standard(n)

    return "".join(map(str, word))
