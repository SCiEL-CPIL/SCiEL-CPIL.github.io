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

    # Iterative deletion using priority queue based on window frequencies
    # Positions with higher surrounding window frequencies are more likely deletable
    
    def rebuild_priority_queue():
        """Build a max-heap of (negative_max_freq, pos) for positions 1..L-2."""
        L = len(word)
        heap = []
        for pos in range(1, L - 1):
            lo = max(0, pos - n + 1)
            hi = min(pos, L - n)
            if lo > hi:
                continue
            max_freq = 0
            for w in range(lo, hi + 1):
                p = tuple(word[w:w + n])
                f = window_freq.get(p, 0)
                if f > max_freq:
                    max_freq = f
            if max_freq > 1:
                heapq.heappush(heap, (-max_freq, pos))
        return heap

    def try_delete_from_heap(heap, word_ref, freq_ref):
        """Try to delete positions from the heap. Returns number of successful deletions."""
        deleted = 0
        while heap and time.time() < deadline - 15:
            neg_freq, pos = heapq.heappop(heap)
            L = len(word_ref)
            if pos <= 0 or pos >= L - 1:
                continue
            lo = max(0, pos - n + 1)
            hi = min(pos, L - n)
            if lo > hi:
                continue
            # Check if still deletable
            can_delete = True
            for w in range(lo, hi + 1):
                p = tuple(word_ref[w:w + n])
                if freq_ref.get(p, 0) <= 1:
                    can_delete = False
                    break
            if can_delete:
                # Perform deletion
                old_perms = [tuple(word_ref[w:w + n]) for w in range(lo, hi + 1)]
                word_ref[pos] = None  # Mark for deletion
                # Actually delete
                word_ref = word_ref[:pos] + word_ref[pos + 1:]
                L = len(word_ref)
                new_lo = max(0, pos - n)
                new_hi = min(pos, L - n)
                for p in old_perms:
                    freq_ref[p] -= 1
                    if freq_ref[p] == 0:
                        del freq_ref[p]
                for w in range(new_lo, new_hi + 1):
                    p = tuple(word_ref[w:w + n])
                    freq_ref[p] = freq_ref.get(p, 0) + 1
                deleted += 1
                # Add nearby positions to heap
                for new_pos in range(max(1, pos - n), min(len(word_ref) - 1, pos + n)):
                    lo2 = max(0, new_pos - n + 1)
                    hi2 = min(new_pos, len(word_ref) - n)
                    if lo2 > hi2:
                        continue
                    max_freq = 0
                    for w2 in range(lo2, hi2 + 1):
                        p2 = tuple(word_ref[w2:w2 + n])
                        f2 = freq_ref.get(p2, 0)
                        if f2 > max_freq:
                            max_freq = f2
                    if max_freq > 1:
                        heapq.heappush(heap, (-max_freq, new_pos))
                # Return updated word_ref
                return word_ref, deleted
            # If not deletable, skip
        return word_ref, deleted

    # Phase 1: Priority-queue based deletion
    max_pq_rounds = 20
    for rnd in range(max_pq_rounds):
        if time.time() > deadline - 15:
            break
        L = len(word)
        if L <= n:
            break
        
        heap = rebuild_priority_queue()
        if not heap:
            break
        
        initial_L = len(word)
        word, deleted = try_delete_from_heap(heap, word, window_freq)
        
        if deleted == 0:
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

    # Second round of internal deletions after trimming - use both PQ and linear passes
    if time.time() < deadline - 10:
        L = len(word)
        window_freq = defaultdict(int)
        for i in range(L - n + 1):
            p = tuple(word[i:i + n])
            window_freq[p] += 1

        # PQ-based deletion
        for rnd in range(15):
            if time.time() > deadline - 10:
                break
            L = len(word)
            if L <= n:
                break
            heap = rebuild_priority_queue()
            if not heap:
                break
            word, deleted = try_delete_from_heap(heap, word, window_freq)
            if deleted == 0:
                break

        # Linear passes as backup
        for rnd in range(10):
            if time.time() > deadline - 10:
                break
            L = len(word)
            if L <= n:
                break
            made_change = False
            pos = 1
            while pos < L - 1:
                if time.time() > deadline - 10:
                    break
                if do_delete(pos):
                    made_change = True
                    L = len(word)
                else:
                    pos += 1
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
