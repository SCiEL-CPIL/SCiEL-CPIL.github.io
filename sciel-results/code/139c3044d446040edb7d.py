import time
import math

def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin

    word = generate_classic(n)
    target_count = math.factorial(n)

    # Factorials for Lehmer-code ranking
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i

    def perm_rank(p):
        rank = 0
        used = [False] * (n + 1)
        for i in range(n):
            count = 0
            for j in range(1, p[i]):
                if not used[j]:
                    count += 1
            rank += count * fact[n - 1 - i]
            used[p[i]] = True
        return rank

    def build_state(w):
        Lw = len(w)
        nr = Lw - n + 1
        rk = [0] * nr
        cov = [0] * target_count
        for i in range(nr):
            r = perm_rank(w[i:i + n])
            rk[i] = r
            cov[r] += 1
        return rk, cov

    ranks, cov = build_state(word)
    best_len = len(word)

    # Optimized try_deletion: Does not create new_w. 
    # Computes new ranks for affected windows directly from w.
    def try_deletion(w, i, rk, cov):
        Lw = len(w)
        nr = Lw - n + 1
        if i < 0 or i >= Lw:
            return None
        
        # Windows in new word are indexed 0 to nr-2.
        # A window j in new word:
        # If j < i: covers old indices j..j+n-1, but if it crosses i, we skip i.
        # If j >= i: covers old indices j+1..j+n.
        
        # Affected windows in new word:
        # Old windows that included i are those starting at k in [i-n+1, i].
        # In new word, these correspond to windows starting at j where:
        # If j < i: j corresponds to old j, but modified.
        # If j >= i: j corresponds to old j+1.
        
        # Let's identify the range of new windows that change.
        # A new window j is "different" from the simple shift if it includes the deleted element i.
        # The deleted element i was in old windows [i-n+1, i].
        # In new word:
        # - Windows j < i: The content is w[j..j+n-1] with w[i] removed IF j+n-1 >= i.
        #   So j must satisfy j <= i <= j+n-1 => i-n+1 <= j <= i.
        #   But j < i, so j in [max(0, i-n+1), i-1].
        # - Windows j >= i: The content is w[j+1..j+n]. This is just a shift.
        #   The old window at j+1 was w[j+1..j+n].
        #   So for j >= i, new_rk[j] = old_rk[j+1].
        
        # So the "complex" windows are j in [lo, hi] where:
        lo = max(0, i - (n - 1))
        hi = i - 1
        # Note: if i=0, lo=0, hi=-1 (empty range).
        
        # We need to check validity.
        # 1. Decrement cov for old ranks of windows that are "gone" or "changed".
        # The old windows that are affected are those that contained i.
        # These are old indices k in [max(0, i-n+1), i].
        # Let's just decrement cov for ALL old ranks in the range that maps to the new complex range?
        # Actually, it's safer to:
        # 1. Save old ranks of the specific old windows that are being modified/removed.
        # The old windows k in [max(0, i-n+1), i] are the ones that change.
        # Old k = i becomes new window i (shifted? No).
        # Let's trace:
        # Old w: ... w[i-1] w[i] w[i+1] ...
        # New w: ... w[i-1] w[i+1] ...
        
        # New window j:
        # If j < i: New window j is w[j..i-1] + w[i+1..j+n]. 
        #            This replaces Old window j (which was w[j..j+n-1]).
        #            Wait, if j+n-1 < i, it's identical.
        #            If j+n-1 >= i, it's different.
        #            Condition: j <= i <= j+n-1.
        #            Since j < i, this means i is in the window.
        #            So for j in [max(0, i-n+1), i-1], the window content changes.
        #            Old window j is removed from consideration (its rank decremented).
        #            New window j is added (its rank incremented).
            
        # If j >= i: New window j is w[j+1..j+n].
        #            This is identical to Old window j+1.
        #            So New window j "takes over" Old window j+1.
        #            Old window j+1 is NOT removed, it's just shifted index.
        #            However, Old window i is completely gone?
        #            Old window i is w[i..i+n-1].
        #            In new word, window i-1 is w[i-1..i-1] + w[i+1..i+n-1] (length n).
        #            Window i is w[i+1..i+n].
        #            Old window i is not present in new word.
            
        # So, the set of old ranks to decrement:
        # 1. Old windows j in [max(0, i-n+1), i-1] (these are modified).
        # 2. Old window i (this is removed).
        # Total old ranks to decrement: k in [max(0, i-n+1), i].
        
        old_lo = max(0, i - (n - 1))
        old_hi = i
        
        old_ranks = [rk[k] for k in range(old_lo, old_hi + 1)]
        for r in old_ranks:
            cov[r] -= 1
            if cov[r] < 0:
                # Revert
                for r2 in old_ranks:
                    cov[r2] += 1
                return None

        # Now compute new ranks for the new windows in [lo, hi] (complex ones).
        # New windows j in [lo, hi] where lo = max(0, i-n+1), hi = i-1.
        # If hi < lo, there are no complex windows (e.g., i=0).
        
        new_ranks_complex = []
        for j in range(lo, hi + 1):
            # Construct the virtual window
            # It consists of w[j..i-1] and w[i+1..j+n]
            # Length: (i - j) + (j + n - (i + 1)) = i - j + j + n - i - 1 = n - 1? 
            # Wait.
            # w[j..i-1] has length i - j.
            # w[i+1..j+n] has length (j+n) - (i+1) + 1 = j - i + n.
            # Total length: i - j + j - i + n = n. Correct.
            
            # Extract parts
            part1 = w[j:i]
            part2 = w[i+1:j+n+1] # slice end is exclusive, so j+n+1
            # Note: if i+1 > j+n+1, part2 is empty. This happens if j+n < i.
            # But j >= i-n+1 => j+n >= i+1. So part2 is not empty?
            # If j = i-n+1, j+n = i+1. part2 is w[i+1..i+2] (length 1).
            # If j = i-1, j+n = i+n-1. part2 is w[i+1..i+n] (length n-1).
            
            virtual_window = part1 + part2
            r = perm_rank(virtual_window)
            new_ranks_complex.append(r)
            cov[r] += 1

        # Check if any old rank became 0
        # We only need to check the ranks we decremented.
        for r in old_ranks:
            if cov[r] == 0:
                # Revert
                for r2 in new_ranks_complex:
                    cov[r2] -= 1
                for r3 in old_ranks:
                    cov[r3] += 1
                return None

        # Update rk array
        # New rk array length is nr - 1.
        new_rk = [0] * (nr - 1)
        
        # Copy unaffected parts
        # For j < lo: new_rk[j] = rk[j]
        for j in range(lo):
            new_rk[j] = rk[j]
            
        # For j in [lo, hi]: new_rk[j] = new_ranks_complex[j - lo]
        for idx, r in enumerate(new_ranks_complex):
            new_rk[lo + idx] = r
            
        # For j > hi: new_rk[j] = rk[j+1]
        # hi = i - 1. So for j >= i: new_rk[j] = rk[j+1]
        for j in range(i, nr - 1):
            new_rk[j] = rk[j + 1]

        # Commit
        # We must update the word itself for the next iteration
        new_w = w[:i] + w[i+1:]
        
        return new_w, new_rk, cov

    pass_no = 0
    while time.time() < deadline:
        pass_no += 1
        L = len(word)
        best_len = min(best_len, L)
        improved = False

        # Try single deletions
        for i in range(L):
            if time.time() > deadline:
                break
            res = try_deletion(word, i, ranks, cov)
            if res is not None:
                word, ranks, cov = res
                improved = True
                break

        if not improved:
            # Try 2-opt swaps? Or just break?
            # Given the time budget, single deletion is the primary driver.
            # If no deletion works, we are likely at a local minimum for 1-deletion.
            break

    return "".join(map(str, word))

def generate_classic(n):
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                          if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word
