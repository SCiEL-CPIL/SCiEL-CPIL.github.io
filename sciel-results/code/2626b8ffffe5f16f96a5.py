import time
import math

def construct(n):
    start_time = time.time()
    deadline = 142.0  # safety margin before 150s kill

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

    L = len(word)

    def build_state(w):
        Lw = len(w)
        nr = Lw - n + 1
        rk = [0] * nr
        cov = [0] * target_count
        fp = [-1] * target_count
        lp = [-1] * target_count
        for i in range(nr):
            r = perm_rank(w[i:i + n])
            rk[i] = r
            cov[r] += 1
            if fp[r] < 0:
                fp[r] = i
            lp[r] = i
        return rk, cov, fp, lp

    ranks, cov, fp, lp = build_state(word)
    best_len = L

    # Incremental single-deletion check.
    # Removing position i affects windows j in [i-(n-1), i] (clamped).
    # For each affected window we remove its old rank and add its new rank.
    # If any cov[r] hits 0 -> invalid.
    def try_deletion(w, i, rk, cov, fp, lp):
        Lw = len(w)
        nr = Lw - n + 1
        if i < 0 or i >= Lw:
            return None
        lo = max(0, i - (n - 1))
        hi = min(i, nr - 1)

        # Save old ranks of affected windows
        old_ranks = [rk[j] for j in range(lo, hi + 1)]

        # Compute new ranks for affected windows in the new word.
        # New word: w[:i] + w[i+1:]
        # New window j (in new word) for j in [lo, hi]:
        #   covers new positions j..j+n-1
        #   in old coords: j..i-1, i+1..j+n-1
        # We need to check cov never drops to 0.
        # First, decrement cov for old ranks of affected windows.
        for r in old_ranks:
            cov[r] -= 1
            if cov[r] < 0:
                # shouldn't happen
                pass

        # Now compute new ranks and increment.
        # Build the new word once (cheap enough, length ~13k)
        new_w = w[:i] + w[i + 1:]
        new_nr = nr - 1
        new_rk = [0] * new_nr
        # Unaffected windows keep their ranks (shifted index).
        # For j < lo: new_rk[j] = rk[j]
        # For j > hi: new_rk[j] = rk[j+1]
        # For lo <= j <= hi: recompute
        for j in range(lo):
            new_rk[j] = rk[j]
        for j in range(hi + 1, new_nr):
            new_rk[j] = rk[j + 1]
        for j in range(lo, hi + 1):
            new_rk[j] = perm_rank(new_w[j:j + n])

        # Increment cov for new ranks of affected windows
        for j in range(lo, hi + 1):
            r = new_rk[j]
            cov[r] += 1

        # Check validity: all cov >= 1
        # We only need to check ranks that were decremented (old ranks) since
        # increments can't cause a 0.
        for r in old_ranks:
            if cov[r] == 0:
                # revert
                for j in range(lo, hi + 1):
                    cov[new_rk[j]] -= 1
                for r2 in old_ranks:
                    cov[r2] += 1
                return None

        # Update fp/lp for affected ranks (approx: recompute for changed ranks)
        # For correctness of future checks we need fp/lp consistent.
        # Recompute fp/lp for the set of ranks that changed.
        changed = set(old_ranks)
        for j in range(lo, hi + 1):
            changed.add(new_rk[j])
        for r in changed:
            # find new first/last
            new_fp = -1
            new_lp = -1
            # scan new_rk for r (only in affected region matters, but r may appear elsewhere)
            # To be safe and cheap, scan whole new_rk
            for j in range(new_nr):
                if new_rk[j] == r:
                    if new_fp < 0:
                        new_fp = j
                    new_lp = j
            fp[r] = new_fp
            lp[r] = new_lp

        # Commit
        word = new_w
        ranks = new_rk
        return word, ranks, cov, fp, lp

    # Incremental block-removal check.
    # Remove positions [a, a+b-1], b = 2n-1.
    # New word length Lw-b, new window count nr-b.
    # New window j:
    #   if j + n - 1 < a: old j
    #   if j >= a: old j + b
    #   else (overlap): new positions j..j+n-1 in new_w
    # Affected j: a-(n-1) .. a+b-1  (clamped)
    def try_block_removal(w, a, rk, cov, fp, lp):
        Lw = len(w)
        b = 2 * n - 1
        if a + b > Lw:
            return None
        nr = Lw - n + 1
        lo = max(0, a - (n - 1))
        hi = min(a + b - 1, nr - 1)

        old_ranks = [rk[j] for j in range(lo, hi + 1)]
        for r in old_ranks:
            cov[r] -= 1

        new_w = w[:a] + w[a + b:]
        new_nr = len(new_w) - n + 1
        new_rk = [0] * new_nr
        for j in range(lo):
            new_rk[j] = rk[j]
        for j in range(hi + 1, new_nr):
            new_rk[j] = rk[j + b]
        for j in range(lo, hi + 1):
            new_rk[j] = perm_rank(new_w[j:j + n])

        for j in range(lo, hi + 1):
            cov[new_rk[j]] += 1

        for r in old_ranks:
            if cov[r] == 0:
                for j in range(lo, hi + 1):
                    cov[new_rk[j]] -= 1
                for r2 in old_ranks:
                    cov[r2] += 1
                return None

        changed = set(old_ranks)
        for j in range(lo, hi + 1):
            changed.add(new_rk[j])
        for r in changed:
            new_fp = -1
            new_lp = -1
            for j in range(new_nr):
                if new_rk[j] == r:
                    if new_fp < 0:
                        new_fp = j
                    new_lp = j
            fp[r] = new_fp
            lp[r] = new_lp

        return new_w, new_rk, cov, fp, lp

    pass_no = 0
    while time.time() < deadline:
        pass_no += 1
        L = len(word)
        best_len = min(best_len, L)
        improved = False

        if pass_no % 2 == 1:
            b = 2 * n - 1
            for a in range(L - b + 1):
                if time.time() > deadline:
                    break
                P1 = word[a:a + n - 1]
                m = word[a + n - 1]
                P2 = word[a + n:a + 2 * n - 1]
                if P1 != P2 or m in P1:
                    continue
                res = try_block_removal(word, a, ranks, cov, fp, lp)
                if res is not None:
                    word, ranks, cov, fp, lp = res
                    improved = True
                    break
        else:
            for i in range(L):
                if time.time() > deadline:
                    break
                res = try_deletion(word, i, ranks, cov, fp, lp)
                if res is not None:
                    word, ranks, cov, fp, lp = res
                    improved = True
                    break

        if not improved:
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
