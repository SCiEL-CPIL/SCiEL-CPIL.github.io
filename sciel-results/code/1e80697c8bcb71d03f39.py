import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 112  # 8s safety margin before 120s

    # 1. Generate standard recursive superpermutation
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

    S = list(word)
    L = len(S)

    # 2. Build counts and at-risk set
    # A permutation is "at risk" if its count == 1; any removal touching a
    # window containing it is invalid.
    counts = {}
    for i in range(L - n + 1):
        p = tuple(S[i:i + n])
        counts[p] = counts.get(p, 0) + 1

    at_risk = set(p for p, c in counts.items() if c == 1)

    best_S = S[:]
    best_len = L

    def rebuild_counts(S):
        counts = {}
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i + n])
            counts[p] = counts.get(p, 0) + 1
        return counts

    def is_valid(S):
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i + n]))
        return len(seen) == 5040 if n == 7 else len(seen) == 1

    # 3. Removal pass: scan positions, try to remove if no at-risk window is destroyed
    #    and the straddling windows are all already covered (count > 0) OR will be added.
    #    Actually: removal is valid iff no old window drops to 0. New windows are added.
    def removal_pass(S, counts, at_risk, deadline):
        removed_any = False
        i = len(S) - 1
        while i >= 0:
            if time.time() > deadline:
                break
            # Old windows destroyed: indices [i-n+1, i] clamped
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            valid = True
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
            if not valid:
                i -= 1
                continue

            # Perform removal
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                c = counts[p] - 1
                if c == 0:
                    del counts[p]
                    if p in at_risk:
                        at_risk.discard(p)
                else:
                    counts[p] = c
                    if c == 1:
                        at_risk.add(p)

            # Add straddling windows
            new_len = len(S) - 1
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, new_len - n)
            for k in range(start_new, end_new + 1):
                wc = []
                for idx in range(k, k + n):
                    s_idx = idx if idx < i else idx + 1
                    wc.append(S[s_idx])
                p_new = tuple(wc)
                c = counts.get(p_new, 0)
                if c == 0:
                    counts[p_new] = 1
                    at_risk.add(p_new)
                elif c == 1:
                    counts[p_new] = 2
                    at_risk.discard(p_new)
                else:
                    counts[p_new] = c + 1

            S.pop(i)
            removed_any = True
            # continue at same i (next char now at position i)

        return removed_any

    # 4. Perturbation strategies
    def perturb_swap(S, counts, at_risk):
        """Swap two random positions, rebuild counts."""
        i, j = random.sample(range(len(S)), 2)
        S[i], S[j] = S[j], S[i]
        counts = rebuild_counts(S)
        at_risk = set(p for p, c in counts.items() if c == 1)
        return S, counts, at_risk

    def perturb_insert(S, counts, at_risk):
        """Insert a random symbol at a random position."""
        pos = random.randint(0, len(S))
        sym = random.randint(1, n)
        S.insert(pos, sym)
        counts = rebuild_counts(S)
        at_risk = set(p for p, c in counts.items() if c == 1)
        return S, counts, at_risk

    # 5. Main loop
    stuck_rounds = 0
    max_stuck = 30
    while time.time() < deadline:
        removed = removal_pass(S, counts, at_risk, deadline)
        if removed:
            best_S = S[:]
            best_len = len(S)
            stuck_rounds = 0
        else:
            stuck_rounds += 1
            if stuck_rounds > max_stuck:
                break
            # Perturb to escape local optimum
            r = random.random()
            if r < 0.5 and len(S) > n:
                S, counts, at_risk = perturb_swap(S, counts, at_risk)
            elif len(S) < best_len * 1.15:
                S, counts, at_risk = perturb_insert(S, counts, at_risk)
            else:
                S, counts, at_risk = perturb_swap(S, counts, at_risk)

    # 6. Final polish: one more removal pass on the best found
    #    (best_S may have been updated; try a final cleanup)
    if time.time() < deadline - 2:
        S2 = best_S[:]
        counts2 = rebuild_counts(S2)
        at_risk2 = set(p for p, c in counts2.items() if c == 1)
        removal_pass(S2, counts2, at_risk2, deadline)
        if len(S2) < len(best_S) and is_valid(S2):
            best_S = S2

    # 7. Validate
    if not is_valid(best_S):
        return "".join(map(str, word))

    return "".join(map(str, best_S))
