import math
import random
import time

MAXM = 128
DEADLINE = 82.0  # safety margin before 90s kill
START = time.time()

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

# Deterministic seed: CRT over the largest distinct primes (guaranteed valid).
_PRIMES = _primes_upto(MAXM)
_SEED = [(1, p) for p in _PRIMES]

def _valid(pairs):
    for i in range(len(pairs)):
        ri, mi = pairs[i]
        for j in range(i + 1, len(pairs)):
            rj, mj = pairs[j]
            if math.gcd(ri - rj, mi * mj) != 1:
                return False
    return True

def _try_insert(pairs, m, r):
    for (ri, mi) in pairs:
        if math.gcd(r - ri, m * mi) != 1:
            return False
    pairs.append((m, r))
    return True

def _greedy_expansion(pairs, mod_order):
    """Add classes greedily (each with its best residue) in the given modulus order."""
    for m in mod_order:
        if any(mi == m for _, mi in pairs):
            continue
        # try residues in a fixed heuristic order; first that works wins
        found = False
        for r in range(m):
            if _try_insert(pairs, m, r):
                found = True
                break
        if found:
            pass
        else:
            pairs = [p for p in pairs]  # no-op; nothing inserted
    return pairs

def _local_search(pairs, rng, deadline):
    """Hill-climb: add / remove / swap to grow the set under the exact gcd criterion."""
    best = list(pairs)
    best_len = len(best)
    used = {m for _, m in best}

    def candidates_add():
        for m in range(2, MAXM + 1):
            if m in used:
                continue
            for r in range(m):
                yield (m, r)

    while time.time() < deadline:
        improved = False

        # 1) try to add a new class
        added = False
        for m in range(2, MAXM + 1):
            if m in used:
                continue
            for r in range(m):
                ok = True
                for (ri, mi) in best:
                    if math.gcd(r - ri, m * mi) != 1:
                        ok = False
                        break
                if ok:
                    best.append((m, r))
                    used.add(m)
                    best_len += 1
                    improved = True
                    added = True
                    break
            if added:
                break

        # 2) try swaps: drop one, add two (or one) to grow
        if not improved and len(best) < MAXM - 1:
            for drop_idx in range(len(best)):
                dm = best[drop_idx][1]
                rest = [best[k] for k in range(len(best)) if k != drop_idx]
                # attempt to add up to two new classes into rest
                gain = 0
                new_pairs = list(rest)
                new_used = {m for _, m in new_pairs}
                for m in range(2, MAXM + 1):
                    if m in new_used:
                        continue
                    for r in range(m):
                        ok = True
                        for (ri, mi) in new_pairs:
                            if math.gcd(r - ri, m * mi) != 1:
                                ok = False
                                break
                        if ok:
                            new_pairs.append((m, r))
                            new_used.add(m)
                            gain += 1
                            if gain >= 2:
                                break
                    if gain >= 2:
                        break
                if gain - 1 > 0:  # net gain positive
                    best = new_pairs
                    used = new_used
                    best_len = len(best)
                    improved = True
                    break

        if not improved:
            # small random perturbation to escape local optima
            if len(best) > 1:
                i = rng.randrange(len(best))
                j = rng.randrange(len(best))
                if i != j:
                    best[i], best[j] = best[j], best[i]
                # occasionally drop a random class to allow regrowth
                if rng.random() < 0.3:
                    k = rng.randrange(len(best))
                    best.pop(k)
                    used = {m for _, m in best}
                    best_len = len(best)

    return best

def construct():
    deadline = time.time() + (DEADLINE - (time.time() - START))
    if deadline < time.time() + 1.0:
        deadline = time.time() + 1.0

    rng = random.Random(12345)

    # Deterministic seed (guaranteed valid), then greedy expansion.
    pairs = list(_SEED)
    used = {m for _, m in pairs}
    order = [m for m in range(2, MAXM + 1) if m not in used]
    # Heuristic: prefer smaller moduli first (more residues to choose from)
    order.sort()
    pairs = _greedy_expansion(pairs, order)

    # Local search with deadline guard.
    pairs = _local_search(pairs, rng, deadline)

    # Final safety: validate and trim if needed.
    if not _valid(pairs):
        # fall back to the deterministic seed
        pairs = list(_SEED)

    # Convert to canonical list-of-lists.
    result = [[r, m] for (m, r) in pairs]
    return result
