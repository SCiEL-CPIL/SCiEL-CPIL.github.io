import math
import random
import time

MAX_MOD = 128
DEADLINE_FRAC = 0.80  # safety margin vs 90s wall-clock

# ---------- number theory ----------
def factorize(n):
    """Return sorted list of distinct prime factors of n."""
    fs = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            fs.append(d)
            while n % d == 0:
                n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        fs.append(n)
    return fs

def gcd(a, b):
    return math.gcd(a, b)

# ---------- disjointness ----------
def is_disjoint(r1, m1, r2, m2):
    """Exact gcd/CRT criterion: classes are disjoint iff the CRT system
    x ≡ r1 (mod m1), x ≡ r2 (mod m2) is inconsistent.
    Consistent iff gcd(m1,m2) | (r2 - r1). Disjoint iff NOT that."""
    g = gcd(m1, m2)
    if g == 1:
        # always consistent -> NOT disjoint
        return False
    return ((r2 - r1) % g) != 0

def all_disjoint(pairs):
    n = len(pairs)
    for i in range(n):
        r1, m1 = pairs[i]
        for j in range(i + 1, n):
            r2, m2 = pairs[j]
            if not is_disjoint(r1, m1, r2, m2):
                return False
    return True

# ---------- candidate moduli & prime supports ----------
# Build the set of moduli 2..128 with their prime-support frozensets.
MODS = list(range(2, MAX_MOD + 1))
SUPPORT = {m: frozenset(factorize(m)) for m in MODS}

# ---------- validity of a full assignment ----------
# pairs: dict modulus -> residue (canonical 0<=r<m)
def valid(pairs):
    items = list(pairs.items())
    for i in range(len(items)):
        r1, m1 = items[i]
        for j in range(i + 1, len(items)):
            r2, m2 = items[j]
            if not is_disjoint(r1, m1, r2, m2):
                return False
    return True

def count_valid(pairs):
    return len(pairs) if valid(pairs) else -1

# ---------- greedy most-constrained-first ----------
def greedy(seed_rng):
    """Build a maximal (not necessarily maximum) assignment greedily.
    Order moduli by how 'hard' they are: fewer residue choices that stay
    disjoint from already-placed moduli. We approximate difficulty by
    the size of the prime-support (more shared primes => more constraints)."""
    pairs = {}
    # Precompute, for each modulus, the number of residue classes r mod m
    # that are disjoint from a *given* placed modulus. We'll just try all
    # residues in a random order and keep the first that works.
    order = sorted(MODS, key=lambda m: (-len(SUPPORT[m]), m))
    for m in order:
        # try residues in random order
        residues = list(range(m))
        seed_rng.shuffle(residues)
        placed = False
        for r in residues:
            ok = True
            for m2, r2 in pairs.items():
                if not is_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                pairs[m] = r
                placed = True
                break
        # if not placed, skip (greedy leaves a gap; local search may fill)
    return pairs

# ---------- local search: direct add + swap ----------
def local_search(pairs, seed_rng, deadline):
    """Improve a valid assignment. Returns (pairs, improved)."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # --- direct add: try adding each missing modulus with some residue ---
        missing = [m for m in MODS if m not in pairs]
        seed_rng.shuffle(missing)
        added = False
        for m in missing:
            if time.time() >= deadline:
                break
            residues = list(range(m))
            seed_rng.shuffle(residues)
            for r in residues:
                ok = True
                for m2, r2 in pairs.items():
                    if not is_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    pairs[m] = r
                    improved = True
                    added = True
                    break
            if added:
                # restart add phase to catch further additions
                break
        if added:
            continue
        # --- swap: change an existing residue to free up a missing modulus ---
        # Try: for each existing modulus m1, try a new residue r1' that keeps
        # all current disjointness, then see if some missing modulus becomes addable.
        # This is expensive; do a bounded random sample.
        items = list(pairs.items())
        seed_rng.shuffle(items)
        for m1, r1 in items[:40]:
            if time.time() >= deadline:
                break
            cand = list(range(m1))
            seed_rng.shuffle(cand)
            for r1_new in cand[:20]:
                if r1_new == r1:
                    continue
                # check m1 still disjoint from all others
                ok = True
                for m2, r2 in pairs.items():
                    if m2 == m1:
                        continue
                    if not is_disjoint(r1_new, m1, r2, m2):
                        ok = False
                        break
                if not ok:
                    continue
                # now try to add a missing modulus
                missing = [m for m in MODS if m not in pairs]
                seed_rng.shuffle(missing)
                for m in missing[:12]:
                    residues = list(range(m))
                    seed_rng.shuffle(residues)
                    for r in residues[:15]:
                        ok2 = True
                        for m2, r2 in pairs.items():
                            if m2 == m1:
                                continue
                            if not is_disjoint(r, m, r2, m2):
                                ok2 = False
                                break
                        if ok2 and is_disjoint(r, m, r1_new, m1):
                            pairs[m1] = r1_new
                            pairs[m] = r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return pairs, improved

# ---------- main ----------
def construct():
    t0 = time.time()
    deadline = t0 + (90.0 * DEADLINE_FRAC)  # ~72s safety margin
    seed = 12345
    best = None
    best_size = 0

    # Run several greedy seeds, then local search on each.
    for trial in range(60):
        if time.time() >= deadline:
            break
        rng = random.Random(seed + trial * 7919)
        pairs = greedy(rng)
        # local search
        pairs, _ = local_search(pairs, rng, deadline)
        if valid(pairs):
            sz = len(pairs)
            if sz > best_size:
                best_size = sz
                best = dict(pairs)
        # early exit if we already have a very large solution
        if best_size >= 40:
            # keep trying a bit more but stop soon
            if best_size >= 45:
                break

    if best is None:
        # Fallback: trivially valid single class
        return [[0, 2]]

    out = [[r, m] for m, r in best.items()]
    # canonical form 0<=r<m already holds by construction
    return out
