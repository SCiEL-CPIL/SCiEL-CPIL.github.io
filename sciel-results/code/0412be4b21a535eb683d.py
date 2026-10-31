import time
import random
import numpy as np

# EVOLVE-BLOCK-START

# Precomputed tables (v <= 19 assumed; 2^19 = 524288)
MAXV = 19
SIZE = 1 << MAXV
RANK = np.zeros(SIZE, dtype=np.int32)
for _i in range(1, SIZE):
    RANK[_i] = RANK[_i & (_i - 1)] + 1

# For each mask, a list of its 1-bit positions (for vectorized block scoring)
BITS = [None] * SIZE
for _i in range(SIZE):
    _b = []
    _x = _i
    while _x:
        _x &= _x - 1
        _b.append(_x.bit_length() - 1)
    BITS[_i] = _b

# Precomputed (k-1)-subset masks for each mask (for fast replacement search)
KSUB = [None] * SIZE
for _i in range(SIZE):
    _m = []
    _r = RANK[_i]
    for _x in BITS[_i]:
        _m.append(_i ^ (1 << _x))
    KSUB[_i] = _m


def _zeta_superset(f, v):
    """f[mask] = sum over supermasks s of mask of f[s]. In-place."""
    f = f.copy()
    for b in range(v):
        bit = 1 << b
        for mask in range(SIZE):
            if (mask & bit) == 0:
                f[mask] += f[mask | bit]
    return f


def _zeta_subset(f, v):
    """f[mask] = sum over submasks s of mask of f[s]. In-place."""
    f = f.copy()
    for b in range(v):
        bit = 1 << b
        for mask in range(SIZE):
            if (mask & bit) != 0:
                f[mask] += f[mask ^ bit]
    return f


def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 112.0  # safety margin under 120s
    rng = random.Random(20240517)

    all_masks = np.array([m for m in range(SIZE) if RANK[m] == k], dtype=np.int32)
    if len(all_masks) == 0:
        return []
    ksub = KSUB
    bits = BITS
    rank = RANK

    def coverage(blocks):
        cov = np.zeros(SIZE, dtype=np.int64)
        for b in blocks:
            cov[b] += 1
        z = _zeta_superset(cov, v)
        return z

    def uncovered(z):
        return int(np.sum(z < 1))

    # ---- Phase 1: greedy construction (vectorized scoring) ----
    z = np.zeros(SIZE, dtype=np.int64)
    blocks = []
    while True:
        if time.time() > DEADLINE:
            break
        un = z < 1
        if not un.any():
            break
        # score each k-block by number of currently-uncovered t-subsets it covers
        sc = np.zeros(len(all_masks), dtype=np.int64)
        for i, b in enumerate(all_masks):
            sc[i] = np.sum(un[ksub[b]])
        j = int(np.argmax(sc))
        b = int(all_masks[j])
        blocks.append(b)
        z[b] += 1
        z = _zeta_superset(z, v)

    if not blocks:
        # fallback: cannot have happened for k>=t, but be safe
        return []

    # ---- Phase 2: local search ----
    # current coverage vector z (superset), current block list
    def try_replacements(blocks, z, iters):
        """Try to replace blocks to reduce uncovered count."""
        improved = True
        while improved and iters > 0 and time.time() < DEADLINE:
            improved = False
            iters -= 1
            # find block with fewest unique (only-this-block) t-subsets -> good candidate to replace
            # unique count per block = number of t-subsets covered only by it
            # Compute per-block unique via: for each block, count t-subsets where z==1 and block covers them
            # cheaper: pick a random block and try to find a better replacement
            idx = rng.randrange(len(blocks))
            b = blocks[idx]
            # current unique contribution of b
            z[b] -= 1
            z = _zeta_superset(z, v)
            cur_un = int(np.sum(z < 1))
            # find best replacement block
            sc = np.zeros(len(all_masks), dtype=np.int64)
            un = z < 1
            for i, nb in enumerate(all_masks):
                if nb == b:
                    continue
                sc[i] = np.sum(un[ksub[nb]])
            j = int(np.argmax(sc))
            nb = int(all_masks[j])
            z[nb] += 1
            z = _zeta_superset(z, v)
            new_un = int(np.sum(z < 1))
            if new_un < cur_un:
                blocks[idx] = nb
                improved = True
            else:
                # revert
                z[nb] -= 1
                z = _zeta_superset(z, v)
                z[b] += 1
                z = _zeta_superset(z, v)
        return blocks, z

    def try_drops(blocks, z, iters):
        """Try to drop redundant blocks."""
        improved = True
        while improved and iters > 0 and time.time() < DEADLINE:
            improved = False
            iters -= 1
            idx = rng.randrange(len(blocks))
            b = blocks[idx]
            z[b] -= 1
            z = _zeta_superset(z, v)
            cur_un = int(np.sum(z < 1))
            if cur_un == 0:
                # drop it
                blocks.pop(idx)
                improved = True
            else:
                z[b] += 1
                z = _zeta_superset(z, v)
        return blocks, z

    best_blocks = list(blocks)
    best_z = z.copy()
    best_un = uncovered(best_z)

    # annealing loop
    T = 3.0
    while time.time() < DEADLINE:
        # random restart / perturb: add a few random blocks
        pert = list(blocks)
        zc = z.copy()
        for _ in range(rng.randint(0, 3)):
            nb = int(all_masks[rng.randrange(len(all_masks))])
            pert.append(nb)
            zc[nb] += 1
        zc = _zeta_superset(zc, v)
        # local search from perturbed state
        pert, zc = try_replacements(pert, zc, iters=200)
        pert, zc = try_drops(pert, zc, iters=200)
        pert, zc = try_replacements(pert, zc, iters=200)
        un = uncovered(zc)
        if un < best_un:
            best_un = un
            best_blocks = list(pert)
            best_z = zc.copy()
            blocks = list(pert)
            z = zc.copy()
            T = 3.0
        else:
            # keep current (no regression in accepted state)
            T *= 0.95

    # final: remove duplicates
    seen = set()
    out = []
    for b in best_blocks:
        if b not in seen:
            seen.add(b)
            out.append(b)

    # convert masks to sorted point lists
    result = []
    for b in out:
        result.append(sorted(bits[b]))
    return result

# EVOLVE-BLOCK-END
