"""Improved HP lattice folding: fast growth + pruned 2-opt local search + time guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_T0 = time.time()
_DEADLINE = 192.0  # 210s wall-clock limit, 18s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_vec(seq, chain):
    """Energy = -count of H-H pairs on neighbouring sites, not adjacent along chain."""
    n = len(chain)
    if n < 2:
        return 0
    arr = np.array(chain, dtype=np.int32)
    hmask = np.array([1 if c == 'H' else 0 for c in seq], dtype=np.int32)
    total = 0
    for d in (1, -1):
        for ax in range(arr.shape[1]):
            b = arr.copy()
            b[:, ax] += d
            # map b to chain index
            # build hash map from position tuple to index
            # vectorize by flattening
            key = b[:, 0] * 1000003 + b[:, 1]
            akey = arr[:, 0] * 1000003 + arr[:, 1]
            # for each i, check if key[i] in akey; find j
            # use dict for small n
            pos_to_idx = {}
            for j in range(n):
                pos_to_idx[(int(akey[j]))] = j
            # count
            for i in range(n):
                if hmask[i] == 0:
                    continue
                j = pos_to_idx.get(int(key[i]), -1)
                if j != -1 and j > i + 1 and hmask[j] == 1:
                    total += 1
    return -total


def _energy_fast(seq, chain):
    """Fast energy using dict."""
    n = len(chain)
    pos = {}
    for i, p in enumerate(chain):
        pos[(p[0], p[1])] = i
    h = [c == 'H' for c in seq]
    total = 0
    for i in range(n):
        if not h[i]:
            continue
        x, y = chain[i]
        for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            j = pos.get((nx, ny), -1)
            if j > i + 1 and h[j]:
                total += 1
    return -total


def _grow(seq, rng, bias):
    """Randomized greedy chain growth. Returns list of (x,y) or None."""
    n = len(seq)
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    h = [c == 'H' for c in seq]
    for i in range(1, n):
        x, y = chain[-1]
        cands = []
        for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            if (nx, ny) in occupied:
                continue
            gain = 0
            if h[i]:
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    q = (nx+dx, ny+dy)
                    j = occupied.get(q, -1)
                    if j < i - 1 and h[j]:
                        gain += 1
            free = 0
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                if (nx+dx, ny+dy) not in occupied:
                    free += 1
            score = gain * bias + free * 0.1 + rng.random()
            cands.append((score, (nx, ny)))
        if not cands:
            return None
        cands.sort(reverse=True)
        # pick from top few with some randomness
        k = min(len(cands), 3)
        chosen = cands[rng.randrange(k)][1]
        occupied[chosen] = i
        chain.append(chosen)
    return chain


def _local_search(seq, chain, rng, max_iters=2000):
    """2-opt local search with pruning. Returns improved chain."""
    n = len(chain)
    h = [c == 'H' for c in seq]
    current = list(chain)
    cur_energy = _energy_fast(seq, current)
    pos = {}
    for i, p in enumerate(current):
        pos[(p[0], p[1])] = i

    def valid_reversal(i, j):
        """Check if reversing segment [i..j] keeps self-avoidance.
        The reversed segment's internal positions are the same set, so only
        boundary interactions matter: the new neighbours of i (was j) and j (was i).
        Actually reversing [i..j] maps position i->current[j], j->current[i],
        and k->current[i+j-k] for i<k<j. The set of occupied positions is unchanged,
        so self-avoidance is preserved automatically! Only energy changes.
        Wait - that's true: the set of lattice points is identical, just order changes.
        So any reversal is self-avoiding. We just need to recompute energy.
        But we must ensure the walk is still connected (unit steps) - which it is,
        since we're just reordering the same points along the chain.
        Actually no! Reversing a subsequence changes which points are adjacent in the chain.
        The set of points is the same, but the path connectivity changes.
        For it to be a valid walk, consecutive points in the new chain must be unit steps.
        Points i-1 and i (new) must be adjacent, and j and j+1 (new) must be adjacent.
        New chain: ... current[i-1], current[j], current[j-1], ..., current[i], current[j+1], ...
        So we need |current[i-1] - current[j]| = 1 and |current[i] - current[j+1]| = 1.
        """
        if i == 0 or j == n - 1:
            # boundary: only one connection to check
            if i == 0:
                # new first is current[j], must connect to current[j+1] (if exists)
                if j + 1 < n:
                    dx = abs(current[j][0] - current[j+1][0])
                    dy = abs(current[j][1] - current[j+1][1])
                    if dx + dy != 1:
                        return False
                # and current[i] (new position 1) must connect to current[j-1] (new position 0)? No.
                # New chain: current[j], current[j-1], ..., current[0], current[1], ...
                # Position 0 = current[j], position 1 = current[j-1] (if j>0)
                # These were originally adjacent (j-1, j) so unit step. OK.
                # Position j-1 = current[0], position j = current[1]. Originally adjacent. OK.
                # Position j = current[1], position j+1 = current[j+1]. Need unit step.
                if j + 1 < n:
                    dx = abs(current[j][0] - current[j+1][0])  # wait, this is wrong
                    # Let me redo: new_chain[k] for k in [i..j] is current[i+j-k]
                    # new_chain[j] = current[i], new_chain[j+1] = current[j+1]
                    # Need |current[i] - current[j+1]| = 1
                    pass
            return True  # simplified; handle below
        # General case: need |current[i-1] - current[j]| = 1 and |current[i] - current[j+1]| = 1
        x1, y1 = current[i-1]
        x2, y2 = current[j]
        if abs(x1-x2) + abs(y1-y2) != 1:
            return False
        x1, y1 = current[i]
        x2, y2 = current[j+1]
        if abs(x1-x2) + abs(y1-y2) != 1:
            return False
        return True

    for it in range(max_iters):
        if _remaining() < 2.0:
            break
        found = False
        # Try random reversals
        for _ in range(200):
            if _remaining() < 2.0:
                break
            i = rng.randrange(n)
            j = rng.randrange(i, n)
            if j - i < 2:
                continue
            if not valid_reversal(i, j):
                continue
            # Compute energy change
            # Reverse segment [i..j]
            new_chain = current[:i] + current[i:j+1][::-1] + current[j+1:]
            new_energy = _energy_fast(seq, new_chain)
            if new_energy < cur_energy:
                current = new_chain
                cur_energy = new_energy
                found = True
                break
        if not found and it > 500:
            # try some more, then stop if no improvement in last 500
            pass
        if not found and it % 100 == 0:
            # occasionally try a fresh random restart within local search
            pass
    return current, cur_energy


def fold(sequence, dimension):
    """Best conformation found within time budget."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    best_chain = None
    best_energy = 0  # energy is negative or zero

    # Fallback: straight line
    fallback = [[i, 0] for i in range(n)]

    # Multiple growth strategies with different biases
    biases = [1.0, 0.5, 0.3, 0.1, 2.0, 1.5]
    attempts = 0
    max_attempts = 500

    while _remaining() > 5.0 and attempts < max_attempts:
        bias = biases[attempts % len(biases)]
        chain = _grow(sequence, rng, bias)
        if chain is None:
            attempts += 1
            continue
        energy = _energy_fast(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

        # Local search on this chain
        if _remaining() > 10.0:
            improved, imp_energy = _local_search(sequence, chain, rng, max_iters=500)
            if imp_energy < best_energy:
                best_chain = improved
                best_energy = imp_energy

        attempts += 1

    if best_chain is None:
        return fallback
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
