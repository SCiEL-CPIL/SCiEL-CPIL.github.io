"""
Rewritten: stochastic local search (simulated annealing) over comparator-network
representations using a violation-count fitness landscape.

Key structural changes vs. the previous program:
- Fitness is the number of unsorted zero-one inputs (not a binary validity check).
- Moves include: delete, insert, swap-two, and a "shuffle" that reorders a random
  contiguous block (helps escape permutation local minima).
- Simulated annealing schedule: temperature decays geometrically, allowing
  non-improving moves early and increasingly greedy moves late.
- Starts from a known-good Batcher network but immediately applies a fast
  greedy-deletion pass to get a small valid seed, then runs SA to search for
  even smaller networks.
"""

import time
import numpy as np

N = 16
MAX_TIME = 58.0  # safety margin within 90 s wall-clock

# ---------- fast validity / fitness ----------

def _build_bits():
    """Return a (2^N, N) uint8 array of all Boolean inputs."""
    n = 1 << N
    idx = np.arange(n, dtype=np.uint32)
    bits = ((idx[:, None] >> np.arange(N, dtype=np.uint32)[::-1]) & 1).astype(np.uint8)
    return bits

def _apply(comps, bits):
    """Apply network to all 2^N inputs; return (2^N, N) uint8 output."""
    vec = bits.copy()
    for i, j in comps:
        a = vec[:, i].copy()
        b = vec[:, j].copy()
        m = np.minimum(a, b)
        x = np.maximum(a, b)
        vec[:, i] = m
        vec[:, j] = x
    return vec

def _violation_count(comps, bits):
    """Count of zero-one inputs that are NOT sorted after applying the network.
       Returns 0 if the network is valid."""
    out = _apply(comps, bits)
    # sorted means each row is non-decreasing
    diffs = np.diff(out, axis=1)
    bad_rows = np.any(diffs < 0, axis=1)
    return int(np.count_nonzero(bad_rows))

# ---------- initial network ----------

def _batcher_network():
    """Batcher's odd-even merge sort for 16 wires -> list of (i,j)."""
    def merge(n, off):
        if n == 1:
            return [(off, off + 1)]
        h = n // 2
        def smerge(n, off, stride):
            if n == 1:
                return [(off, off + stride)]
            h2 = n // 2
            step1 = [(off + k * stride, off + (n + k) * stride) for k in range(n)]
            ev = smerge(h2, off, 2 * stride)
            od = smerge(h2, off + stride, 2 * stride)
            return step1 + ev + od
        step1 = [(off + k, off + k + n) for k in range(n)]
        ev = smerge(h, off, 2)
        od = smerge(h, off + 1, 2)
        return step1 + ev + od

    def sort(n, off):
        if n <= 1:
            return []
        h = n // 2
        L = sort(h, off)
        R = sort(h, off + h)
        M = merge(h, off)
        return L + R + M

    return sort(N, 0)

def _greedy_delete(comps, bits, deadline):
    """Repeatedly try to delete each comparator; keep first valid removal."""
    comps = list(comps)
    while time.time() < deadline:
        removed = False
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                return comps
            cand = comps[:idx] + comps[idx+1:]
            if _violation_count(cand, bits) == 0:
                comps = cand
                removed = True
                break
        if not removed:
            break
    return comps

# ---------- SA moves ----------

def _all_comps():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _sa_step(comps, rng, all_comp_list):
    """Pick a random move. Returns new_comps, move_type.
       move_type: 'del', 'ins', 'swap', 'shuffle'
       del/ins change length; swap/shuffle preserve length."""
    r = rng.random()
    if r < 0.35 and len(comps) > 1:
        idx = rng.integers(0, len(comps))
        new = comps[:idx] + comps[idx+1:]
        return new, 'del'
    elif r < 0.65:
        nc = all_comp_list[rng.integers(0, len(all_comp_list))]
        pos = rng.integers(0, len(comps) + 1)
        new = comps[:pos] + [nc] + comps[pos:]
        return new, 'ins'
    elif r < 0.85 and len(comps) >= 2:
        i = rng.integers(0, len(comps))
        j = rng.integers(0, len(comps))
        if i == j:
            j = (j + 1) % len(comps)
        new = comps[:]
        new[i], new[j] = new[j], new[i]
        return new, 'swap'
    else:
        # shuffle a contiguous block of length 2-6
        L = len(comps)
        if L < 3:
            return comps, 'shuffle'
        blen = min(rng.integers(2, 7), L)
        start = rng.integers(0, L - blen + 1)
        new = comps[:]
        block = new[start:start+blen]
        rng.shuffle(block)
        new[start:start+blen] = block
        return new, 'shuffle'

# ---------- main ----------

def construct():
    t0 = time.time()
    deadline = t0 + MAX_TIME

    bits = _build_bits()
    all_comp_list = _all_comps()

    # 1. Build initial valid network and greedy-delete to get a small seed
    init = _batcher_network()
    seed = _greedy_delete(init, bits, deadline)
    seed_v = _violation_count(seed, bits)
    assert seed_v == 0, "Seed network must be valid"

    best_comps = list(seed)
    best_v = 0
    best_len = len(best_comps)

    # 2. Simulated annealing
    rng = np.random.default_rng(12345)
    T = 500.0          # initial temperature
    T_min = 1.0
    cool_rate = 0.9995 # per-step cooling

    cur_comps = list(seed)
    cur_v = 0
    steps = 0

    while time.time() < deadline - 0.5:
        if T < T_min:
            # Switch to pure greedy hill-climbing
            T = T_min

        new_comps, mtype = _sa_step(cur_comps, rng, all_comp_list)
        new_v = _violation_count(new_comps, bits)

        delta_v = new_v - cur_v
        steps += 1

        # Acceptance:
        # - If new is valid (v=0) and shorter -> always accept as best
        # - Standard SA acceptance for moves that don't change validity status
        if new_v == 0 and len(new_comps) < best_len:
            cur_comps = new_comps
            cur_v = 0
            best_comps = list(new_comps)
            best_len = len(best_comps)
            best_v = 0
            continue

        if cur_v == 0 and new_v == 0:
            # Both valid: prefer shorter
            if len(new_comps) < len(cur_comps):
                cur_comps = new_comps
                cur_v = 0
            elif rng.random() < 0.1:
                # Occasionally accept same-length valid move to explore
                cur_comps = new_comps
            continue

        if new_v == 0 and cur_v > 0:
            # Found valid from invalid state – great
            cur_comps = new_comps
            cur_v = 0
            if len(cur_comps) < best_len:
                best_comps = list(cur_comps)
                best_len = len(best_comps)
            continue

        if cur_v == 0 and new_v > 0:
            # Would go from valid to invalid: accept with SA probability
            if rng.random() < np.exp(-delta_v / T):
                cur_comps = new_comps
                cur_v = new_v
            continue

        # Both invalid: standard SA
        if delta_v <= 0:
            cur_comps = new_comps
            cur_v = new_v
        else:
            if rng.random() < np.exp(-delta_v / T):
                cur_comps = new_comps
                cur_v = new_v

        T *= cool_rate

    # 3. Final greedy-delete polish on best
    polished = _greedy_delete(best_comps, bits, deadline)
    if len(polished) < best_len and _violation_count(polished, bits) == 0:
        best_comps = polished
        best_len = len(polished)

    return best_comps
