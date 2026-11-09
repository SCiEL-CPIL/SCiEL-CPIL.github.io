"""
Edge-colourings of K16 in which every K4 sees at least 5 colours.

Approach (same family as seed, but much faster + targeted):
- Start from the valid (i+j) mod 17 construction (17 colours).
- Use vectorized NumPy checks so we can run many more search steps.
- The ONLY move we accept is a *colour merge* (map one colour onto another),
  which keeps the palette size monotonically non-increasing and every
  accepted state valid.  We also run a SA-style search over merges and a
  dedicated "fold the least-used colour" phase.
- Return the best valid colouring before the deadline.
"""

import time
import random
import numpy as np

DEADLINE = 145.0  # safety margin before the 150s kill

# ---- edge index / K4 precomputation -----------------------------------------

def _edge_index(i, j):
    return i * 15 - i * (i - 1) // 2 + j - 1 - i

def _precompute_k4s():
    k4s = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    k4s.append((
                        _edge_index(a, b), _edge_index(a, c), _edge_index(a, d),
                        _edge_index(b, c), _edge_index(b, d), _edge_index(c, d),
                    ))
    return k4s

_K4S = _precompute_k4s()
_K4S_ARR = np.array(_K4S, dtype=np.int64)          # (1820, 6)
_NK4 = len(_K4S)

# ---- initial colouring -------------------------------------------------------

def _initial_colouring():
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

# ---- fast validity / colour-count checks -------------------------------------

def _min_colours_py(colours):
    """Return the minimum number of distinct colours over all K4s (0 = none)."""
    c = np.asarray(colours)
    best = 10**9
    for e in _K4S_ARR:
        vals = c[e]
        k = len(np.unique(vals))
        if k < best:
            best = k
            if best <= 4:
                return best
    return best

def _min_colours(colours):
    """Vectorised: min distinct colours over all K4s."""
    c = np.asarray(colours)
    # sort each K4's 6 colours
    s = np.sort(c[_K4S_ARR], axis=1)
    # count distinct = 1 + number of positions where value changes
    changes = (s[:, 1:] != s[:, :-1]).sum(axis=1)
    return int(changes.min()) + 1

def _is_valid(colours):
    return _min_colours(colours) >= 5

# ---- colour merge helpers ----------------------------------------------------

def _merge(colours, src, dst):
    """Return new list where every occurrence of colour `src` becomes `dst`."""
    out = list(colours)
    for i in range(120):
        if out[i] == src:
            out[i] = dst
    return out

def _used_colours(colours):
    return sorted(set(colours))

def _palette(colours):
    return sorted(set(colours))

# ---- search ------------------------------------------------------------------

def _sa_merge_search(colours, deadline, rng):
    """SA over colour-merge moves.  Palette only shrinks; every accepted state valid."""
    cur = list(colours)
    best = list(cur)
    cur_palette = _palette(cur)
    best_palette = list(cur_palette)
    cur_size = len(cur_palette)
    best_size = cur_size

    T = 30.0
    T_min = 1.0
    cooling = 0.99995

    while time.time() < deadline:
        if cur_size <= best_size:
            best = list(cur)
            best_palette = list(cur_palette)
            best_size = cur_size
            if best_size <= 13:
                break

        if len(cur_palette) < 2:
            break

        # choose a random colour to fold into another
        src = cur_palette[rng.randrange(len(cur_palette))]
        candidates = [c for c in cur_palette if c != src]
        dst = candidates[rng.randrange(len(candidates))]

        trial = _merge(cur, src, dst)
        mc = _min_colours(trial)
        if mc >= 5:
            # valid merge: accept (palette shrinks)
            cur = trial
            cur_palette = _palette(cur)
            cur_size = len(cur_palette)
            # accept improvement in palette size
        else:
            # invalid: accept with small probability (SA) to escape local optima,
            # but we must keep `cur` valid, so we only tentatively explore.
            # Instead of corrupting cur, we simply reject (keep cur valid).
            pass

        T *= cooling
        if T < T_min:
            T = T_min

    return best, best_size

def _fold_phase(colours, deadline, rng):
    """Repeatedly try to fold the least-used colour into the best partner."""
    cur = list(colours)
    best = list(cur)
    best_size = len(_palette(cur))

    while time.time() < deadline:
        if best_size <= 13:
            break
        palette = _palette(cur)
        if len(palette) < 2:
            break

        # colour usage counts
        counts = {}
        for c in cur:
            counts[c] = counts.get(c, 0) + 1

        # pick the least-used colour to fold
        src = min(palette, key=lambda c: (counts[c], c))

        # try to find a partner dst such that merging src->dst stays valid.
        # Order partners by "slack": prefer colours that co-occur with src on
        # K4s that have many colours (more room to absorb).
        # Simple approach: try all, accept the first valid; if none, try
        # recolouring a few src edges to a new colour to create slack, then retry.
        merged = False
        for dst in palette:
            if dst == src:
                continue
            trial = _merge(cur, src, dst)
            if _is_valid(trial):
                cur = trial
                best = list(cur)
                best_size = len(_palette(cur))
                merged = True
                break

        if not merged:
            # create slack: recolour a random subset of src edges to a fresh
            # colour, then attempt merges again.
            src_edges = [i for i in range(120) if cur[i] == src]
            if not src_edges:
                break
            fresh = max(palette) + 1
            k = max(1, len(src_edges) // 2)
            pick = rng.sample(src_edges, min(k, len(src_edges)))
            trial = list(cur)
            for i in pick:
                trial[i] = fresh
            # now try merging fresh into each colour
            done = False
            for dst in _palette(trial):
                if dst == fresh:
                    continue
                t2 = _merge(trial, fresh, dst)
                if _is_valid(t2):
                    cur = t2
                    best = list(cur)
                    best_size = len(_palette(cur))
                    done = True
                    break
            if not done:
                # no progress possible this round; stop to avoid spin
                break

    return best, best_size

def _refine_recolours(colours, deadline, rng):
    """Final polish: single-edge recolours that keep validity and reduce palette."""
    cur = list(colours)
    best = list(cur)
    best_size = len(_palette(cur))
    palette = _palette(cur)

    while time.time() < deadline:
        if best_size <= 13:
            break
        # try a random edge recolour to a colour in palette or a fresh one
        idx = rng.randrange(120)
        old = cur[idx]
        # candidate new colours: all palette colours + a couple fresh
        cands = list(palette) + [max(palette) + 1, max(palette) + 2]
        newc = cands[rng.randrange(len(cands))]
        if newc == old:
            continue
        trial = list(cur)
        trial[idx] = newc
        if _is_valid(trial):
            # accept if it reduces palette, else accept with small prob
            new_size = len(_palette(trial))
            if new_size < best_size:
                cur = trial
                best = list(trial)
                best_size = new_size
                palette = _palette(cur)
            elif rng.random() < 0.15:
                cur = trial
                palette = _palette(cur)

    return best, best_size

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = random.Random(12345)

    colours = _initial_colouring()
    assert _is_valid(colours), "initial construction must be valid"

    best = list(colours)
    best_size = len(_palette(best))

    # Phase 1: SA over merges
    if time.time() < deadline:
        r, s = _sa_merge_search(best, deadline, rng)
        if s < best_size and _is_valid(r):
            best, best_size = r, s

    # Phase 2: targeted fold phase (most direct attack on colour count)
    if time.time() < deadline:
        r, s = _fold_phase(best, deadline, rng)
        if s < best_size and _is_valid(r):
            best, best_size = r, s

    # Phase 3: recolour polish
    if time.time() < deadline:
        r, s = _refine_recolours(best, deadline, rng)
        if s < best_size and _is_valid(r):
            best, best_size = r, s

    # Final validity guard
    if not _is_valid(best):
        best = _initial_colouring()

    # return exactly 120 ints
    return [int(c) for c in best]

# Pre-compute K4s at module load time for speed
_K4S = _precompute_k4s()
_K4S_ARR = np.array(_K4S, dtype=np.int64)
_NK4 = len(_K4S)
