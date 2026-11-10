# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Precompute forbidden differences and conflict masks once at module load.
_M = 205
_FORBIDDEN = set()
for _x in range(1, _M):
    _FORBIDDEN.add((_x * _x) % _M)
_FORBIDDEN.discard(0)

# conflict_mask[i] is a bitmask (as a Python int with _M bits) where bit j is 1
# iff (i - j) mod _M is in _FORBIDDEN (i.e., i and j conflict).
_CONFLICT_MASK = [0] * _M
for _i in range(_M):
    _mask = 0
    for _j in range(_M):
        if _i == _j:
            continue
        if (_i - _j) % _M in _FORBIDDEN:
            _mask |= (1 << _j)
    _CONFLICT_MASK[_i] = _mask

_ALL_MASK = (1 << _M) - 1


def _is_valid(selected_mask: int) -> bool:
    """Check that no two selected residues conflict."""
    # For each selected i, ensure no other selected j conflicts with i.
    m = selected_mask
    while m:
        low = m & (-m)
        i = low.bit_length() - 1
        m ^= low
        if (_CONFLICT_MASK[i] & m) != 0:
            return False
    return True


def _greedy_random(rng: random.Random) -> int:
    """Random greedy: build a valid set by adding random residues that fit."""
    order = list(range(_M))
    rng.shuffle(order)
    mask = 0
    for cand in order:
        if (_CONFLICT_MASK[cand] & mask) == 0:
            mask |= (1 << cand)
    return mask


def _local_search(mask: int, rng: random.Random, deadline: float) -> int:
    """Refine mask via 1-add, 1-swap, and 2-swap moves until no improvement or deadline."""
    improved = True
    while improved and time.time() < deadline:
        improved = False

        # 1-add: try to add any residue that doesn't conflict with current set.
        addable = []
        for c in range(_M):
            if (mask >> c) & 1:
                continue
            if (_CONFLICT_MASK[c] & mask) == 0:
                addable.append(c)
        if addable:
            # Pick the candidate that minimizes future conflicts with non-selected nodes.
            best_c = None
            best_score = None
            for c in addable:
                non_sel = _ALL_MASK ^ mask
                cnt = bin(_CONFLICT_MASK[c] & non_sel).count('1')
                if best_score is None or cnt < best_score:
                    best_score = cnt
                    best_c = c
            if best_c is not None:
                mask |= (1 << best_c)
                improved = True
                continue

        # Build list of selected elements.
        selected = []
        m = mask
        while m:
            low = m & (-m)
            i = low.bit_length() - 1
            m ^= low
            selected.append(i)
        if not selected:
            break

        # 1-swap: try removing one selected s and adding one non-selected c.
        rng.shuffle(selected)
        for s in selected:
            if time.time() >= deadline:
                break
            base = mask & ~(1 << s)
            best_c = None
            best_score = None
            for c in range(_M):
                if c == s:
                    continue
                if (base >> c) & 1:
                    continue
                if (_CONFLICT_MASK[c] & base) != 0:
                    continue
                non_sel = _ALL_MASK ^ base
                non_sel &= ~(1 << c)
                cnt = bin(_CONFLICT_MASK[c] & non_sel).count('1')
                if best_score is None or cnt < best_score:
                    best_score = cnt
                    best_c = c
            if best_c is not None:
                new_mask = base | (1 << best_c)
                if new_mask != mask:
                    mask = new_mask
                    improved = True
                    break  # restart outer while to re-scan

        # 2-swap: try removing two selected s1, s2 and adding two non-selected c1, c2.
        # This can escape local minima that 1-swap cannot.
        if len(selected) >= 2 and time.time() < deadline:
            # Try a limited number of 2-swap attempts to avoid O(n^4) blowup.
            rng.shuffle(selected)
            n_selected = len(selected)
            # Sample up to ~50 pairs (s1, s2) from selected.
            max_pairs = min(50, n_selected * (n_selected - 1) // 2)
            pairs_tried = 0
            for i in range(n_selected):
                if time.time() >= deadline or pairs_tried >= max_pairs:
                    break
                for j in range(i + 1, n_selected):
                    if time.time() >= deadline or pairs_tried >= max_pairs:
                        break
                    pairs_tried += 1
                    s1, s2 = selected[i], selected[j]
                    base = mask & ~(1 << s1) & ~(1 << s2)
                    # Find all c that can be added to base.
                    addable_to_base = []
                    for c in range(_M):
                        if c == s1 or c == s2:
                            continue
                        if (base >> c) & 1:
                            continue
                        if (_CONFLICT_MASK[c] & base) != 0:
                            continue
                        addable_to_base.append(c)
                    if len(addable_to_base) < 2:
                        continue
                    # Need to pick two c's that don't conflict with each other.
                    # Try pairs from addable_to_base.
                    rng.shuffle(addable_to_base)
                    max_c_pairs = min(30, len(addable_to_base) * (len(addable_to_base) - 1) // 2)
                    c_pairs_tried = 0
                    for a in range(len(addable_to_base)):
                        if time.time() >= deadline or c_pairs_tried >= max_c_pairs:
                            break
                        for b in range(a + 1, len(addable_to_base)):
                            if time.time() >= deadline or c_pairs_tried >= max_c_pairs:
                                break
                            c_pairs_tried += 1
                            c1, c2 = addable_to_base[a], addable_to_base[b]
                            # Check c1 and c2 don't conflict.
                            if (_CONFLICT_MASK[c1] & (1 << c2)) != 0:
                                continue
                            new_mask = base | (1 << c1) | (1 << c2)
                            # Verify new_mask is valid (should be by construction, but double-check).
                            if new_mask != mask:
                                # Quick validity check: no conflicts within new_mask.
                                valid = True
                                m2 = new_mask
                                while m2:
                                    low = m2 & (-m2)
                                    idx = low.bit_length() - 1
                                    m2 ^= low
                                    if (_CONFLICT_MASK[idx] & m2) != 0:
                                        valid = False
                                        break
                                if valid:
                                    mask = new_mask
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break  # restart outer while
    return mask


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero
    square modulo 205. Uses multiple random greedy starts plus bitmask-accelerated
    local search (1-add, 1-swap) with a wall-clock deadline guard.
    """
    start = time.time()
    deadline = start + 540.0  # safety margin under 630s wall-clock

    best_mask = 0
    best_size = 0

    # Deterministic greedy (original order) as a baseline.
    mask = 0
    for cand in range(_M):
        if (_CONFLICT_MASK[cand] & mask) == 0:
            mask |= (1 << cand)
    if _is_valid(mask):
        sz = bin(mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Random restarts with local search.
    rng = random.Random(20240520)
    restart = 0
    while time.time() < deadline:
        restart += 1
        m0 = _greedy_random(rng)
        m1 = _local_search(m0, rng, deadline)
        sz = bin(m1).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = m1
        # Try local search on the current best more frequently.
        if restart % 5 == 0 and time.time() < deadline:
            m2 = _local_search(best_mask, rng, deadline)
            sz2 = bin(m2).count('1')
            if sz2 > best_size:
                best_size = sz2
                best_mask = m2
        # Also try a second local search pass on the best every 3 restarts.
        if restart % 3 == 0 and time.time() < deadline:
            m3 = _local_search(best_mask, rng, deadline)
            sz3 = bin(m3).count('1')
            if sz3 > best_size:
                best_size = sz3
                best_mask = m3

    # Extract residues from best_mask.
    residues = []
    m = best_mask
    while m:
        low = m & (-m)
        i = low.bit_length() - 1
        m ^= low
        residues.append(i)
    residues.sort()

    # Final validity check (defensive).
    sel = set(residues)
    for a in residues:
        for b in residues:
            if a == b:
                continue
            if (a - b) % _M in _FORBIDDEN:
                # Should not happen; fall back to a known-valid trivial set.
                return np.array([0], dtype=np.int64)

    if not residues:
        return np.array([0], dtype=np.int64)

    return np.array(residues, dtype=np.int64)


# EVOLVE-BLOCK-END
