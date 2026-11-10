import numpy as np
import time
import random
import sys

_p = 7
_kq = 3
_Q = 49
_SLOTS = 2401
_INF = 10  # vertical direction index

def _encode(xa, xb, ya, yb):
    return ((xa * 7 + xb) * 49) + (ya * 7 + yb)

def _xy_from_slot(s):
    yb = s % 7; s //= 7
    ya = s % 7; s //= 7
    xb = s % 7; s //= 7
    xa = s % 7
    return xa, xb, ya, yb

# Precompute all points
_PTS = np.zeros((_SLOTS, 4), dtype=np.uint8)
for s in range(_SLOTS):
    xa, xb, ya, yb = _xy_from_slot(s)
    _PTS[s] = [xa, xb, ya, yb]

# Precompute lines for each direction
_lines = []
_line_of = []

for d in range(50):
    if d == _INF:
        dl = []
        d_lof = np.zeros(_SLOTS, dtype=np.int16)
        for xa in range(7):
            for xb in range(7):
                line = []
                for ya in range(7):
                    for yb in range(7):
                        s = _encode(xa, xb, ya, yb)
                        line.append(s)
                        d_lof[s] = len(dl)
                dl.append(np.array(line, dtype=np.int32))
        _lines.append(dl)
        _line_of.append(d_lof)
    else:
        da, db = (d // 7, d % 7)
        dl = []
        d_lof = np.zeros(_SLOTS, dtype=np.int16)
        for ya0 in range(7):
            for yb0 in range(7):
                line = []
                for ta in range(7):
                    for tb in range(7):
                        txa = (ta * da + tb * db * _kq) % _p
                        txb = (ta * db + tb * da) % _p
                        nxa = ta % _p
                        nxb = tb % _p
                        nya = (ya0 + txa) % _p
                        nyb = (yb0 + txb) % _p
                        s = _encode(nxa, nxb, nya, nyb)
                        line.append(s)
                        d_lof[s] = len(dl)
                dl.append(np.array(line, dtype=np.int32))
        _lines.append(dl)
        _line_of.append(d_lof)

def _compute_Scount(S):
    Scount = np.zeros((50, _SLOTS), dtype=np.int16)
    S_arr = np.frombuffer(S, dtype=np.uint8) if isinstance(S, bytes) else np.array(S, dtype=np.uint8)
    for d in range(50):
        for line in _lines[d]:
            c = np.sum(S_arr[line])
            Scount[d, line] = c
    return Scount

def _is_valid(S):
    S_arr = np.frombuffer(S, dtype=np.uint8) if isinstance(S, bytes) else np.array(S, dtype=np.uint8)
    Scount = _compute_Scount(S)
    for s in range(_SLOTS):
        ss = S_arr[s]
        ok = False
        for d in range(50):
            if Scount[d, s] - ss == 0:
                ok = True
                break
        if not ok:
            return False
    return True

def _to_array(S):
    S_arr = np.frombuffer(S, dtype=np.uint8) if isinstance(S, bytes) else np.array(S, dtype=np.uint8)
    pts = []
    for s in range(_SLOTS):
        if not S_arr[s]:
            xa, xb, ya, yb = _xy_from_slot(s)
            pts.append([xa, xb, ya, yb])
    return np.array(pts, dtype=np.int64).reshape(-1, 2, 2)

def _fast_greedy(S, avail_dirs, order):
    """
    Fast greedy: add points to S in given order if safe.
    Returns the updated S, avail_dirs, and count added.
    """
    added = 0
    for p in order:
        if S[p]:
            continue
        safe = True
        changes = []
        for d in range(50):
            line_idx = _line_of[d][p]
            line = _lines[d][line_idx]
            mask = 1 << d
            for q in line:
                if q == p:
                    continue
                if (avail_dirs[q] & mask) != 0:
                    if (avail_dirs[q] & ~mask) == 0:
                        safe = False
                        break
                    changes.append((q, d))
            if not safe:
                break
        if safe:
            for q, d in changes:
                avail_dirs[q] &= ~(1 << d)
            S[p] = 1
            added += 1
    return S, avail_dirs, added

def _local_search(S, avail_dirs, deadline, rng):
    """
    Local search: try to add more points to S by finding points that are now safe
    (maybe they became safe after other points were added? No, adding to S only
    removes directions, so if a point was unsafe before, it's still unsafe).
    
    Actually, the only way to add more points is to REMOVE some points from S first.
    This is a 2-opt style move: remove a set of points, then re-add a different set.
    
    Simpler: try to swap. For a point p not in S, find a point q in S such that
    removing q and adding p keeps validity.
    
    But this is complex. Let's just do multiple random restarts of the greedy.
    """
    best_S = S[:]
    best_size = sum(S)
    
    # Try a few restarts with different orderings
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        S_try = bytearray(_SLOTS)
        avail_dirs = [0xFFFFFFFFFF] * _SLOTS
        order = list(range(_SLOTS))
        rng.shuffle(order)
        
        S_try, avail_dirs, added = _fast_greedy(S_try, avail_dirs, order)
        
        if added > best_size:
            best_size = added
            best_S = S_try[:]
    
    return best_S, best_size

def _simulated_annealing(S, avail_dirs, deadline, rng):
    """
    Simulated annealing on the set S.
    Moves: add a point to S, remove a point from S.
    Energy: -|S| (we want to minimize this, i.e., maximize |S|)
    Constraint: S must be valid.
    
    We maintain validity by checking after each move.
    To speed up, we can check only locally affected points.
    """
    current_S = S[:]
    current_size = sum(S)
    best_S = current_S[:]
    best_size = current_size
    
    # Rebuild avail_dirs for current_S
    avail_dirs = [0xFFFFFFFFFF] * _SLOTS
    for p in range(_SLOTS):
        if current_S[p]:
            for d in range(50):
                line_idx = _line_of[d][p]
                line = _lines[d][line_idx]
                mask = 1 << d
                for q in line:
                    if q != p:
                        avail_dirs[q] &= ~mask
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    steps = 0
    while time.time() < deadline - 10 and T > T_min:
        steps += 1
        
        # Choose a random move
        if rng.random() < 0.5:
            # Try to add a point
            candidates = [p for p in range(_SLOTS) if not current_S[p]]
            if not candidates:
                T *= alpha
                continue
            p = rng.choice(candidates)
            
            # Check if adding p is safe
            safe = True
            changes = []
            for d in range(50):
                line_idx = _line_of[d][p]
                line = _lines[d][line_idx]
                mask = 1 << d
                for q in line:
                    if q == p:
                        continue
                    if (avail_dirs[q] & mask) != 0:
                        if (avail_dirs[q] & ~mask) == 0:
                            safe = False
                            break
                        changes.append((q, d))
                if not safe:
                    break
            
            if safe:
                # Apply
                for q, d in changes:
                    avail_dirs[q] &= ~(1 << d)
                current_S[p] = 1
                current_size += 1
                
                if current_size > best_size:
                    best_size = current_size
                    best_S = current_S[:]
            # If not safe, skip (don't accept invalid states)
        
        else:
            # Try to remove a point
            candidates = [p for p in range(_SLOTS) if current_S[p]]
            if not candidates:
                T *= alpha
                continue
            p = rng.choice(candidates)
            
            # Remove p: this restores directions for neighbors
            for d in range(50):
                line_idx = _line_of[d][p]
                line = _lines[d][line_idx]
                mask = 1 << d
                for q in line:
                    if q != p:
                        avail_dirs[q] |= mask
            current_S[p] = 0
            current_size -= 1
        
        T *= alpha
    
    return best_S, best_size

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 580
    
    rng = random.Random(42)
    
    best_S = None
    best_size = 0
    
    # Phase 1: Multiple random restarts of greedy
    for restart in range(50):
        if time.time() > deadline - 30:
            break
        
        S_try = bytearray(_SLOTS)
        avail_dirs = [0xFFFFFFFFFF] * _SLOTS
        order = list(range(_SLOTS))
        rng.shuffle(order)
        
        S_try, avail_dirs, added = _fast_greedy(S_try, avail_dirs, order)
        
        if added > best_size:
            best_size = added
            best_S = S_try[:]
    
    if best_S is None:
        best_S = bytearray(_SLOTS)
        best_size = 0
    
    # Phase 2: Local search / simulated annealing from the best solution
    if time.time() < deadline - 15:
        # Rebuild avail_dirs for best_S
        avail_dirs = [0xFFFFFFFFFF] * _SLOTS
        for p in range(_SLOTS):
            if best_S[p]:
                for d in range(50):
                    line_idx = _line_of[d][p]
                    line = _lines[d][line_idx]
                    mask = 1 << d
                    for q in line:
                        if q != p:
                            avail_dirs[q] &= ~mask
        
        best_S, best_size = _simulated_annealing(best_S, avail_dirs, deadline, rng)
    
    # Verify validity
    if not _is_valid(bytes(best_S)):
        best_S = bytearray(_SLOTS)
    
    return _to_array(bytes(best_S))
