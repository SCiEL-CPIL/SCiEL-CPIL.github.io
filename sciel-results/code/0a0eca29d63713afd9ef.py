"""Hexagonal construction for 32 points on the triangular lattice, maximizing unit-distance edges."""
# EVOLVE-BLOCK-START
import time

_NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

def _hex_dist(a, b):
    """Hex distance from origin on triangular lattice: max(|a|,|b|,|a+b|)."""
    return max(abs(a), abs(b), abs(a + b))

def _hexagon_points(r):
    """All lattice points with hex distance <= r from origin."""
    pts = []
    for a in range(-r, r + 1):
        for b in range(-r, r + 1):
            if _hex_dist(a, b) <= r:
                pts.append((a, b))
    return pts

def _count_edges(point_set):
    """Count unit-distance pairs among a set of (a,b) points."""
    cnt = 0
    for (a, b) in point_set:
        for da, db in _NEIGHBORS:
            if (a + da, b + db) in point_set:
                cnt += 1
    return cnt // 2

def _marginal_edges(pt, point_set):
    """Number of new unit edges adding pt would create with existing points."""
    a, b = pt
    c = 0
    for da, db in _NEIGHBORS:
        if (a + da, b + db) in point_set:
            c += 1
    return c

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin under 150s wall-clock

    # Core: full hexagon of radius 2 (19 points).
    core = set(_hexagon_points(2))
    assert len(core) == 19, len(core)

    # Candidate ring-3 points (hex distance exactly 3).
    ring3 = [p for p in _hexagon_points(3) if _hex_dist(*p) == 3]
    assert len(ring3) == 18, len(ring3)

    # Need 13 more points to reach 32. Greedy: pick ring-3 point with max marginal gain.
    chosen = set(core)
    remaining = list(ring3)
    target = 32
    while len(chosen) < target and remaining:
        if time.time() > deadline:
            break
        best = max(remaining, key=lambda p: _marginal_edges(p, chosen))
        chosen.add(best)
        remaining.remove(best)

    # If greedy didn't reach 32 (shouldn't happen), fill with any unused ring-3.
    if len(chosen) < target:
        for p in ring3:
            if len(chosen) >= target:
                break
            if p not in chosen:
                chosen.add(p)

    # Final safety: ensure exactly 32 distinct points within bounds.
    pts = list(chosen)
    # Verify bounds |a|,|b| <= 64 (trivially true for hex r=3, but guard anyway).
    pts = [p for p in pts if abs(p[0]) <= 64 and abs(p[1]) <= 64]
    if len(pts) < 32:
        # Pad deterministically (should never trigger).
        for a in range(65):
            if len(pts) >= 32:
                break
            for b in range(65):
                if len(pts) >= 32:
                    break
                if (a, b) not in chosen:
                    pts.append((a, b))
                    chosen.add((a, b))
    pts = pts[:32]

    # Deadline guard: return immediately; construction is O(1) anyway.
    return [[a, b] for (a, b) in pts]
# EVOLVE-BLOCK-END
