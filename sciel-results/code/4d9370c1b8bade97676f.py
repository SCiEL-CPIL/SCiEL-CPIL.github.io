# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (q = 49, p = 7, alpha^2 = 3).

    Baseline: exclude points on the Hermitian unital U = {Trace(x)+Norm(y)=0}
    whose y-coordinate is in Y = {(u^2 mod p, v)}. Then greedily remove more
    points from N while every remaining point still has at least one witness
    line (vertical or slope (1,m)) whose other points all lie in N.

    A point x has a witness line iff there exists a line L (through x) with
    L \ {x} subset N, i.e. the count of excluded points on L is <= 1 (and if
    ==1 it must be x itself). We maintain per-line excluded counts for fast checks.
    """
    start_time = time.time()
    deadline = start_time + 570  # safety margin before 600/630s kill

    p = 7
    q = p * p
    k = 3  # alpha^2 = 3 (smallest quadratic non-residue mod 7)

    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 * a2 + k * b1 * b2) % p, (a1 * b2 + a2 * b1) % p)

    # All points indexed
    all_points = [(xa, xb, ya, yb)
                  for xa in range(p) for xb in range(p)
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}

    # Precompute all lines: 49 vertical + 48 * 49 slope lines
    # Each line stored as list of point indices.
    vert_lines = []
    for xa in range(p):
        for xb in range(p):
            pts = [i for i, (pxa, pxb, pya, pyb) in enumerate(all_points)
                   if pxa == xa and pxb == xb]
            vert_lines.append(pts)

    slope_lines = []
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            lines = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines[key].append(i)
            for key in lines:
                slope_lines.append(lines[key])

    all_lines = vert_lines + slope_lines  # 49 + 48*49 = 2393 lines

    # For each point, which lines pass through it
    point_lines = [[] for _ in range(n)]
    for li, line in enumerate(all_lines):
        for i in line:
            point_lines[i].append(li)

    # Initial excluded set S: points on unital with y in Y (special)
    y_special = set((u * u % p, v) for u in range(p) for v in range(p))
    in_N = [True] * n
    excluded = [False] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2 * xa + ya * ya - yb * yb * k) % p == 0 and (ya, yb) in y_special:
            in_N[i] = False
            excluded[i] = True

    # Per-line excluded counts
    line_excl = [0] * len(all_lines)
    for i in range(n):
        if excluded[i]:
            for li in point_lines[i]:
                line_excl[li] += 1

    def try_remove(pt):
        """Attempt to exclude point pt. Returns True if successful."""
        if excluded[pt]:
            return False
        # Lines through pt currently have some excluded count; after removing pt,
        # pt needs a witness line L with line_excl[L] (excluding pt) == 0,
        # i.e. current line_excl[L] == 0 (since pt not yet excluded) OR
        # line_excl[L] == 1 and that one is... but pt not excluded so the 1 is
        # some other point -> not clean. So we need a line L through pt with
        # current line_excl[L] == 0.
        witness = None
        for li in point_lines[pt]:
            if line_excl[li] == 0:
                witness = li
                break
        if witness is None:
            return False
        # Remove pt: update line_excl for lines through pt
        for li in point_lines[pt]:
            line_excl[li] -= 1
        excluded[pt] = True
        in_N[pt] = False
        return True

    def restore(pt):
        for li in point_lines[pt]:
            line_excl[li] += 1
        excluded[pt] = False
        in_N[pt] = True

    # Greedy pass: iterate through candidate points (on unital, not in Y) and
    # any other point, try to remove.
    # Candidates: prioritize points on the unital (most "structured") first.
    candidates = []
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if excluded[i]:
            continue
        on_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
        in_y = (ya, yb) in y_special
        # We can try to remove any point, but on-unital points are the natural
        # "extra" exclusions.
        if on_unital and not in_y:
            candidates.append((0, i))
        else:
            candidates.append((1, i))
    candidates.sort()

    for _, pt in candidates:
        if time.time() > deadline - 30:
            break
        if try_remove(pt):
            pass

    # Second pass: after removals, new points may become removable.
    for _ in range(3):
        if time.time() > deadline - 30:
            break
        progress = False
        for _, pt in candidates:
            if excluded[pt]:
                continue
            if try_remove(pt):
                progress = True
        if not progress:
            break

    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts],
                      dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
