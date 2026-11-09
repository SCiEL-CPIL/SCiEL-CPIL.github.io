"""Construct 20 integer points, |x|,|y|<=100, no four collinear, minimizing largest 3-collinear-free subset."""
import random
import time
from math import gcd
from itertools import combinations

DEADLINE = 130.0  # safety margin below 150s wall clock


def _line_key(p, q):
    """Canonical (A,B,C) for line through p,q: A x + B y + C = 0, gcd-normalized, A>0 or A==0,B>0."""
    x1, y1 = p
    x2, y2 = q
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g:
        A //= g; B //= g; C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _no_four_collinear(points):
    """Return True if no line contains 4 or more of the given points."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                key = _line_key(points[i], points[j])
                # count points on this line
                cnt = 0
                for p in points:
                    A, B, C = key
                    if A * p[0] + B * p[1] + C == 0:
                        cnt += 1
                if cnt >= 4:
                    return False
    return True


def _largest_no_three_collinear(points):
    """Exact branch and bound for the largest subset with no three collinear."""
    n = len(points)
    # Precompute, for each pair, the set of other points collinear with them
    # coll[i] = set of j such that i,j and some k are collinear? Better: for each i, the set of
    # points j that form a collinear triple with i and some k.
    # We'll use: conflict[i] = set of j where there exists k != i,j with i,j,k collinear.
    # Then a subset is valid iff for all i in subset, no two others in subset are in conflict[i]... 
    # Actually simpler: a triple (i,j,k) is forbidden if collinear. A subset is valid iff it contains
    # no collinear triple.
    # Precompute collinear triples as a set of frozensets.
    # For branch and bound, we track a partial set S; when adding point p, we must ensure that for
    # every pair (a,b) in S, a,b,p are not collinear.
    # Precompute for each pair (a,b) the set of points collinear with them.
    pair_collinear = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            s = set()
            for k in range(n):
                if k == i or k == j:
                    continue
                A, B, C = key
                if A * points[k][0] + B * points[k][1] + C == 0:
                    s.add(k)
            pair_collinear[(i, j)] = s

    best = [0]
    best_set = [set()]

    def rec(S, cand):
        if not cand:
            if len(S) > best[0]:
                best[0] = len(S)
                best_set[0] = set(S)
            return
        # Upper bound: len(S) + len(cand)
        if len(S) + len(cand) <= best[0]:
            return
        # Try to add each candidate
        for idx in range(len(cand)):
            p = cand[idx]
            # Check if p can be added: for every pair (a,b) in S, p not in pair_collinear[(a,b)]
            ok = True
            Slist = list(S)
            for ii in range(len(Slist)):
                for jj in range(ii + 1, len(Slist)):
                    a, b = Slist[ii], Slist[jj]
                    key = (a, b) if a < b else (b, a)
                    if p in pair_collinear.get(key, set()):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            S.add(p)
            rec(S, cand[idx + 1:])
            S.discard(p)

    rec(set(), list(range(n)))
    return best[0]


def _construct_grid_arrangement():
    """5 lines in general position, 4 points each, distinct, no four collinear."""
    # Define 5 lines with distinct slopes and intercepts, then pick 4 integer points on each.
    # To keep coordinates small and ensure distinctness, we'll define lines and sample points.
    # Lines: y = m x + b for various m, b.
    # We'll choose lines so that all 20 sampled points are distinct and |x|,|y|<=100.
    lines = [
        # (slope, intercept) as rational; we'll sample integer x giving integer y.
        (1, 0),     # y = x
        (2, 1),     # y = 2x + 1
        (3, -1),    # y = 3x - 1
        (-1, 2),    # y = -x + 2
        (4, 3),     # y = 4x + 3
    ]
    # For each line, pick 4 integer points with small coordinates.
    # We need to ensure all 20 are distinct.
    # Let's pick x values carefully.
    point_sets = []
    for m, b in lines:
        pts = []
        # Try x from -5 to 5, pick 4 that give y in range and distinct
        for x in range(-10, 11):
            y = m * x + b
            if abs(x) <= 100 and abs(y) <= 100:
                pts.append((x, y))
        point_sets.append(pts)
    
    # Now select 4 from each, ensuring global distinctness
    # Greedy with backtracking would be complex; let's just try a fixed selection and verify.
    # We'll pick the first 4 from each that are distinct from all previously chosen.
    chosen = []
    for i in range(5):
        for p in point_sets[i]:
            if p not in chosen:
                chosen.append(p)
                if len([c for c in chosen if c in point_sets[i]]) == 4:
                    break
        if len([c for c in chosen if c in point_sets[i]]) < 4:
            return None  # fallback
    
    # Verify exactly 20 and no four collinear
    if len(chosen) != 20 or len(set(chosen)) != 20:
        return None
    if not _no_four_collinear(chosen):
        return None
    return chosen


def construct():
    start = time.time()
    
    # First try the deterministic grid arrangement
    result = _construct_grid_arrangement()
    if result is not None:
        return [[x, y] for x, y in result]
    
    # Fallback: random search for a valid configuration
    # We want to minimize the score. A good target is around 4-5.
    # Strategy: try to construct 5 lines with 4 points each (score <= 4) via random lines.
    rng = random.Random(42)
    best_score = 20
    best_points = None
    
    # Try many random 5-line arrangements
    for trial in range(5000):
        if time.time() - start > DEADLINE:
            break
        # Generate 5 random lines (as y = m x + b with small integer m, b)
        lines = []
        for _ in range(5):
            m = rng.randint(-5, 5)
            b = rng.randint(-20, 20)
            lines.append((m, b))
        
        # For each line, pick 4 random integer points
        chosen = []
        valid = True
        for m, b in lines:
            # Sample candidate x values
            candidates = []
            for x in range(-50, 51):
                y = m * x + b
                if abs(y) <= 100:
                    candidates.append((x, y))
            if len(candidates) < 4:
                valid = False
                break
            # Pick 4 random from candidates
            rng.shuffle(candidates)
            for p in candidates:
                if p not in chosen:
                    chosen.append(p)
                    if len([c for c in chosen if c in candidates]) == 4:
                        break
            if len([c for c in chosen if c in candidates]) < 4:
                valid = False
                break
        
        if not valid or len(chosen) != 20 or len(set(chosen)) != 20:
            continue
        if not _no_four_collinear(chosen):
            continue
        
        # Compute score
        score = _largest_no_three_collinear(chosen)
        if score < best_score:
            best_score = score
            best_points = chosen
            if score <= 4:
                break  # Can't do better than 4 with 5 lines of 4
        if time.time() - start > DEADLINE:
            break
    
    if best_points is not None:
        return [[x, y] for x, y in best_points]
    
    # Final fallback: return a valid (if not optimal) configuration
    # Use a simple valid set: 20 points on a circle-like arrangement
    # Just return the parabola (score 20) as last resort
    return [[x, x * x] for x in range(-10, 10)]
