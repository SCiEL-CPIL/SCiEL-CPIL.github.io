"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import Counter
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 87.0  # safety margin under 90s

    def evaluate(points):
        """Compute the maximum multiplicity of any positive doubled area using NumPy."""
        pts = np.array(points, dtype=np.int32)
        n = len(pts)
        # Compute all C(n,3) determinants vectorized
        # For each pair (i,j), compute det with all k > j
        counts = Counter()
        # Use a faster approach: for each pair, compute cross products with all remaining
        for i in range(n):
            x1, y1 = pts[i, 0], pts[i, 1]
            for j in range(i + 1, n):
                x2, y2 = pts[j, 0], pts[j, 1]
                dx1 = x2 - x1
                dy1 = y2 - y1
                # Vectorized: all k > j
                remaining = pts[j+1:, :]
                if len(remaining) == 0:
                    continue
                dx2 = remaining[:, 0] - x1
                dy2 = remaining[:, 1] - y1
                dets = np.abs(dx1 * dy2 - dy1 * dx2)
                # Count non-zero determinants
                mask = dets > 0
                if mask.any():
                    vals = dets[mask]
                    for v in vals:
                        counts[v] += 1
        if not counts:
            return 0
        return max(counts.values())

    def evaluate_fast(points):
        """Even faster evaluation using full vectorization per pair."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        counts = Counter()
        for i in range(n - 2):
            x1, y1 = pts[i, 0], pts[i, 1]
            for j in range(i + 1, n - 1):
                x2, y2 = pts[j, 0], pts[j, 1]
                dx1 = x2 - x1
                dy1 = y2 - y1
                rem = pts[j+1:, :]
                if len(rem) == 0:
                    continue
                dx2 = rem[:, 0] - x1
                dy2 = rem[:, 1] - y1
                dets = np.abs(dx1 * dy2 - dy1 * dx2)
                mask = dets > 0
                if mask.any():
                    vals = dets[mask].tolist()
                    for v in vals:
                        counts[v] += 1
        if not counts:
            return 0
        return max(counts.values())

    def evaluate_vectorized(points):
        """Fully vectorized evaluation - fastest option."""
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        # Generate all index triples
        indices = np.array(list(combinations(range(n), 3)), dtype=np.int32)
        if len(indices) == 0:
            return 0
        p1 = pts[indices[:, 0]]
        p2 = pts[indices[:, 1]]
        p3 = pts[indices[:, 2]]
        # det = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
        dets = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                      (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
        mask = dets > 0
        if not mask.any():
            return 0
        vals = dets[mask]
        # Use np.unique with return_counts for fast counting
        unique_vals, counts_arr = np.unique(vals, return_counts=True)
        return int(counts_arr.max())

    best_points = None
    best_score = 0

    def try_points(pts):
        nonlocal best_points, best_score
        if len(pts) != 30:
            return
        if len(set(pts)) != 30:
            return
        if not all(-30 <= px <= 30 and -30 <= py <= 30 for px, py in pts):
            return
        score = evaluate_vectorized(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)

    # Initial configuration: two horizontal lines at y=0 and y=1, each with 15 points
    try_points([(x, 0) for x in range(15)] + [(x, 1) for x in range(15)])

    # Try grid configurations: w x h grids with exactly 30 points
    grid_configs = [
        (5, 6), (6, 5), (10, 3), (3, 10), (15, 2), (2, 15)
    ]
    for w, h in grid_configs:
        if w * h != 30:
            continue
        pts = [(x, y) for x in range(w) for y in range(h)]
        try_points(pts)
        if time.time() > deadline:
            return best_points

    # Try different line distances for two parallel lines
    for d in range(1, 8):
        pts = [(x, 0) for x in range(15)] + [(x, d) for x in range(15)]
        try_points(pts)
        if time.time() > deadline:
            return best_points

    # Try 3-line configurations with varying distributions and offsets
    for d in range(1, 6):
        for a in range(5, 26):
            for b in range(5, 30 - a + 1):
                c = 30 - a - b
                if c < 1:
                    continue
                # Try with offset on middle line
                for off in range(0, 3):
                    pts = [(x, 0) for x in range(a)] + [(x + off, d) for x in range(b)] + [(x, 2 * d) for x in range(c)]
                    try_points(pts)
                    if time.time() > deadline:
                        return best_points
        if time.time() > deadline:
            return best_points

    # Try 4-line configurations
    for d in range(1, 4):
        for a in range(4, 16):
            for b in range(4, 20):
                for c in range(4, 24):
                    e = 30 - a - b - c
                    if e < 1:
                        continue
                    pts = [(x, 0) for x in range(a)] + [(x, d) for x in range(b)] + \
                          [(x, 2*d) for x in range(c)] + [(x, 3*d) for x in range(e)]
                    try_points(pts)
                    if time.time() > deadline:
                        return best_points
    if time.time() > deadline:
        return best_points

    # Number-theoretic construction: generalized lattice
    for n1, n2 in [(5, 6), (6, 5), (10, 3), (3, 10), (15, 2), (2, 15)]:
        for a in range(-4, 5):
            for c in range(-4, 5):
                for b in range(-4, 5):
                    for d in range(-4, 5):
                        if a == 0 and c == 0:
                            continue
                        if b == 0 and d == 0:
                            continue
                        det_basis = a * d - b * c
                        if det_basis == 0:
                            continue
                        pts = []
                        for i in range(n1):
                            for j in range(n2):
                                px = i * a + j * b
                                py = i * c + j * d
                                pts.append((px, py))
                        if len(set(pts)) != 30:
                            continue
                        if not all(-30 <= px <= 30 and -30 <= py <= 30 for px, py in pts):
                            continue
                        try_points(pts)
                        if time.time() > deadline:
                            return best_points
    if time.time() > deadline:
        return best_points

    # Local search with delta evaluation for speed
    import random
    random.seed(42)

    if best_points is None:
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        best_score = evaluate_vectorized(best_points)

    def evaluate_delta(points, changed_idx, old_pt):
        """Evaluate only the triples involving the changed point."""
        pts = list(points)
        pts[changed_idx] = old_pt
        n = 30
        new_x, new_y = old_pt
        # Compute all determinants involving point changed_idx
        dets = []
        for i in range(n):
            if i == changed_idx:
                continue
            for j in range(i + 1, n):
                if j == changed_idx:
                    continue
                if i == changed_idx or j == changed_idx:
                    # One of them is the changed point
                    other_i = j if i == changed_idx else i
                    other_j = i if j == changed_idx else j
                    # Actually let me redo this more carefully
                    pass
        # Simpler: just recompute all triples involving the changed index
        count = Counter()
        idx = changed_idx
        x1, y1 = pts[idx]
        for i in range(n):
            if i == idx:
                continue
            for j in range(i + 1, n):
                if j == idx:
                    continue
                if i == idx or j == idx:
                    # Triple includes the changed point
                    if i == idx:
                        x2, y2 = pts[j]
                        x3, y3 = pts[i+1 if i+1 < n and i+1 != idx else i+2]
                        # This is getting complicated, just do full eval
                        pass
        # Just do full evaluation for correctness
        return evaluate_vectorized(pts)

    # Simple local search: try moving each point
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(30):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[idx]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    new_points = list(best_points)
                    new_points[idx] = (new_x, new_y)
                    if len(set(new_points)) != 30:
                        continue
                    score = evaluate_vectorized(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Random restarts with larger perturbations
    for iteration in range(200):
        if time.time() > deadline:
            break
        new_points = list(best_points)
        idx = random.randint(0, 29)
        old_x, old_y = new_points[idx]
        new_x = old_x + random.randint(-6, 6)
        new_y = old_y + random.randint(-6, 6)
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            continue
        new_points[idx] = (new_x, new_y)
        if len(set(new_points)) != 30:
            continue
        score = evaluate_vectorized(new_points)
        if score > best_score:
            best_score = score
            best_points = new_points

    # Pair swaps
    for iteration in range(300):
        if time.time() > deadline:
            break
        new_points = list(best_points)
        i = random.randint(0, 29)
        j = random.randint(0, 29)
        if i == j:
            continue
        new_points[i], new_points[j] = new_points[j], new_points[i]
        if len(set(new_points)) != 30:
            continue
        score = evaluate_vectorized(new_points)
        if score > best_score:
            best_score = score
            best_points = new_points

    # Final polish: very small moves
    if time.time() < deadline - 2:
        for idx in range(30):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[idx]
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    new_points = list(best_points)
                    new_points[idx] = (new_x, new_y)
                    if len(set(new_points)) != 30:
                        continue
                    score = evaluate_vectorized(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points

    return best_points
# EVOLVE-BLOCK-END
