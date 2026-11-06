"""Finite-field-inspired 4x4 grid seed + fast incremental simulated annealing.
Maximizes the number of lines containing exactly 4 of 16 integer points,
with no five collinear and |x|,|y| <= 1000."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

_N = 16
_LIMIT = 1000

def _line_key(p1, p2):
    """Normalized line key (a,b,c) for a*x + b*y = c through p1,p2 (gcd-normalized)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * p1[0] + b * p1[1]
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

class _Scorer:
    """Incremental exact-4 line counter over a fixed set of 16 points."""
    def __init__(self, points):
        self.points = [tuple(p) for p in points]
        self.keys = []
        self.pair_key = [[None] * _N for _ in range(_N)]
        for i in range(_N):
            for j in range(i + 1, _N):
                k = _line_key(self.points[i], self.points[j])
                self.pair_key[i][j] = k
                self.pair_key[j][i] = k
                self.keys.append(k)
        self.counts = defaultdict(int)
        for k in self.keys:
            self.counts[k] += 1
        self.exact4 = sum(1 for v in self.counts.values() if v == 4)
        self.has5 = any(v >= 5 for v in self.counts.values())

    def move(self, idx, new_pt):
        """Return (exact4, has5) if point idx moved to new_pt; does NOT mutate state."""
        new_pt = tuple(new_pt)
        old_pt = self.points[idx]
        if new_pt == old_pt:
            return self.exact4, self.has5
        touched = []
        for j in range(_N):
            if j == idx:
                continue
            k = self.pair_key[idx][j]
            if k is not None and k not in touched:
                touched.append(k)
        for k in touched:
            self.counts[k] -= 1
        for j in range(_N):
            if j == idx:
                continue
            k = _line_key(new_pt, self.points[j])
            if k is not None and k not in touched:
                touched.append(k)
            self.counts[k] += 1
        exact4 = sum(1 for v in self.counts.values() if v == 4)
        has5 = any(v >= 5 for v in self.counts.values())
        for k in touched:
            self.counts[k] -= 1
        for j in range(_N):
            if j == idx:
                continue
            k = self.pair_key[idx][j]
            if k is not None and k in touched:
                self.counts[k] += 1
        return exact4, has5

    def commit(self, idx, new_pt):
        """Apply move idx->new_pt to internal state (assumes validity checked)."""
        new_pt = tuple(new_pt)
        old_pt = self.points[idx]
        if new_pt == old_pt:
            return
        touched = []
        for j in range(_N):
            if j == idx:
                continue
            k = self.pair_key[idx][j]
            if k is not None and k not in touched:
                touched.append(k)
        for k in touched:
            self.counts[k] -= 1
        for j in range(_N):
            if j == idx:
                continue
            k = _line_key(new_pt, self.points[j])
            if k is not None and k not in touched:
                touched.append(k)
            self.counts[k] += 1
        self.exact4 = sum(1 for v in self.counts.values() if v == 4)
        self.has5 = any(v >= 5 for v in self.counts.values())
        self.points[idx] = new_pt
        for j in range(_N):
            if j == idx:
                continue
            self.pair_key[idx][j] = _line_key(new_pt, self.points[j])
            self.pair_key[j][idx] = self.pair_key[idx][j]
        for ki, k in enumerate(self.keys):
            if k is None:
                continue
            # update keys list entries that referenced the moved point's pairs
            pass
        # rebuild keys list to keep consistency (cheap: 120 entries)
        self.keys = []
        for i in range(_N):
            for j in range(i + 1, _N):
                self.keys.append(self.pair_key[i][j])

def _full_score(points):
    """Return (exact4, has5) by full recount (verification)."""
    counts = defaultdict(int)
    for i in range(_N):
        for j in range(i + 1, _N):
            k = _line_key(points[i], points[j])
            if k is not None:
                counts[k] += 1
    exact4 = sum(1 for v in counts.values() if v == 4)
    has5 = any(v >= 5 for v in counts.values())
    return exact4, has5

def _candidates_on_line(line_key, existing_indices, points, used_set):
    """Integer points on line within bounds, not already used."""
    a, b, c = line_key
    ref = list(existing_indices)[0]
    x_ref, y_ref = points[ref]
    dx_step, dy_step = b, -a
    if dx_step == 0 and dy_step == 0:
        return []
    k_min, k_max = 10**9, -10**9
    if dx_step > 0:
        k_min = max(k_min, math.ceil((-_LIMIT - x_ref) / dx_step))
        k_max = min(k_max, math.floor((_LIMIT - x_ref) / dx_step))
    elif dx_step < 0:
        k_min = max(k_min, math.ceil((_LIMIT - x_ref) / dx_step))
        k_max = min(k_max, math.floor((-_LIMIT - x_ref) / dx_step))
    else:
        if not (-_LIMIT <= x_ref <= _LIMIT):
            return []
    if dy_step > 0:
        k_min = max(k_min, math.ceil((-_LIMIT - y_ref) / dy_step))
        k_max = min(k_max, math.floor((_LIMIT - y_ref) / dy_step))
    elif dy_step < 0:
        k_min = max(k_min, math.ceil((_LIMIT - y_ref) / dy_step))
        k_max = min(k_max, math.floor((-_LIMIT - y_ref) / dy_step))
    else:
        if not (-_LIMIT <= y_ref <= _LIMIT):
            return []
    if k_min > k_max:
        return []
    out = []
    for k in range(k_min, k_max + 1):
        x = x_ref + k * dx_step
        y = y_ref + k * dy_step
        pt = (x, y)
        if pt not in used_set:
            out.append(pt)
    return out

def _local_search(points, deadline, rng):
    """Fast SA: targeted 3->4 line completion, swaps, perturbations."""
    pts = [tuple(p) for p in points]
    used = set(pts)
    scorer = _Scorer(pts)
    cur_score = scorer.exact4
    if scorer.has5:
        return [list(p) for p in pts], 0
    best = [p for p in pts]
    best_score = cur_score
    T = 2.0
    T_min = 0.05
    cooling = 0.995
    no_improve = 0
    max_no_improve = 4000

    def _line_groups():
        groups = defaultdict(set)
        for i in range(_N):
            for j in range(i + 1, _N):
                k = scorer.pair_key[i][j]
                if k is not None:
                    groups[k].add(i)
                    groups[k].add(j)
        return groups

    while time.time() < deadline - 6 and no_improve < max_no_improve:
        mt = rng.random()
        accepted = False
        if mt < 0.6:
            # Targeted: complete a 3-point line into a 4-point line.
            groups = _line_groups()
            three = [k for k, v in groups.items() if len(v) == 3]
            if three:
                # Prefer lines whose completion doesn't create a 5-collinear.
                target = rng.choice(three)
                on_line = groups[target]
                used_now = set(pts)
                cands = _candidates_on_line(target, on_line, pts, used_now)
                if cands:
                    new_pt = rng.choice(cands)
                    # choose which off-line point to replace; prefer one that is
                    # NOT on any other 3-line to avoid breaking lines.
                    on_any3 = set()
                    for k, v in groups.items():
                        if len(v) == 3:
                            on_any3 |= v
                    repl = [i for i in range(_N) if i not in on_line and i not in on_any3]
                    if not repl:
                        repl = [i for i in range(_N) if i not in on_line]
                    if repl:
                        idx = rng.choice(repl)
                        old_pt = pts[idx]
                        ns, nh5 = scorer.move(idx, new_pt)
                        if not nh5:
                            delta = ns - cur_score
                            if delta > 0 or rng.random() < math.exp(delta / T):
                                scorer.commit(idx, new_pt)
                                used.discard(old_pt)
                                used.add(new_pt)
                                pts[idx] = new_pt
                                cur_score = ns
                                accepted = True
                                if cur_score > best_score:
                                    best_score = cur_score
                                    best = [p for p in pts]
                                    no_improve = 0
                                else:
                                    no_improve += 1
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
        elif mt < 0.85:
            # Swap two points.
            i, j = rng.sample(range(_N), 2)
            oi, oj = pts[i], pts[j]
            ns, nh5 = scorer.move(i, oj)
            if nh5:
                no_improve += 1
            else:
                # also need j -> oi; approximate by two-step
                ns2, nh5b = scorer.move(j, oi)
                if nh5b:
                    no_improve += 1
                else:
                    delta = ns2 - cur_score
                    if delta > 0 or rng.random() < math.exp(delta / T):
                        # commit both
                        scorer.commit(i, oj)
                        used.discard(oi); used.add(oj)
                        pts[i] = oj
                        scorer.commit(j, oi)
                        used.discard(oj); used.add(oi)
                        pts[j] = oi
                        cur_score = ns2
                        accepted = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best = [p for p in pts]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
        else:
            # Perturb a random point.
            idx = rng.randint(0, _N - 1)
            old_pt = pts[idx]
            step = rng.randint(1, 25)
            nx = old_pt[0] + rng.randint(-step, step)
            ny = old_pt[1] + rng.randint(-step, step)
            if abs(nx) > _LIMIT or abs(ny) > _LIMIT:
                continue
            new_pt = (nx, ny)
            if new_pt in used:
                continue
            ns, nh5 = scorer.move(idx, new_pt)
            if not nh5:
                delta = ns - cur_score
                if delta > 0 or rng.random() < math.exp(delta / T):
                    scorer.commit(idx, new_pt)
                    used.discard(old_pt)
                    used.add(new_pt)
                    pts[idx] = new_pt
                    cur_score = ns
                    accepted = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best = [p for p in pts]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
        T = max(T_min, T * cooling)

    return [list(p) for p in best], best_score

def _grid_seed(rng, perturb):
    pts = [[x, y] for x in range(4) for y in range(4)]
    if perturb:
        for _ in range(rng.randint(8, 18)):
            idx = rng.randint(0, _N - 1)
            nx = pts[idx][0] + rng.randint(-25, 25)
            ny = pts[idx][1] + rng.randint(-25, 25)
            if abs(nx) > _LIMIT or abs(ny) > _LIMIT:
                continue
            cand = [nx, ny]
            if any(cand == p for p in pts):
                continue
            pts[idx] = cand
    return pts

def construct():
    start = time.time()
    deadline = start + 145  # safety margin under 150 s
    rng = random.Random(20240517)

    base = _grid_seed(rng, perturb=False)
    s, h5 = _full_score(base)
    best_points = [p[:] for p in base]
    best_score = 0 if h5 else s

    restarts = 12
    for r in range(restarts):
        if time.time() > deadline - 25:
            break
        seed = _grid_seed(rng, perturb=(r > 0))
        s, h5 = _full_score(seed)
        if h5:
            continue
        opt_pts, opt_score = _local_search(seed, deadline, rng)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts

    # Final verification with full recount.
    final_score, final_h5 = _full_score(best_points)
    if final_h5:
        # Fallback to a clean base grid.
        base = _grid_seed(random.Random(1), perturb=False)
        bs, bh5 = _full_score(base)
        if not bh5:
            best_points = base
            best_score = bs
    return best_points
# EVOLVE-BLOCK-END
