"""Search for 16 integer points maximizing exact-4-collinear lines."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_DEADLINE = 140.0
_START = None


def _count_four(pts):
    n = len(pts)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            g = abs(dx)
            if dy != 0:
                g = g if g == 0 else _gcd(g, abs(dy))
            else:
                g = g if g else 1
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            # normalize intercept
            a, b = -dy, dx
            c = a * x1 + b * y1
            g2 = _gcd(_gcd(abs(a), abs(b)), abs(c))
            if g2 == 0:
                continue
            a //= g2
            b //= g2
            c //= g2
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in lines:
                lines[key] = {i, j}
            else:
                lines[key].add(i)
                lines[key].add(j)
    cnt = 0
    for key, S in lines.items():
        if len(S) == 4:
            cnt += 1
    return cnt


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _has_five(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            g = abs(dx)
            if dy != 0:
                g = g if g == 0 else _gcd(g, abs(dy))
            else:
                g = g if g else 1
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            a, b = -dy, dx
            c = a * x1 + b * y1
            g2 = _gcd(_gcd(abs(a), abs(b)), abs(c))
            if g2 == 0:
                continue
            a //= g2
            b //= g2
            c //= g2
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a, b, c = -a, -b, -c
            cnt = 0
            for k in range(n):
                if a * pts[k][0] + b * pts[k][1] + c == 0:
                    cnt += 1
                    if cnt >= 5:
                        return True
    return False


def _init_grid():
    return [[x, y] for x in range(4) for y in range(4)]


def construct():
    global _START, _DEADLINE
    if _START is None:
        _START = time.time()

    best = _init_grid()
    best_score = _count_four(best)

    # Greedy single-point move search
    while time.time() < _DEADLINE:
        improved = False
        n = len(best)
        # For each point, try moving it to candidate positions
        for idx in range(n):
            if time.time() > _DEADLINE:
                break
            x0, y0 = best[idx]
            # Candidate positions: neighbors in a small window + some random-ish
            tried = set()
            # Try small moves around current position
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if nx in tried or ny in tried:
                        continue
                    tried.add(nx)
                    tried.add(ny)
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    cand = best[:]
                    cand[idx] = [nx, ny]
                    # distinctness
                    if len(set(tuple(p) for p in cand)) != 16:
                        continue
                    if _has_five(cand):
                        continue
                    s = _count_four(cand)
                    if s > best_score:
                        best = cand
                        best_score = s
                        improved = True
                        break
                else:
                    continue
                break
        # If no improvement, try a random perturbation
        if not improved:
            import random
            random.seed(int(time.time() * 1000) % 100000)
            # Random restart: perturb a few points
            cand = best[:]
            for _ in range(3):
                idx = random.randrange(16)
                cx, cy = cand[idx]
                cand[idx] = [cx + random.randint(-5, 5), cy + random.randint(-5, 5)]
            if len(set(tuple(p) for p in cand)) == 16 and all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in cand) and not _has_five(cand):
                s = _count_four(cand)
                if s > best_score:
                    best = cand
                    best_score = s

    return best
# EVOLVE-BLOCK-END
