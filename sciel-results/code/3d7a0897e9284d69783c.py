import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 80

    def compute_score(pts):
        n = len(pts)
        counter = Counter()
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    d = abs(pts[i][0] * (pts[j][1] - pts[k][1]) + pts[j][0] * (pts[k][1] - pts[i][1]) + pts[k][0] * (pts[i][1] - pts[j][1]))
                    if d > 0:
                        counter[d] += 1
        if not counter:
            return 0
        return max(counter.values())

    candidates = []
    # 3 parallel lines, 10 points each
    candidates.append([(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)])
    # 2 parallel lines, 15 points each
    candidates.append([(x, 0) for x in range(15)] + [(x, 1) for x in range(15)])
    # 4 parallel lines
    candidates.append([(x, 0) for x in range(8)] + [(x, 1) for x in range(8)] + [(x, 2) for x in range(7)] + [(x, 3) for x in range(7)])
    # 5x6 grid
    candidates.append([(x, y) for x in range(6) for y in range(5)])
    # Sheared grid
    candidates.append([(x, y + x // 2) for x in range(6) for y in range(5)])
    # 3 lines with different spacing
    candidates.append([(2*x, 0) for x in range(10)] + [(2*x, 1) for x in range(10)] + [(2*x, 2) for x in range(10)])

    best_pts = None
    best_score = 0
    for c in candidates:
        if time.time() > deadline:
            break
        s = compute_score(c)
        if s > best_score:
            best_score = s
            best_pts = list(c)

    if best_pts is None:
        return candidates[0]

    # Local search with small neighborhood
    for _ in range(5):
        if time.time() > deadline:
            break
        improved = False
        for i in range(30):
            if time.time() > deadline:
                break
            old = best_pts[i]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = old[0] + dx, old[1] + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    if (nx, ny) == old:
                        continue
                    if (nx, ny) in best_pts:
                        continue
                    best_pts[i] = (nx, ny)
                    s = compute_score(best_pts)
                    if s > best_score:
                        best_score = s
                        improved = True
                    else:
                        best_pts[i] = old
        if not improved:
            break

    return best_pts
