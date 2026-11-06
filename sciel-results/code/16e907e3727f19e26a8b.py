import numpy as np
import time
from itertools import combinations

def _score(points):
    """Calculate the maximum multiplicity of a positive doubled area."""
    points = np.array(points, dtype=np.int64)
    n = len(points)
    counts = {}
    # 4060 triples is small
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Doubl Area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                # Or cross product
                area2 = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area2 > 0:
                    counts[area2] = counts.get(area2, 0) + 1
    if not counts:
        return 0
    return max(counts.values())

def construct():
    """
    Returns 30 distinct integer coordinate pairs in [-30,30]^2.
    Maximizes the largest multiplicity of a single positive doubled area.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin for 90s limit

    candidates = []

    # Structure 1: 3 lines, 10 points each
    # y=0,1,2; x=0..9
    p1 = [(x, y) for y in range(3) for x in range(10)]
    candidates.append(p1)

    # Structure 2: 5 lines, 6 points each
    # y=0..4; x=0..5
    p2 = [(x, y) for y in range(5) for x in range(6)]
    candidates.append(p2)

    # Structure 3: 6 lines, 5 points each
    # y=0..5; x=0..4
    p3 = [(x, y) for y in range(6) for x in range(5)]
    candidates.append(p3)

    # Structure 4: 2 lines, 15 points each
    # y=0,1; x=0..14
    p4 = [(x, y) for y in range(2) for x in range(15)]
    candidates.append(p4)
    
    # Structure 5: 10 lines, 3 points each
    # y=0..9; x=0..2
    p5 = [(x, y) for y in range(10) for x in range(3)]
    candidates.append(p5)

    best_points = p1
    best_score = -1

    for c in candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_points = c
        if time.time() > deadline:
            break

    # If time allows, we could do a local search, but structured grids are usually strong.
    # Given the small number of candidates, we return the best one found.
    return best_points
