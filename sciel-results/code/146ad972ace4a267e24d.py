import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    p = 7
    start_time = time.time()
    deadline = 600  # safety margin before 630s kill

    n_points = 2401

    def point_to_idx(xa, xb, ya, yb):
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb

    def idx_to_point(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return (xa, xb, ya, yb)

    # Precompute all lines for all 50 directions
    # For direction (1, ma, mb): line ID = ((ya - xa*ma - 3*xb*mb) % 7, (yb - xa*mb - xb*ma) % 7)
    # For vertical direction (0, 1): line ID = (xa, xb)

    # Precompute line membership: for each direction, for each line, the 49 points
    # Also precompute: for each point and direction, which line it's on

    all_lines = []  # all_lines[d][l] = array of 49 point indices
    point_line_dir = np.zeros((n_points, 50), dtype=np.int32)

    for d in range(50):
        if d < 49:
            ma = d // 7
            mb = d % 7
        else:
            ma, mb = None, None

        line_to_points = defaultdict(list)
        for idx in range(n_points):
            xa, xb, ya, yb = idx_to_point(idx)
            if d < 49:
                c1 = (ya - xa * ma - 3 * xb * mb) % 7
                c2 = (yb - xa * mb - xb * ma) % 7
                lid = c1 * 7 + c2
            else:
                lid = xa * 7 + xb
            line_to_points[lid].append(idx)
            point_line_dir[idx][d] = lid

        lines_in_dir = []
        for lid in range(49):
            pts = np.array(line_to_points[lid], dtype=np.int32)
            lines_in_dir.append(pts)

        all_lines.append(lines_in_dir)

    # Precompute lines_through_point: for each point, list of (direction, line_id)
    lines_through_point = []
    for idx in range(n_points):
        ltp = []
        for d in range(50):
            lid = point_line_dir[idx][d]
            ltp.append((d, lid))
        lines_through_point.append(ltp)

    # Precompute points_on_line: for each (d, lid), the 49 points
    # Already have all_lines[d][lid]

    # Initial N: all points
    in_N = np.ones(n_points, dtype=bool)

    # Count in_N per line
    n_in_N = np.zeros((50, 49), dtype=np.int32)
    for d in range(50):
        for l in range(49):
            n_in_N[d][l] = np.sum(in_N[all_lines[d][l]])

    # Compute clean_count: for each point, number of directions where the line is clean
    # For point q, direction d is clean iff:
    #   If in_N[q]: n_in_N[d][line(q,d)] == 49
    #   Else: n_in_N[d][line(q,d)] == 48

    def compute_clean_count():
        cc = np.zeros(n_points, dtype=np.int32)
        for d in range(50):
            for l in range(49):
                line_pts = all_lines[d][l]
                n_in = n_in_N[d][l]
                for idx in line_pts:
                    if in_N[idx]:
                        if n_in == 49:
                            cc[idx] += 1
                    else:
                        if n_in == 48:
                            cc[idx] += 1
        return cc

    # Greedy removal: try to remove points from N one by one
    # A point p can be removed if after removal, every point still has a clean direction

    def can_remove(p):
        # Check if removing p from N maintains validity
        # Temporarily remove p
        in_N[p] = False
        affected_lines = {}
        for d in range(50):
            lid = point_line_dir[p][d]
            affected_lines[(d, lid)] = n_in_N[d][lid]
            n_in_N[d][lid] -= 1

        # Check all points
        valid = True
        for q in range(n_points):
            has_clean = False
            for d in range(50):
                lid = point_line_dir[q][d]
                n_in = n_in_N[d][lid]
                if in_N[q]:
                    if n_in == 49:
                        has_clean = True
                        break
                else:
                    if n_in == 48:
                        has_clean = True
                        break
            if not has_clean:
                valid = False
                break

        # Restore
        in_N[p] = True
        for (d, lid), val in affected_lines.items():
            n_in_N[d][lid] = val

        return valid

    # Try removing points in order of index
    for p in range(n_points):
        if time.time() - start_time > deadline:
            break
        if not in_N[p]:
            continue
        if can_remove(p):
            in_N[p] = False
            for d in range(50):
                lid = point_line_dir[p][d]
                n_in_N[d][lid] -= 1

    # Build result
    points = []
    for idx in range(n_points):
        if in_N[idx]:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])

    if not points:
        points = [[[0, 0], [0, 0]]]

    return np.array(points, dtype=np.int64)
