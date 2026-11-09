"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

# Fast evaluation: returns (r, violating_index, violating_other)
def _eval(pts):
    """Compute r = min(pairs/2, walls). Return (r, i, j) where (i,j) is the
    tightest constraint; j=-1 means wall constraint for point i."""
    # Wall clearance
    wall = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    # Pairwise min distance via KDTree (only nearest neighbor matters)
    tree = cKDTree(pts)
    d, nn = tree.query(pts, k=2)  # k=2 -> self + nearest
    pair = d[:, 1]
    # Find global min
    wp = int(np.argmin(wall))
    wi = int(np.argmin(pair))
    if wall[wp] <= pair[wi]:
        return wall[wp], wp, -1
    else:
        return pair[wi], wi, int(nn[wi, 1])


def _grid_init():
    """Good initial grid: 15x15 = 225 slots, take 200, centered."""
    m = 15
    # 200 = 15*13 + 5 ; use rows of 15 except last 2 rows have fewer
    # Simpler: build 15x15 and drop 25 from corners/edges
    xs = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in xs for x in xs])  # 225
    # Drop 25 points: remove the 25 with largest |x|+|y| (corners first)
    # Actually remove to keep it symmetric-ish: remove last 25 in row-major
    pts = pts[:N]
    return pts.astype(float)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    np.random.seed(42)

    pts = _grid_init()
    # small perturbation to break symmetry
    pts += np.random.randn(N, 2) * 0.0005
    pts = np.clip(pts, -0.4999, 0.4999)

    r, _, _ = _eval(pts)
    best_pts = pts.copy()
    best_r = r
    cur_pts = pts.copy()
    cur_r = r

    T = 0.004
    T_min = 1e-7
    cooling = 0.9995

    while time.time() < deadline - 30:  # reserve 30s for polish
        T = max(T, T_min)
        # batch of moves
        for _ in range(200):
            if time.time() > deadline - 30:
                break
            idx = np.random.randint(N)
            # propose move: away from tightest neighbor + random jitter
            tree = cKDTree(cur_pts)
            d, nn = tree.query(cur_pts[idx], k=2)
            nn_idx = int(nn[1])
            direction = cur_pts[idx] - cur_pts[nn_idx]
            nrm = np.linalg.norm(direction)
            if nrm > 1e-12:
                direction /= nrm
            else:
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
            step = T * 2.0
            move = direction * step + np.random.randn(2) * T * 0.3
            new_pts = cur_pts.copy()
            new_pts[idx] = cur_pts[idx] + move
            new_pts[idx] = np.clip(new_pts[idx], -0.4999, 0.4999)
            new_r, _, _ = _eval(new_pts)
            delta = new_r - cur_r
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                cur_pts = new_pts
                cur_r = new_r
                if cur_r > best_r:
                    best_r = cur_r
                    best_pts = cur_pts.copy()
        T *= cooling

    # ---- Final high-precision polish: direct optimization of r ----
    # Maximize r subject to all pair dist >= 2r and wall >= r.
    # Use a simple projected ascent on r with repair moves.
    pts = best_pts.copy()
    r = best_r
    for it in range(20000):
        if time.time() > deadline - 2:
            break
        # find tightest constraint
        r_cur, i, j = _eval(pts)
        if r_cur < r - 1e-12:
            # violated; repair by moving point i away
            if j == -1:
                # wall: push toward center along the tight axis
                axis = 0 if (0.5 - abs(pts[i, 0])) <= (0.5 - abs(pts[i, 1])) else 1
                sign = 1.0 if pts[i, axis] > 0 else -1.0
                pts[i, axis] -= sign * 0.0005
            else:
                direction = pts[i] - pts[j]
                nrm = np.linalg.norm(direction)
                if nrm > 1e-12:
                    direction /= nrm
                else:
                    direction = np.random.randn(2); direction /= np.linalg.norm(direction)
                pts[i] += direction * 0.0005
            pts[i] = np.clip(pts[i], -0.4999, 0.4999)
            continue
        # feasible: try to increase r by nudging tightest pair apart
        if j == -1:
            axis = 0 if (0.5 - abs(pts[i, 0])) <= (0.5 - abs(pts[i, 1])) else 1
            sign = 1.0 if pts[i, axis] > 0 else -1.0
            pts[i, axis] -= sign * 0.0002
        else:
            direction = pts[i] - pts[j]
            nrm = np.linalg.norm(direction)
            if nrm > 1e-12:
                direction /= nrm
                pts[i] += direction * 0.0002
                pts[j] -= direction * 0.0002
            else:
                pts[i] += np.random.randn(2) * 0.0002
        pts = np.clip(pts, -0.4999, 0.4999)
        r_new, _, _ = _eval(pts)
        if r_new > r:
            r = r_new
            best_r = r
            best_pts = pts.copy()

    best_pts = np.clip(best_pts, -0.5, 0.5)
    return best_pts
# EVOLVE-BLOCK-END
