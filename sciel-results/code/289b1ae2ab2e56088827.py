# EVOLVE-BLOCK-START
import time
import random
import numpy as np

P = 7
K = 3  # alpha^2 = 3 (quadratic non-residue mod 7)
Q = P * P  # 49
NPTS = Q * Q  # 2401


def _add_f(a, b, c, d):
    return ((a + c) % P, (b + d) % P)


def _mul_f(a, b, c, d):
    # (a + b*alpha)*(c + d*alpha) = (ac - bd*K) + (ad + bc)*alpha
    return ((a * c - b * d * K) % P, (a * d + b * c) % P)


def _pid(xa, xb, ya, yb):
    return ((xa * P + xb) * P + ya) * P + yb


def _unpid(pid):
    yb = pid % P
    ya = (pid // P) % P
    xb = (pid // (P * P)) % P
    xa = (pid // (P * P * P)) % P
    return xa, xb, ya, yb


# Build all points
POINTS = []
for xa in range(P):
    for xb in range(P):
        for ya in range(P):
            for yb in range(P):
                POINTS.append((xa, xb, ya, yb))

# Precompute the 50 lines through each point as bitmasks (Python int)
# and the affected list aff[p] = list of (q, Lglobal) for incremental updates.
# We need a global line index for each (p, direction) to maintain cL.
# Actually cL is per line; each line is shared by its 49 points. We'll index lines
# by (p, dir) but that's 2401*50 entries where many represent the same geometric line.
# For cL we need a canonical line id. Build canonical lines: each line has 49 points.
# Total distinct lines = (number of directions)*(number of parallel classes per direction)
# directions = 50 (49 slopes + vertical). Each direction has 49 parallel lines. Total 2450 lines.

# Build canonical lines: for each direction, for each offset, the 49 points.
# Direction as a step (dx, dy) in F_49 pairs. Vertical: dx=(0,0),dy=(1,0). Slope m: dx=(1,0), dy=m.

DIRS = []
# vertical
DIRS.append(((0, 0), (1, 0)))
for ma in range(P):
    for mb in range(P):
        DIRS.append(((1, 0), (ma, mb)))
# Note: slope (1,0) i.e. m=0 gives dx=(1,0),dy=(0,0) -> horizontal lines (y const). Fine, distinct from vertical.

# For each direction, generate the 49 lines by choosing a base point and stepping; to get all
# parallel lines, fix a representative: for direction (dx,dy), lines are cosets. Use base points
# that are "minimal" in the direction. Simpler: for each direction, take all points, group by
# the line they lie on via a coordinate invariant.
# Easier: for each direction, for each point p, the line = {p + t*(dx,dy)}. Assign canonical id =
# the min point id on that line. Build mapping.

line_of = {}  # (pid, diridx) -> canonical line id
canon_points = []  # list of sets of pids

def _line_pts(pid, diridx):
    xa, xb, ya, yb = _unpid(pid)
    dx, dy = DIRS[diridx]
    pts = []
    for ta in range(P):
        for tb in range(P):
            t = (ta, tb)
            tx = _add_f(xa, xb, dx[0] * ta + dx[1] * tb, dx[1] * ta + dx[0] * tb)  # t*dx
            # t*dx where t=(ta,tb), dx=(dxa,dxb): (ta*dxa - tb*dxb*K, ta*dxb + tb*dxa)
            tx = ((xa + (ta * dx[0] - tb * dx[1] * K)) % P, (xb + (ta * dx[1] + tb * dx[0])) % P)
            ty = _add_f(ya, yb, _mul_f(ta, tb, dy[0], dy[1]))
            pts.append(_pid(tx[0], tx[1], ty[0], ty[1]))
    return pts


# Build canonical lines
canon_id = {}
for diridx in range(len(DIRS)):
    seen = set()
    for pid in range(NPTS):
        if pid in seen:
            continue
        pts = _line_pts(pid, diridx)
        key = frozenset(pts)
        if key not in canon_id:
            canon_id[key] = len(canon_id)
        seen.update(pts)

NLINES = len(canon_id)

# For each point, the 50 line ids through it, and the affected (q, lineid) list.
pt_lines = [None] * NPTS
for pid in range(NPTS):
    lset = []
    for diridx in range(len(DIRS)):
        pts = _line_pts(pid, diridx)
        key = frozenset(pts)
        lset.append(canon_id[key])
    pt_lines[pid] = lset

# Precompute aff[pid] = list of (q, lineid) for q on a line with pid, q != pid
aff = [None] * NPTS
for pid in range(NPTS):
    lst = []
    for lineid in pt_lines[pid]:
        # need points on this line; recompute
        # we can store line point lists
        pass

# Store line point lists
LINE_PTS = []
for key, lid in canon_id.items():
    LINE_PTS.append(sorted(key))
# map lid -> list
lid_pts = [None] * NLINES
for key, lid in canon_id.items():
    lid_pts[lid] = sorted(key)

for pid in range(NPTS):
    lst = []
    for lineid in pt_lines[pid]:
        for q in lid_pts[lineid]:
            if q != pid:
                lst.append((q, lineid))
    aff[pid] = lst

# Also for each point, own bit
OWN_BIT = [1 << pid for pid in range(NPTS)]


def _greedy_random(rng, deadline):
    # count[q] = number of witness lines for q
    count = [len(DIRS)] * NPTS  # 50 each initially (C empty)
    cL = [0] * NLINES
    C = [False] * NPTS
    csize = 0
    while time.time() < deadline:
        order = list(range(NPTS))
        rng.shuffle(order)
        added_any = False
        for pid in order:
            if C[pid]:
                continue
            # simulate adding pid
            # for each (q, lineid) in aff[pid], determine if q loses a witness
            ok = True
            deltas = []
            for (q, lineid) in aff[pid]:
                cl = cL[lineid]
                # q loses witness via this line iff (cl==0 and not C[q]) or (cl==1 and C[q])
                if (cl == 0 and not C[q]) or (cl == 1 and C[q]):
                    if count[q] <= 1:
                        ok = False
                        break
                    deltas.append(q)
            if not ok:
                continue
            # apply
            for q in deltas:
                count[q] -= 1
            for lineid in pt_lines[pid]:
                cL[lineid] += 1
            C[pid] = True
            csize += 1
            added_any = True
        if not added_any:
            break
    return csize, C


def nikodym_set() -> np.ndarray:
    deadline = time.time() + 500
    best_C = None
    best_size = -1
    rng = random.Random(12345)
    while time.time() < deadline:
        sz, C = _greedy_random(rng, deadline)
        if sz > best_size:
            best_size = sz
            best_C = C
    if best_C is None:
        best_C = [False] * NPTS
        best_size = 0
    # Build N = all points not in C
    pts = []
    for pid in range(NPTS):
        if not best_C[pid]:
            xa, xb, ya, yb = _unpid(pid)
            pts.append([[xa, xb], [ya, yb]])
    return np.array(pts, dtype=np.int64)


# EVOLVE-BLOCK-END
