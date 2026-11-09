# EVOLVE-BLOCK-START
import time
import numpy as np

# ---------------- Corridor (closed) ----------------
# {x<=0, 0<=y<=1, 0<=z<=1} U {0<=x<=1, 0<=y<=5, 0<=z<=1} U {0<=x<=1, 5<=y<=6, z>=0}
def _in_corridor(x, y, z):
    a = (x <= 0.0) & (y >= 0.0) & (y <= 1.0) & (z >= 0.0) & (z <= 1.0)
    b = (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 5.0) & (z >= 0.0) & (z <= 1.0)
    c = (x >= 0.0) & (x <= 1.0) & (y >= 5.0) & (y <= 6.0) & (z >= 0.0)
    return a | b | c

def _in_corridor_scalar(x, y, z):
    if x <= 0.0 and 0.0 <= y <= 1.0 and 0.0 <= z <= 1.0:
        return True
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 5.0 and 0.0 <= z <= 1.0:
        return True
    if 0.0 <= x <= 1.0 and 5.0 <= y <= 6.0 and 0.0 <= z:
        return True
    return False

# ---------------- Grid ----------------
GX = np.linspace(-4.0, 0.0, 81)
GY = np.linspace(0.0, 1.0, 21)
GZ = np.linspace(0.0, 1.0, 21)
NX, NY, NZ = GX.size, GY.size, GZ.size
STEP = 0.05

# Block (i,j,k) = points with x in [i, i+3], y in [j, j+3], z in [k, k+3]
# Number of 4x4x4 blocks
NBX, NBY, NBZ = NX - 3, NY - 3, NZ - 3

# Precompute per-point block membership: for each point, list of blocks it belongs to
# point (i,j,k) belongs to blocks (bi,bj,bk) where bi in [i-3, i], etc.
# For survival we need, for each block, whether ALL its 64 points survive.
# We'll track survival per point as a bool array (NX,NY,NZ), then reduce.

def _point_blocks(i, j, k):
    """Return list of (bi,bj,bk) block indices containing point (i,j,k)."""
    out = []
    for bi in range(max(0, i - 3), min(i + 1, NBX)):
        for bj in range(max(0, j - 3), min(j + 1, NBY)):
            for bk in range(max(0, k - 3), min(k + 1, NBZ)):
                out.append((bi, bj, bk))
    return out

# ---------------- Rotation helpers ----------------
def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def _rotation_matrix(yaw, pitch, roll):
    # R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    return _rot_x(roll) @ _rot_y(pitch) @ _rot_z(yaw)

# ---------------- Evaluator simulation ----------------
# The evaluator:
#  - prepends (-3,0,0,0,0,0), appends (0,6,7,90,90,90)
#  - inserts linearly interpolated poses so each step moves at most 0.05 (Euclidean)
#    and changes each angle by at most 2 deg.
#  - each step: x move, y move, z move, then rotation about a SINGLE axis
#    from old to new orientation.
#  - a grid point survives if in corridor at first pose and after each move,
#    AND the axis-aligned box bounding its rotation arc lies in the corridor.
#  - volume = 0.05^3 * (# of 4x4x4 blocks whose points all survive)

# We densify the path to satisfy the step constraints, then simulate.

def _densify(path):
    """Insert linearly interpolated poses so consecutive poses differ by <=0.05
    in position (Euclidean) and <=2 deg in each angle. Returns list of poses."""
    poses = [(-3.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    for p in path:
        poses.append(tuple(float(v) for v in p))
    poses.append((0.0, 6.0, 7.0, 90.0, 90.0, 90.0))

    # Now densify between consecutive poses
    out = [poses[0]]
    for a, b in zip(poses[:-1], poses[1:]):
        # position distance
        dp = np.hypot(b[0]-a[0], b[1]-a[1], b[2]-a[2])
        # angle max delta
        da = max(abs(b[3]-a[3]), abs(b[4]-a[4]), abs(b[5]-a[5]))
        n_pos = int(np.ceil(dp / 0.05))
        n_ang = int(np.ceil(da / 2.0))
        n = max(n_pos, n_ang, 1)
        for t in range(1, n+1):
            f = t / n
            out.append(tuple(a[i] + (b[i]-a[i]) * f for i in range(6)))
    return out

def _simulate(path):
    """Simulate the evaluator. Return number of surviving 4x4x4 blocks."""
    dens = _densify(path)
    # initial pose
    x0, y0, z0, yaw0, pitch0, roll0 = dens[0]
    R0 = _rotation_matrix(np.radians(yaw0), np.radians(pitch0), np.radians(roll0))
    t0 = np.array([x0, y0, z0])

    # First pose check: all points must be in corridor
    X = np.tile(GX[:, None, None], (1, NY, NZ))
    Y = np.tile(GY[None, :, None], (NX, 1, NZ))
    Z = np.tile(GZ[None, None, :], (NX, NY, 1))
    P = np.stack([X, Y, Z], axis=-1)  # (NX,NY,NZ,3)

    # R0 @ p + t0
    def place(R, t):
        # (NX,NY,NZ,3)
        return np.einsum('ij,...j->...i', R, P) + t

    def in_corr(pos):
        return _in_corridor(pos[..., 0], pos[..., 1], pos[..., 2])

    surv = in_corr(place(R0, t0))  # bool (NX,NY,NZ)

    for a, b in zip(dens[:-1], dens[1:]):
        ax, ay, az, ayaw, apitch, aroll = a
        bx, by, bz, byaw, bpitch, broll = b
        # step: x move, y move, z move, then rotation about single axis
        # We'll do: translate to (bx,ay,az) then (bx,by,az) then (bx,by,bz),
        # then rotate. For rotation, the evaluator rotates about a single axis
        # from old to new orientation. We approximate: rotate about the axis
        # with the largest angle change, but the arc bounding box must be in corridor.
        # To be safe and match "single axis", we'll rotate about z (yaw) first,
        # then y (pitch), then x (roll) — but the contract says a SINGLE axis.
        # The evaluator picks one axis. We'll emulate the most conservative:
        # rotate about the axis whose angle changes, using the arc bounding box.
        R_prev = _rotation_matrix(np.radians(ayaw), np.radians(apitch), np.radians(aroll))
        t_prev = np.array([ax, ay, az])

        # x move
        t1 = np.array([bx, ay, az])
        surv &= in_corr(place(R_prev, t1))
        # y move
        t2 = np.array([bx, by, az])
        surv &= in_corr(place(R_prev, t2))
        # z move
        t3 = np.array([bx, by, bz])
        surv &= in_corr(place(R_prev, t3))

        # rotation about single axis from old to new orientation
        R_new = _rotation_matrix(np.radians(byaw), np.radians(bpitch), np.radians(broll))
        # Determine which axis changed most
        dyaw = abs(byaw - ayaw)
        dpitch = abs(bpitch - apitch)
        droll = abs(broll - aroll)
        # pick axis with max change; if all ~0, skip
        if max(dyaw, dpitch, droll) < 1e-9:
            surv &= in_corr(place(R_new, t3))
            continue
        # rotate about that axis. We sample the arc and take bounding box.
        # For a rotation about a single axis, the point traces an arc; its
        # axis-aligned bounding box is the convex hull of arc endpoints plus
        # the arc extremes. We approximate by sampling the arc densely.
        # The rotation about a fixed axis from R_prev to R_new: we need the
        # actual rotation. Since only one axis changes, R_new = R_axis(delta) @ R_prev
        # (or R_prev @ R_axis(delta) depending on convention). The contract:
        # R = Rx(roll) Ry(pitch) Rz(yaw). Changing yaw means R_new = Rx(roll) Ry(pitch) Rz(yaw_new).
        # So the rotation is applied on the right in the yaw case.
        # For simplicity and safety, we sample the full set of intermediate
        # orientations between R_prev and R_new by interpolating the single
        # changing angle, and check the bounding box of the point's arc.
        if dyaw >= dpitch and dyaw >= droll:
            # yaw changes: interpolate yaw
            ys = np.linspace(ayaw, byaw, 17)
        elif dpitch >= droll:
            ys = None
            ps = np.linspace(apitch, bpitch, 17)
        else:
            ys = None
            ps = None
            rs = np.linspace(aroll, broll, 17)
        # Build intermediate rotation matrices
        mats = []
        if ys is not None:
            for yv in ys:
                mats.append(_rotation_matrix(np.radians(yv), np.radians(apitch), np.radians(aroll)))
        elif ps is not None:
            for pv in ps:
                mats.append(_rotation_matrix(np.radians(ayaw), np.radians(pv), np.radians(aroll)))
        else:
            for rv in rs:
                mats.append(_rotation_matrix(np.radians(ayaw), np.radians(apitch), np.radians(rv)))
        # For each point, compute its positions across the arc; bounding box must be in corridor.
        # We check that the min and max of each coordinate over the arc are within corridor bounds
        # by sampling (corridor is a union of boxes, so we check each sampled point is in corridor,
        # which is conservative and matches "bounding box lies in corridor" approximately).
        # Actually the contract: axis-aligned box bounding the rotation arc lies in corridor.
        # We check all sampled arc points are in corridor (a sufficient condition for the
        # box to be in corridor when corridor is convex-ish; here we use it as proxy).
        for M in mats:
            surv &= in_corr(place(M, t3))
    # Count surviving 4x4x4 blocks
    # A block (bi,bj,bk) survives if all points (i in bi..bi+3, j in bj..bj+3, k in bk..bk+3) survive
    count = 0
    for bi in range(NBX):
        for bj in range(NBY):
            for bk in range(NBZ):
                if surv[bi:bi+4, bj:bj+4, bk:bk+4].all():
                    count += 1
    return count

# ---------------- Seed path (from current program) ----------------
def _seed_path():
    return np.array([
        [0.939609406, 0.00998329527, -0.00407296132, 0.0886163961, 0.0514178831, 0.789799574],
        [0.976853949, 0.0843900243, -0.00187236014, 2.22743402, 0.0275244655, 0.595653008],
        [1.01563377, 0.158600496, 9.83065911e-05, 4.35572642, -0.00995231395, 0.363970701],
        [1.08493385, 0.307755287, 0.00404501406, 8.62938219, -0.000435990906, -0.335077274],
        [1.12726319, 0.436994587, 0.0165792562, 12.3522097, -0.444355561, -1.09977761],
        [1.15035596, 0.509864783, 0.0246729498, 14.4926365, -0.703150883, -1.77573027],
        [1.20020002, 0.674818962, 0.0413920989, 19.2858257, -1.46824802, -3.24301616],
        [1.22921833, 0.78032244, 0.0508392229, 22.3604231, -2.05934965, -4.19183977],
        [1.25656232, 0.886955923, 0.0622851979, 25.4006424, -2.79233605, -4.94301221],
        [1.28161834, 0.9941859, 0.0713168622, 28.4982429, -3.53439006, -5.72810662],
        [1.30160821, 1.12543063, 0.0720451855, 32.0688854, -3.9280592, -5.56488257],
        [1.32438826, 1.32927434, 0.0724034864, 37.7041144, -4.47472162, -5.17490305],
        [1.33399008, 1.48086681, 0.0758177845, 41.9024902, -4.92138226, -4.72192269],
        [1.33698854, 1.59471234, 0.075956822, 45.1249666, -5.16592127, -4.32429483],
        [1.33554955, 1.70831674, 0.0733949585, 48.4117195, -5.24474942, -3.95318659],
        [1.33301955, 1.77848192, 0.0730557819, 50.4395679, -5.29391038, -3.60268508],
        [1.32299384, 1.89675169, 0.066224521, 54.0094721, -5.07170268, -3.10511721],
        [1.31609554, 1.95530642, 0.0645580036, 55.7906035, -4.95025879, -2.75163185],
        [1.29699464, 2.0647489, 0.0528964026, 59.2092384, -4.24840518, -2.08208301],
        [1.28467533, 2.12603596, 0.0465722033, 61.19788, -3.85394213, -1.77381137],
        [1.27612175, 2.16246336, 0.0429218547, 62.3960278, -3.60313312, -1.60391888],
        [1.25918263, 2.22238115, 0.0352027157, 64.3876548, -2.94645508, -1.14315788],
        [1.21195969, 2.37272154, 0.0185866337, 69.5501344, -1.44731306, -0.351817856],
        [1.17367747, 2.46947271, 0.00787738711, 73.3278162, -0.59130695, -0.204315921],
        [1.13343824, 2.56892736, 0.000240804007, 77.2294036, 0.0742535282, -0.0118797041],
        [1.10227764, 2.63702207, -0.000813164259, 80.2921591, 0.325638556, -0.00909491031],
        [1.08115946, 2.67872855, -0.0019446761, 82.2507337, 0.456149966, 0.00923364344],
        [1.06864394, 2.70464551, -0.00209793986, 83.4698923, 0.395788406, -0.00251260641],
        [0.993054246, 2.83888423, -0.00277991124, 90.0331465, 0.651004673, -0.0200027208],
        [0.992050998, 3.42584708, -0.00292617372, 90.0630824, 0.648340038, -0.0336076805],
        [0.990524577, 4.81296934, -0.00378757857, 90.0846127, 0.741107384, -0.0439029391],
        [0.988680148, 5.99738121, -0.00451717463, 90.104646, 0.883485061, -0.041030785],
        [0.957624286, 5.9528988, 0.0102741306, 90.1327839, 4.19080459, 1.12928446],
        [0.942862105, 5.99368058, 0.0162024942, 90.1288679, 5.76538787, 1.62560793],
        [0.931969027, 5.99412693, 0.020619344, 90.1263482, 6.90261077, 1.98875845],
        [0.904773557, 6.00941729, 0.0313747872, 90.128735, 9.61483109, 2.83144875],
        [0.8858989, 6.01830162, 0.0381113604, 90.135911, 11.4508584, 3.34953358],
        [0.863540853, 6.02619691, 0.04487657, 90.1598001, 13.5181743, 3.88721021],
        [0.836463078, 6.03765479, 0.0528267196, 90.1967295, 15.9298717, 4.48186113],
        [0.804731182, 6.04627379, 0.0625195903, 90.2336098, 18.6990514, 5.14505668],
        [0.788296376, 6.05202369, 0.0675622892, 90.2479959, 20.1264906, 5.47138803],
        [0.778474876, 6.05538925, 0.0706050032, 90.2577268, 20.9723911, 5.66341348],
        [0.770888811, 6.05788046, 0.0730068444, 90.2657676, 21.6229248, 5.80874653],
        [0.764648333, 6.0599328, 0.0749383027, 90.2726388, 22.1532161, 5.92642676],
        [0.748753549, 6.06467644, 0.0801606053, 90.2892425, 23.4927925, 6.21910882],
        [0.737683416, 6.06785126, 0.0838714978, 90.3001161, 24.4180728, 6.42046395],
        [0.726094096, 6.07124306, 0.0879934282, 90.3107738, 25.3815663, 6.631397],
        [0.708478913, 6.07645575, 0.094580537, 90.324703, 26.8427434, 6.95357494],
        [0.685450204, 6.08320452, 0.103688262, 90.3410972, 28.7527373, 7.37124273],
        [0.660511655, 6.09058578, 0.114647767, 90.341297, 30.8426719, 7.83842912],
        [0.631106763, 6.09898622, 0.127562353, 90.3399229, 33.2683226, 8.35622932],
        [0.60239252, 6.10718508, 0.141197449, 90.3187387, 35.6585372, 8.87446837],
        [0.574685996, 6.1149268, 0.154862481, 90.2890178, 37.982628, 9.36856002],
        [0.549279838, 6.12170874, 0.168073709, 90.2538308, 40.1148352, 9.81523017],
        [0.523487025, 6.12859076, 0.182443691, 90.2080219, 42.279737, 10.2799932],
        [0.498121776, 6.13520082, 0.197968844, 90.1482985, 44.4163048, 10.7608292],
        [0.476465121, 6.14082429, 0.2120163, 90.0844905, 46.2626926, 11.1825901],
        [0.456153588, 6.14607051, 0.226245551, 90.0101313, 48.012554, 11.601748],
        [0.433791775, 6.15190581, 0.243138907, 89.9157178, 49.954221, 12.084816],
        [0.404025852, 6.15902259, 0.266186906, 89.7734311, 52.548863, 12.7293228],
        [0.366005234, 6.16774759, 0.297031124, 89.5738271, 55.893128, 13.5723105],
        [0.321312876, 6.17691876, 0.335574819, 89.2922705, 59.8692011, 14.594953],
        [0.278427663, 6.18417441, 0.373123513, 88.9719858, 63.7515612, 15.5832162],
        [0.239534769, 6.19553929, 0.413872182, 88.6402231, 67.3322992, 16.6327601],
        [0.204472084, 6.19968082, 0.464675511, 88.3088859, 70.5963626, 17.9436724],
        [0.175394931, 6.21048495, 0.524148773, 88.0051881, 73.3241903, 19.4940659],
        [0.153758289, 6.22132345, 0.583751542, 87.7681793, 75.3731869, 21.0578654],
        [0.126442905, 6.23697046, 0.675335517, 87.4796317, 77.9661162, 23.4337293],
        [0.11557363, 6.25005689, 0.744980177, 87.365021, 79.008173, 25.2702993],
        [0.106313411, 6.26635361, 0.836966776, 87.2610122, 79.9079131, 27.7208667],
        [0.0989070592, 6.2783871, 0.912917517, 87.1845804, 80.6362714, 29.7408798],
        [0.0945573652, 6.28550694, 0.960719372, 87.1536509, 81.0361104, 31.0067456],
        [0.0902778512, 6.29266034, 1.01249647, 87.1182294, 81.451701, 32.3850431],
        [0.0852959027, 6.30150778, 1.08113362, 87.0519944, 81.9245444, 34.2396643],
        [0.083875265, 6.30582796, 1.11958943, 86.977003, 82.1005075, 35.3326516],
        [0.0825523794, 6.30959352, 1.15442156, 86.9227818, 82.2502751, 36.313756],
        [0.0813272459, 6.31280445, 1.18563003, 86.8893308, 82.3738474, 37.1785229],
        [0.0795266549, 6.31701618, 1.22936759, 86.8298912, 82.5573864, 38.4051678],
        [0.0760640114, 6.32390774, 1.30814197, 86.730576, 82.9155599, 40.6273839],
        [0.0733345584, 6.32845988, 1.38417722, 86.6469757, 83.1880754, 42.7737559],
        [0.0728754398, 6.33436689, 1.51928934, 85.9570928, 83.321603, 47.2189208],
        [0.0740241462, 6.33651625, 1.67086132, 85.1992673, 83.4017941, 52.295749],
        [0.0709732742, 6.32973557, 1.82119648, 84.427344, 83.8384848, 57.4161351],
        [0.0689950542, 6.32296873, 1.89917235, 84.1030756, 84.0687341, 60.0778089],
        [0.0654533403, 6.31403661, 1.97470221, 83.8065972, 84.4296311, 62.6673579],
        [0.0598040781, 6.30238868, 2.04639195, 83.482155, 84.9583795, 65.1956567],
        [0.0532315605, 6.28848325, 2.11585078, 83.1684777, 85.5618308, 67.7111146],
        [0.0467194205, 6.27295621, 2.18521241, 82.8588875, 86.1559337, 70.2420111],
        [0.0211915399, 6.21382682, 2.3716358, 82.5672227, 88.3741932, 76.9350513],
        [0.0119033827, 6.16998589, 2.48828327, 83.7295105, 89.1215994, 80.3298032],
        [0.0114904543, 6.16373556, 2.50403291, 83.888837, 89.1782177, 80.7917836],
        [0.00847440903, 6.15720704, 2.52053877, 84.0621014, 89.3189944, 81.2648114],
        [0.0054345193, 6.12168354, 2.60523315, 84.9795728, 89.7896324, 83.7983903],
        [0.00500718906, 6.08778028, 2.67536568, 85.9496099, 89.8305811, 86.1593636],
        [0.00496310192, 6.07475445, 2.70089071, 86.2819807, 89.9205737, 87.0030543],
        [0.00447449792, 6.05605429, 2.7350364, 86.757467, 89.802935, 88.1996353],
        [0.00297530192, 6.02087628, 2.79766649, 87.459475, 90.1346094, 90.5955514],
        [0.000953058079, 5.9970912, 2.84427868, 87.6265515, 90.27883, 92.3854717],
        [-0.00231375478, 5.99154793, 4.38250817, 87.3674843, 90.6652632, 92.7094979],
    ], dtype=np.float64)

def moving_sofa_3d_path() -> np.ndarray:
    start = time.time()
    deadline = start + 540.0  # safety margin before 630s kill

    best = _seed_path()
    best_score = _simulate(best)

    # Coordinate-descent style local search: perturb each pose slightly, keep if better.
    rng = np.random.default_rng(12345)
    n = best.shape[0]
    # Per-coordinate step sizes (translation in meters, angles in degrees)
    pos_step = 0.01
    ang_step = 0.3

    improved = True
    while improved and (time.time() < deadline - 30.0):
        improved = False
        for it in range(n):
            if time.time() > deadline - 30.0:
                break
            for trial in range(4):
                cand = best.copy()
                # random small perturbation of one pose
                di = rng.integers(0, 6)
                if di < 3:
                    cand[it, di] += rng.normal(0.0, pos_step)
                else:
                    cand[it, di] += rng.normal(0.0, ang_step)
                s = _simulate(cand)
                if s > best_score:
                    best = cand
                    best_score = s
                    improved = True
                    break
    return best

# EVOLVE-BLOCK-END
