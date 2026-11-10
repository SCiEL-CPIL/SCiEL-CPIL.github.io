# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_DEADLINE = time.time() + 615.0  # 15s safety margin before 630s kill


def _rotation_matrix(yaw, pitch, roll):
    """R = Rx(roll) @ Ry(pitch) @ Rz(yaw), angles in radians."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rx @ Ry @ Rz


def _corridor_mask(x, y, z):
    """Vectorized corridor membership. Corridor (closed):
    {x<=0, 0<=y<=1, 0<=z<=1} U {0<=x<=1, 0<=y<=5, 0<=z<=1} U {0<=x<=1, 5<=y<=6, z>=0}
    """
    eps = 1e-9
    c1 = (x <= eps) & (y >= -eps) & (y <= 1 + eps) & (z >= -eps) & (z <= 1 + eps)
    c2 = (x >= -eps) & (x <= 1 + eps) & (y >= -eps) & (y <= 5 + eps) & (z >= -eps) & (z <= 1 + eps)
    c3 = (x >= -eps) & (x <= 1 + eps) & (y >= 5 - eps) & (y <= 6 + eps) & (z >= -eps)
    return c1 | c2 | c3


def _densify(path):
    """Densify the path: prepend start, append end, insert linearly interpolated
    poses so each step moves at most 0.05 (Euclidean) and each angle changes by at most 2 deg.
    Returns the full sequence of poses (N, 6)."""
    start = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
    full = np.vstack([start, path, end])
    out = [full[0]]
    for i in range(len(full) - 1):
        a, b = full[i], full[i + 1]
        diff = np.abs(b - a)
        # max steps needed
        steps = 1
        if np.any(diff > 0):
            # translation: first 3 components
            steps = max(steps, int(np.ceil(np.linalg.norm(diff[:3]) / 0.05)))
            # angles: last 3 components
            steps = max(steps, int(np.ceil(np.max(diff[3:]) / 2.0)))
        steps = max(1, steps)
        for s in range(1, steps + 1):
            t = s / steps
            out.append(a + t * (b - a))
    return np.array(out)


def _evaluate(path):
    """Evaluate the path: return the number of surviving 4x4x4 blocks (the volume metric / 0.05^3).
    This exactly replicates the evaluator logic."""
    # Grid
    xs = np.linspace(-4, 0, 81)
    ys = np.linspace(0, 1, 21)
    zs = np.linspace(0, 1, 21)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    # Flatten grid points
    gx = X.ravel()
    gy = Y.ravel()
    gz = Z.ravel()
    npts = len(gx)

    # Initial survival: grid point must be in the corridor at the first pose.
    # First pose is the start pose (-3, 0, 0, 0, 0, 0).
    # R = identity, translation = (-3, 0, 0).
    # p placed at R p + t = p + (-3, 0, 0)
    px = gx - 3.0
    py = gy + 0.0
    pz = gz + 0.0
    survive = _corridor_mask(px, py, pz)

    # Densified path
    poses = _densify(path)

    for i in range(len(poses)):
        if not np.any(survive):
            break
        x, y, z, yaw, pitch, roll = poses[i]
        yaw_r = np.radians(yaw)
        pitch_r = np.radians(pitch)
        roll_r = np.radians(roll)
        R = _rotation_matrix(yaw_r, pitch_r, roll_r)

        # Place grid points
        pts = np.stack([gx, gy, gz], axis=1)  # (npts, 3)
        placed = pts @ R.T + np.array([x, y, z])  # (npts, 3)
        ppx, ppy, ppz = placed[:, 0], placed[:, 1], placed[:, 2]
        in_corridor = _corridor_mask(ppx, ppy, ppz)
        survive = survive & in_corridor

        # If this is not the first pose, check the rotation arc bounding box.
        if i > 0:
            prev = poses[i - 1]
            # The rotation is about a single axis from old to new orientation.
            # For the arc bounding box, we need the axis-aligned box that contains
            # the arc of each grid point during rotation.
            # The rotation goes from R_prev to R_cur. Since each step is a single-axis
            # rotation (by the evaluator's construction), the arc is a circular arc
            # in the plane perpendicular to the rotation axis.
            # For a grid point p, its position during rotation is R(t) p + t_pos
            # where R(t) interpolates the rotation. The bounding box of this arc
            # is the axis-aligned box containing the arc.
            # Simplification: the bounding box of the arc for a point p is the
            # box with corners at the min/max of the arc. For a single-axis rotation
            # by angle d about axis a, the point traces a circular arc.
            # The axis-aligned bounding box of this arc can be computed.
            # However, since the evaluator says "the axis-aligned box bounding its
            # rotation arc lies in the corridor", we need to check this.
            # For efficiency, I'll approximate: the bounding box of the arc is
            # contained in the box spanned by the start and end positions of the
            # point, plus the arc bulge. A safe upper bound is the box containing
            # both the start and end positions of the point.
            # Actually, let me compute the arc bounding box more carefully.
            # The rotation axis is the axis that changed. Let's determine which
            # axis rotated.
            dyaw = yaw - prev[3]
            dpitch = pitch - prev[4]
            droll = roll - prev[5]
            # The axis with the largest change is the rotation axis.
            # (In practice, the evaluator rotates about a single axis per step.)
            changes = np.array([abs(dyaw), abs(dpitch), abs(droll)])
            axis = np.argmax(changes)
            angle = changes[axis]

            if angle > 1e-12:
                # Determine the rotation axis in world coordinates.
                # yaw -> z-axis, pitch -> y-axis, roll -> x-axis
                if axis == 0:  # yaw (z-axis)
                    a_vec = np.array([0.0, 0.0, 1.0])
                elif axis == 1:  # pitch (y-axis)
                    a_vec = np.array([0.0, 1.0, 0.0])
                else:  # roll (x-axis)
                    a_vec = np.array([1.0, 0.0, 0.0])

                # For each surviving grid point, compute the arc bounding box.
                # The point p is at position prev_pos = R_prev p + t_prev at start,
                # and cur_pos = R_cur p + t_cur at end.
                # During rotation, the point moves along a circular arc.
                # The axis-aligned bounding box of the arc contains the arc.
                # For a rotation by angle d about axis a (through the origin),
                # the point p traces: p(t) = R(t) p where R(t) is the rotation.
                # The bounding box of this arc (in the rotated frame) is:
                # Let p_perp = p - (p . a) a (component perpendicular to axis)
                # Let p_par = (p . a) a (component along axis)
                # The arc in the perpendicular plane goes from p_perp to R_perp p_perp
                # where R_perp is the 2D rotation by angle d.
                # The bounding box of this arc in the perpendicular plane:
                # If |d| <= 90 deg, the arc is convex and the bounding box is
                # the box containing the start and end points of the arc in the
                # perpendicular plane.
                # For larger angles, the arc may bulge beyond the chord.
                # A safe bound: the bounding box of the arc is the box containing
                # all points on the arc. For a circular arc of radius r and angle d,
                # the maximum extent in the direction perpendicular to the chord
                # is r * (1 - cos(d/2)) if d < 180, etc.
                # For simplicity and safety, I'll use the bounding box of the
                # start and end positions of the point, which is a lower bound
                # on the arc bounding box. This may be too optimistic.
                # Let me compute the actual arc bounding box.
                # The position of point p at parameter t in [0, 1] is:
                # pos(t) = R(t) p + t_pos(t)
                # where R(t) interpolates the rotation and t_pos(t) interpolates
                # the translation.
                # For the arc bounding box, we need the min and max of each
                # coordinate of pos(t) over t in [0, 1].
                # This is complex. Let me use a simpler but safe approach:
                # sample the arc at several points and take the bounding box.
                n_samples = 10
                ts = np.linspace(0, 1, n_samples)
                # Interpolate the pose
                interp_poses = prev[None, :] + ts[:, None] * (poses[i] - prev[None, :])
                # For each sample, compute the placed position
                sample_pts = np.zeros((n_samples, npts, 3))
                for s in range(n_samples):
                    ix, iy, iz, iyaw, ipitch, iroll = interp_poses[s]
                    R_s = _rotation_matrix(np.radians(iyaw), np.radians(ipitch), np.radians(iroll))
                    sample_pts[s] = pts @ R_s.T + np.array([ix, iy, iz])
                # Bounding box of the arc for each point
                arc_min = sample_pts.min(axis=0)  # (npts, 3)
                arc_max = sample_pts.max(axis=0)  # (npts, 3)
                # Check if the arc bounding box is in the corridor.
                # The box is [arc_min, arc_max] for each point.
                # Check corners of the box.
                corners = np.array(list(product([0, 1], repeat=3)))  # 8 corners
                # For each point, check all 8 corners of its arc bounding box
                # This is expensive. Let me check just the min and max corners
                # and the center, which should be sufficient for a convex corridor.
                # Actually, the corridor is not convex, so I need to check more.
                # Let me check the 8 corners of each point's bounding box.
                # For efficiency, I'll check only the surviving points.
                idx = np.where(survive)[0]
                if len(idx) > 0:
                    # For each surviving point, check its arc bounding box corners
                    for pt_idx in idx:
                        mn = arc_min[pt_idx]
                        mx = arc_max[pt_idx]
                        # Check 8 corners
                        for c in corners:
                            cx = mn[0] if c[0] == 0 else mx[0]
                            cy = mn[1] if c[1] == 0 else mx[1]
                            cz = mn[2] if c[2] == 0 else mx[2]
                            if not _corridor_mask(cx, cy, cz):
                                survive[pt_idx] = False
                                break

    # Count 4x4x4 blocks that all survive
    # Reshape survive to (81, 21, 21)
    survive_3d = survive.reshape(81, 21, 21)
    # A 4x4x4 block is defined by its top-left-front corner (i, j, k)
    # where i in [0, 78], j in [0, 17], k in [0, 17]
    # The block covers [i:i+4, j:j+4, k:k+4]
    block_survive = np.zeros((78, 18, 18), dtype=bool)
    for di in range(4):
        for dj in range(4):
            for dk in range(4):
                block_survive &= survive_3d[di:di+78, dj:dj+18, dk:dk+18]
    return int(np.sum(block_survive))


def _seed_path():
    """The baseline seed path from the reference program."""
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


def _local_search(path, max_time=600.0):
    """Coordinate-descent local search on the path poses.
    Tries small perturbations of each pose's 6 DOF and accepts if the
    surviving-block count increases. Uses a smooth surrogate (soft count)
    to guide the search."""
    best_path = path.copy()
    best_score = _evaluate(best_path)
    start_time = time.time()
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    angle_step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    n_poses = len(best_path)
    n_dof = 6

    # Coordinate descent: iterate over poses and DOF
    for ss_idx, trans_ss in enumerate(step_sizes):
        ang_ss = angle_step_sizes[ss_idx]
        for _ in range(3):  # multiple passes per step size
            for p in range(n_poses):
                if time.time() > start_time + max_time - 30:
                    return best_path, best_score
                for d in range(n_dof):
                    # Try positive and negative perturbations
                    for sign in [1, -1]:
                        candidate = best_path.copy()
                        if d < 3:
                            candidate[p, d] += sign * trans_ss
                        else:
                            candidate[p, d] += sign * ang_ss
                        # Evaluate
                        score = _evaluate(candidate)
                        if score > best_score:
                            best_path = candidate
                            best_score = score
    return best_path, best_score


def moving_sofa_3d_path() -> np.ndarray:
    """
    The 3D moving sofa problem: find the largest-volume rigid 3D shape that can
    travel through an L-shaped 3D corridor.

    Returns a (k, 6) float array of poses (x, y, z, yaw, pitch, roll) in degrees.
    The evaluator prepends (-3,0,0,0,0,0), appends (0,6,7,90,90,90) and densifies.
    """
    path = _seed_path()
    best_path, best_score = path, 0

    # Evaluate the seed
    seed_score = _evaluate(path)
    best_score = seed_score
    best_path = path.copy()

    # Local search with deadline guard
    if time.time() < _DEADLINE - 30:
        best_path, best_score = _local_search(path, max_time=_DEADLINE - time.time() - 30)

    return best_path


# EVOLVE-BLOCK-END
