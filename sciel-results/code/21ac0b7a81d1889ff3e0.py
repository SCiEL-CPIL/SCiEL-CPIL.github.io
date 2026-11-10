import numpy as np
import time

_START_TIME = None

def _R_of(yaw, pitch, roll):
    yaw = np.radians(yaw); pitch = np.radians(pitch); roll = np.radians(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    Ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
    Rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
    return Rx @ Ry @ Rz

def _in_corridor(x, y, z):
    r1 = (x <= 0) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1)
    r2 = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 5) & (z >= 0) & (z <= 1)
    r3 = (x >= 0) & (x <= 1) & (y >= 5) & (y <= 6) & (z >= 0)
    return r1 | r2 | r3

def _densify(ctrl):
    # ctrl: (K,6)
    pts = [ctrl[0]]
    for i in range(1, len(ctrl)):
        a, b = ctrl[i-1], ctrl[i]
        dist = np.linalg.norm(b[:3] - a[:3])
        dmax = max(abs(b[3]-a[3]), abs(b[4]-a[4]), abs(b[5]-a[5]))
        n = max(int(np.ceil(dist/0.05)), int(np.ceil(dmax/2.0)), 1)
        for t in np.linspace(0.0, 1.0, n+1)[1:]:
            pts.append(a + t*(b-a))
    return np.array(pts)

def _block_count(dense, P, N):
    # P: (N,3) local points; N points
    # returns (survive_bool (N,), block_count int)
    K = len(dense)
    R = np.array([_R_of(dense[i,3], dense[i,4], dense[i,5]) for i in range(K)])
    T = dense[:, :3]
    # survival
    # pose 0
    w0 = (R[0] @ P.T).T + T[0]
    surv = _in_corridor(w0[:,0], w0[:,1], w0[:,2])
    for i in range(1, K):
        if not surv.any():
            break
        R0, R1 = R[i-1], R[i]
        t0, t1 = T[i-1], T[i]
        o = (R0 @ P.T).T  # (N,3) offset
        # translation sub-moves
        px = t0.copy(); px[:,0] = t1[0]
        pos = o + px
        surv &= _in_corridor(pos[:,0], pos[:,1], pos[:,2])
        py = t0.copy(); py[:,0] = t1[0]; py[:,1] = t1[1]
        pos = o + py
        surv &= _in_corridor(pos[:,0], pos[:,1], pos[:,2])
        pz = t1.copy()
        pos = o + pz
        surv &= _in_corridor(pos[:,0], pos[:,1], pos[:,2])
        # rotation about single axis
        # determine axis: which angle changed
        dy, dp, dr = dense[i,3]-dense[i-1,3], dense[i,4]-dense[i-1,4], dense[i,5]-dense[i-1,5]
        # arc sampling
        v0 = o  # = R0@P
        # sample rotation from R0 to R1: interpolate angles
        ang0 = np.array([dense[i-1,3], dense[i-1,4], dense[i-1,5]])
        ang1 = np.array([dense[i,3], dense[i,4], dense[i,5]])
        S = 12
        xs = np.empty((S, N)); ys = np.empty((S, N)); zs = np.empty((S, N))
        for s in range(S):
            a = ang0 + (s/(S-1))*(ang1-ang0)
            Rs = _R_of(a[0], a[1], a[2])
            w = (Rs @ P.T).T + t1
            xs[s] = w[:,0]; ys[s] = w[:,1]; zs[s] = w[:,2]
        xmin = xs.min(0); xmax = xs.max(0)
        ymin = ys.min(0); ymax = ys.max(0)
        zmin = zs.min(0); zmax = zs.max(0)
        # check 8 corners in corridor
        ok = True
        for cx in (xmin, xmax):
            for cy in (ymin, ymax):
                for cz in (zmin, zmax):
                    ok &= _in_corridor(cx, cy, cz)
        surv &= ok
    # reshape survival to grid
    Sg = surv.reshape(81, 21, 21)
    bx = Sg & np.roll(Sg,1,0) & np.roll(Sg,2,0) & np.roll(Sg,3,0)
    by = bx & np.roll(bx,1,1) & np.roll(bx,2,1) & np.roll(bx,3,1)
    bk = by & np.roll(by,1,2) & np.roll(by,2,2) & np.roll(by,3,2)
    cnt = int(bk[:78, :18, :18].sum())
    return surv, cnt
