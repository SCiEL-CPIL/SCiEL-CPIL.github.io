import numpy as np
from scipy.optimize import minimize
import time

N = 11
sqrt3 = np.sqrt(3)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    axial = [(0,2),(-1,1),(0,1),(1,1),(-1,0),(0,0),(1,0),(-1,-1),(0,-1),(1,-1),(0,-2)]
    bp = np.array([[sqrt3*(q+r/2), 1.5*r] for q,r in axial])
    bp -= bp.mean(axis=0)

    def hv(cx, cy, ad):
        a = np.radians(ad)
        c, s = np.cos(a), np.sin(a)
        bv = np.array([[1,0],[.5,sqrt3/2],[-.5,sqrt3/2],[-1,0],[-.5,-sqrt3/2],[.5,-sqrt3/2]])
        return bv @ np.array([[c,-s],[s,c]]) + np.array([cx, cy])

    def side_len(oc, oa, ip, ia):
        oar = np.radians(oa)
        ns = np.array([[np.cos(oar+k*np.pi/3), np.sin(oar+k*np.pi/3)] for k in range(6)])
        mx = 0.0
        for i in range(N):
            vs = hv(ip[i,0], ip[i,1], ia[i])
            for v in vs:
                d = (v - oc) @ ns.T
                mx = max(mx, float(d.max()))
        return mx / (sqrt3/2)

    def f1(p):
        return side_len(np.array([p[0],p[1]]), p[2], bp, [0.0]*N)

    bv = float('inf')
    bo = np.array([0.0, 0.0, 0.0])
    for a0 in np.linspace(0, 60, 7):
        r = minimize(f1, [0,0,a0], method='Nelder-Mead', options={'maxiter':5000,'xatol':1e-10,'fatol':1e-10})
        if r.fun < bv:
            bv = r.fun
            bo = r.x
        if time.time() > deadline - 60:
            break

    inner = [[bp[i,0], bp[i,1], 0.0] for i in range(N)]

    if time.time() > deadline - 40:
        return inner, (float(bo[0]), float(bo[1])), float(bv), float(bo[2])

    def f2(p):
        oc = np.array([p[0], p[1]])
        oa = p[2]
        ip = np.array([[p[3+i*3], p[4+i*3]] for i in range(N)])
        ia = [p[5+i*3] for i in range(N)]
        return side_len(oc, oa, ip, ia)

    x0 = [bo[0], bo[1], bo[2]]
    for i in range(N):
        x0 += [bp[i,0], bp[i,1], 0.0]

    r2 = minimize(f2, x0, method='Nelder-Mead', options={'maxiter':20000,'xatol':1e-9,'fatol':1e-9})
    if r2.fun < bv:
        bv = r2.fun
        inner = []
        for i in range(N):
            inner.append([r2.x[3+i*3], r2.x[4+i*3], r2.x[5+i*3]])
        bo = r2.x[:3]

    return inner, (float(bo[0]), float(bo[1])), float(bv), float(bo[2])
