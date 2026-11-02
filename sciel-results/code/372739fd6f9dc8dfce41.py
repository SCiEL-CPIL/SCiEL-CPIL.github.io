import numpy as np
import time
from scipy.optimize import minimize

N = 11

def _hex_verts(s, angle_deg, center):
    a = np.radians(angle_deg)
    ba = np.array([0, 60, 120, 180, 240, 300], dtype=float) * np.pi / 180
    angs = ba + a
    return np.column_stack([s * np.cos(angs), s * np.sin(angs)]) + np.array(center)

def _sd(px, py, cx, cy, s, ad):
    a = np.radians(ad)
    px -= cx; py -= cy
    ir = s * np.sqrt(3) / 2.0
    md = 1e18
    for k in range(6):
        th = a + np.radians(30 + k * 60)
        d = ir - (np.cos(th) * px + np.sin(th) * py)
        if d < md: md = d
    return md

def _gap(v1, v2):
    mp = 0.0
    for pa, pb in [(v1, v2), (v2, v1)]:
        for k in range(6):
            e = pa[(k+1)%6] - pa[k]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-12: continue
            n /= nn
            p1 = pa @ n; p2 = pb @ n
            g = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
            if g < 0 and -g > mp: mp = -g
    return mp

def _f(p):
    ip = p[:3*N]; ox,oy,os,oa = p[3*N],p[3*N+1],p[3*N+2],p[3*N+3]
    if os < 1: return 1e10
    vs = [_hex_verts(1.0, ip[3*k+2], (ip[3*k], ip[3*k+1])) for k in range(N)]
    cp = sum(max(0, -_sd(v[0],v[1],ox,oy,os,oa))**2 for v in [vv for vs_ in vs for vv in vs_])
    op = sum(_gap(vs[i],vs[j])**2 for i in range(N) for j in range(i+1,N))
    return os + 200*cp + 200*op

def construct():
    t0 = time.time(); tl = 135.0
    d = np.sqrt(3)
    pos = [(0.0,0.0,0.0)]
    for k in range(6):
        a = np.radians(k*60); pos.append((d*np.cos(a), d*np.sin(a), 0.0))
    for k in range(4):
        a = np.radians(k*60); pos.append((2*d*np.cos(a), 2*d*np.sin(a), 0.0))
    bx, bf = None, float('inf')
    np.random.seed(42)
    for s in range(12):
        if time.time()-t0 > tl-5: break
        if s==0:
            x0 = np.array([c for p in pos for c in [p[0],p[1],p[2]]] + [0,0,5.0,0.0])
        else:
            x0 = np.array([p[0]+np.random.randn()*.2 for p in pos]+[p[1]+np.random.randn()*.2 for p in pos]+[p[2]+np.random.randn()*.05 for p in pos]+[np.random.randn()*.1,np.random.randn()*.1,5+np.random.randn()*.3,np.random.randn()*.05])
        try:
            r = minimize(_f, x0, method='L-BFGS-B', options={'maxiter':400,'ftol':1e-10,'gtol':1e-8,'maxfun':10000})
            if r.fun < bf: bf, bx = r.fun, r.x
        except: pass
        if time.time()-t0 > tl-10: break
    if bx is None: bx = x0
    inner = [[bx[3*k], bx[3*k+1], np.degrees(bx[3*k+2])] for k in range(N)]
    return inner, (bx[3*N], bx[3*N+1]), max(bx[3*N+2], 3.0), np.degrees(bx[3*N+3])
