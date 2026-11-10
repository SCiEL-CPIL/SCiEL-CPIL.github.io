import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 620
    
    def hverts(cx, cy, s, ad):
        a = np.radians(ad)
        ang = a + np.radians(60*np.arange(6))
        return np.column_stack([cx+s*np.cos(ang), cy+s*np.sin(ang)])
    
    def sep(v1, v2, tol=1e-6):
        for vi in (v1, v2):
            for i in range(6):
                e = vi[(i+1)%6] - vi[i]
                n = np.array([-e[1], e[0]])
                nn = np.linalg.norm(n)
                if nn < 1e-15: continue
                n /= nn
                p1 = v1 @ n; p2 = v2 @ n
                if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
                    return True
        return False
    
    def pin(p, c, s, ad):
        a = np.radians(ad)
        dx, dy = p[0]-c[0], p[1]-c[1]
        lx = dx*np.cos(-a) - dy*np.sin(-a)
        ly = dx*np.sin(-a) + dy*np.cos(-a)
        v = hverts(0, 0, s, 0)
        for i in range(6):
            e = v[(i+1)%6] - v[i]
            tp = np.array([lx,ly]) - v[i]
            if e[0]*tp[1] - e[1]*tp[0] < -1e-6:
                return False
        return True
    
    # Good initial guess: 7-cluster + 4 satellites
    r3 = np.sqrt(3)
    C = np.array([[0,0],[1.5,r3/2],[0,r3],[-1.5,r3/2],[-1.5,-r3/2],[0,-r3],[1.5,-r3/2],
                  [3,0],[0,2*r3],[-3,0],[0,-2*r3]])
    A = np.zeros(11)
    
    def obj(x):
        if time.time()-t0 > deadline: return 1e10
        C = x[:22].reshape(11,2); A = x[22:33]; s = x[33]
        if s < np.sqrt(11) or not np.isfinite(s): return 1e10
        V = [hverts(C[i,0],C[i,1],1.0,A[i]) for i in range(11)]
        for i in range(11):
            for j in range(i+1,11):
                if not sep(V[i],V[j]): return 1e10+0.1
            for v in V[i]:
                if not pin(v,[0,0],s,0): return 1e10+0.2
        return s
    
    x0 = np.concatenate([C.ravel(), A, [5.0]])
    best, bv = x0.copy(), 5.0
    
    for _ in range(3):
        if time.time()-t0 > deadline-20: break
        x0t = x0 + np.random.randn(34)*0.3
        x0t[33] = max(x0t[33], np.sqrt(11)+0.1)
        try:
            r = minimize(obj, x0t, method='Nelder-Mead', options={'maxiter':3000,'xatol':1e-3,'fatol':1e-3})
            if r.fun < bv and r.fun < 1e9:
                bv, best = r.fun, r.x
        except: pass
    
    C = best[:22].reshape(11,2); A = best[22:33]; s = best[33]
    return np.column_stack([C,A]), np.array([0.0,0.0,0.0]), float(s)
