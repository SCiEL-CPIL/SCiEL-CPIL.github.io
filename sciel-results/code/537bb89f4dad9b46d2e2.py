import numpy as np
import time

N = 11

def min_enclosing_hexagon_side(points, center, angle_deg):
    angle = np.radians(angle_deg)
    normals = np.array([[np.cos(angle + np.pi/6 + k*np.pi/3), np.sin(angle + np.pi/6 + k*np.pi/3)] for k in range(6)])
    max_proj = 0
    for p in points:
        rel = p - center
        projs = np.abs(rel @ normals.T)
        max_proj = max(max_proj, np.max(projs))
    return 2 * max_proj / np.sqrt(3)

def get_all_vertices(inner):
    pts = []
    for cx, cy, ang in inner:
        a0 = np.radians(ang)
        for k in range(6):
            a = a0 + k*np.pi/3
            pts.append((cx + np.cos(a), cy + np.sin(a)))
    return np.array(pts)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    def init_config():
        cfg = [[0.0, 0.0, 0.0]]
        for k in range(6):
            a = k*np.pi/3
            cfg.append([2*np.cos(a), 2*np.sin(a), 0.0])
        for k in range(4):
            a = k*np.pi/2 + np.pi/4
            cfg.append([4*np.cos(a), 4*np.sin(a), 0.0])
        return cfg
    
    best = init_config()
    pts = get_all_vertices(best)
    cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])
    
    best_side = float('inf')
    best_angle = 0
    for ang in range(0, 360, 30):
        s = min_enclosing_hexagon_side(pts, (cx, cy), ang)
        if s < best_side:
            best_side = s
            best_angle = ang
    
    np.random.seed(42)
    np_ = 20
    npars = 36
    parts = np.zeros((np_, npars))
    vels = np.zeros((np_, npars))
    
    for i in range(np_):
        cfg = init_config()
        for k in range(N):
            parts[i,2*k] = cfg[k][0] + np.random.randn()*0.3
            parts[i,2*k+1] = cfg[k][1] + np.random.randn()*0.3
            parts[i,2*N+k] = np.random.uniform(0,60)
        parts[i,3*N] = cx + np.random.randn()*0.3
        parts[i,3*N+1] = cy + np.random.randn()*0.3
        parts[i,3*N+2] = np.random.uniform(0,60)
    
    w,c1,c2 = 0.8,1.5,1.5
    pb = parts.copy()
    pb_fit = np.full(np_, float('inf'))
    gb = parts[0].copy()
    gb_fit = float('inf')
    
    def eval_p(p):
        inner = [[p[2*k],p[2*k+1],p[2*N+k]] for k in range(N)]
        pen = 0
        for i in range(N):
            for j in range(i+1,N):
                d = np.hypot(inner[i][0]-inner[j][0], inner[i][1]-inner[j][1])
                if d < 1.85:
                    pen += (1.85-d)*50
        if pen > 5: return float('inf')
        c = (p[3*N],p[3*N+1])
        a = p[3*N+2]
        pts = get_all_vertices(inner)
        return min_enclosing_hexagon_side(pts,c,a) + pen
    
    for i in range(np_):
        f = eval_p(parts[i])
        if f < pb_fit[i]:
            pb_fit[i] = f
            pb[i] = parts[i].copy()
        if f < gb_fit:
            gb_fit = f
            gb = parts[i].copy()
    
    if gb_fit < best_side:
        best_side = gb_fit
        best = [[gb[2*k],gb[2*k+1],gb[2*N+k]] for k in range(N)]
        cx,cy = gb[3*N],gb[3*N+1]
        best_angle = gb[3*N+2]
    
    for it in range(400):
        if time.time() > deadline: break
        for i in range(np_):
            r1 = np.random.rand(npars)
            r2 = np.random.rand(npars)
            vels[i] = w*vels[i] + c1*r1*(pb[i]-parts[i]) + c2*r2*(gb-parts[i])
            parts[i] += vels[i]
            f = eval_p(parts[i])
            if f < pb_fit[i]:
                pb_fit[i] = f
                pb[i] = parts[i].copy()
            if f < gb_fit:
                gb_fit = f
                gb = parts[i].copy()
        if gb_fit < best_side:
            best_side = gb_fit
            best = [[gb[2*k],gb[2*k+1],gb[2*N+k]] for k in range(N)]
            cx,cy = gb[3*N],gb[3*N+1]
            best_angle = gb[3*N+2]
    
    return [[best[k][0],best[k][1],best[k][2]%360] for k in range(N)], (cx,cy), best_side, best_angle%360
