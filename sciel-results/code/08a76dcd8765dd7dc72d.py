import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def score(points):
        n = len(points)
        d = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if d2 > d:
                    d = d2
        if d <= 0:
            return -1.0
        diam = np.sqrt(d)
        scaled = [ [p[0]/diam, p[1]/diam] for p in points ]
        log_prod = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = scaled[i][0] - scaled[j][0]
                dy = scaled[i][1] - scaled[j][1]
                dd = dx*dx + dy*dy
                if dd > 0:
                    log_prod += 0.5 * np.log(dd)
        return np.exp(log_prod) / (n ** n)

    def reg_octagon():
        pts = []
        for k in range(8):
            ang = 2 * np.pi * k / 8
            pts.append([np.cos(ang), np.sin(ang)])
        return pts

    best = reg_octagon()
    best_score = score(best)

    # Try dual square configuration
    # 4 points at (±a, 0), 4 points at (0, ±a) - this is just a regular octagon rotated
    # Better: two squares at different radii
    # Let's try: 4 at radius R, 4 at radius r, rotated by 45 degrees
    for r_inner in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]:
        pts = []
        # Outer square
        for k in range(4):
            ang = np.pi/4 + 2 * np.pi * k / 4
            pts.append([np.cos(ang), np.sin(ang)])
        # Inner square
        for k in range(4):
            ang = 2 * np.pi * k / 4
            pts.append([r_inner * np.cos(ang), r_inner * np.sin(ang)])
        s = score(pts)
        if s > best_score:
            best_score = s
            best = pts

    # Try rectangular configurations
    for rx in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5]:
        for ry in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5]:
            pts = []
            for sx in [-1, 1]:
                for sy in [-1, 1]:
                    pts.append([sx*rx, sy*ry])
            s = score(pts)
            if s > best_score:
                best_score = s
                best = pts

    # Gradient-based optimization on 16 coordinates with diameter scaling
    def _score_neg(pts_list):
        return -score(pts_list)

    try:
        from scipy.optimize import minimize
        for restart in range(50):
            if time.time() > deadline:
                break
            # Random perturbation of best or random start
            if restart < 20:
                base = np.array(best, dtype=float)
            else:
                base = np.random.randn(8, 2) * 0.5
            noise_scale = 0.1 if restart < 20 else 0.5
            x0 = base + np.random.randn(16) * noise_scale
            x0 = x0.reshape(8, 2)

            def objective(x):
                pts = x.reshape(8, 2).tolist()
                return _score_neg(pts)

            res = minimize(objective, x0.reshape(-1), method='L-BFGS-B',
                          options={'maxiter': 500, 'gtol': 1e-8})
            if res.success and res.fun < -best_score + 1e-12:
                cand = res.x.reshape(8, 2).tolist()
                cs = score(cand)
                if cs > best_score:
                    best_score = cs
                    best = cand
    except Exception:
        pass

    # Simple random search with local improvement
    for it in range(200):
        if time.time() > deadline:
            break
        base = np.array(best, dtype=float)
        x = base + np.random.randn(8, 2) * 0.05
        pts = x.tolist()
        s = score(pts)
        if s > best_score:
            best_score = s
            best = pts

    # Verify final result
    final_score = score(best)
    if final_score < best_score - 1e-9:
        # Fallback to regular octagon
        best = reg_octagon()

    # Ensure distinct points and valid coordinates
    result = []
    for p in best:
        px = float(np.clip(p[0], -999999, 999999))
        py = float(np.clip(p[1], -999999, 999999))
        result.append([px, py])

    # Check distinctness
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(result[i][0] - result[j][0]) < 1e-10 and abs(result[i][1] - result[j][1]) < 1e-10:
                result[j][0] += 1e-8

    return result
