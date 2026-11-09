import math
import time

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Uses Nelder-Mead on g(x)=f(x)^2 for global minimization,
    then secant/Newton polish for high precision.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    tol_final = 1e-8

    x0 = float(x0)
    f0 = f(x0)
    if math.isfinite(f0) and abs(f0) <= tol_final:
        return x0

    # Handle non-finite f0
    if not math.isfinite(f0):
        base_step = 1e-4 * max(1.0, abs(x0))
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64]:
            if time.time() > deadline:
                break
            x_try = x0 + s * base_step
            f_try = f(x_try)
            if math.isfinite(f_try) and abs(f_try) <= tol_final:
                return x_try
            if math.isfinite(f_try):
                x0, f0 = x_try, f_try
                break
        if not math.isfinite(f0):
            return x0

    # Nelder-Mead simplex on g(x) = f(x)^2
    # 1D NM: two points, reflect/contract/expand
    h0 = 1e-4 * max(1.0, abs(x0))
    # Initialize simplex around x0
    simplex = [(x0, f0)]
    # Evaluate a second point
    x1 = x0 + h0
    f1 = f(x1)
    if not math.isfinite(f1):
        # Try other directions
        for s in [-1, 2, -2, 4, -4, 8, -8, 16, -16]:
            if time.time() > deadline:
                break
            x_try = x0 + s * h0
            f_try = f(x_try)
            if math.isfinite(f_try):
                x1, f1 = x_try, f_try
                break
        if not math.isfinite(f1):
            return x0

    simplex.append((x1, f1))

    best_x = simplex[0][0]
    best_f = abs(simplex[0][1])
    for pt in simplex:
        if abs(pt[1]) < best_f:
            best_x = pt[0]
            best_f = abs(pt[1])

    # Nelder-Mead iterations
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    for iteration in range(80):
        if time.time() > deadline - 2.0:
            break

        # Sort simplex: worst point last
        simplex.sort(key=lambda p: abs(p[1]))
        x_best, f_best = simplex[0]
        x_worst, f_worst = simplex[-1]

        if abs(f_best) <= tol_final:
            return x_best

        # Centroid of all but worst
        # In 1D, centroid is just the best point
        x_centroid = x_best

        # Reflect worst point through centroid
        x_ref = x_centroid + alpha * (x_centroid - x_worst)
        f_ref = f(x_ref)
        if not math.isfinite(f_ref):
            # Shrink
            x_shrink = x_centroid + sigma * (x_worst - x_centroid)
            f_shrink = f(x_shrink)
            if math.isfinite(f_shrink):
                simplex[-1] = (x_shrink, f_shrink)
            else:
                break
            continue

        if abs(f_ref) <= tol_final:
            return x_ref

        if abs(f_ref) < abs(f_best):
            # Expand
            x_exp = x_centroid + gamma * (x_ref - x_centroid)
            f_exp = f(x_exp)
            if math.isfinite(f_exp) and abs(f_exp) < abs(f_ref):
                simplex[-1] = (x_exp, f_exp)
            else:
                simplex[-1] = (x_ref, f_ref)
        elif abs(f_ref) < abs(f_worst):
            # Accept reflection
            simplex[-1] = (x_ref, f_ref)
        else:
            # Contract
            if abs(f_ref) < abs(f_worst):
                # Outside contraction
                x_con = x_centroid + rho * (x_ref - x_centroid)
            else:
                # Inside contraction
                x_con = x_centroid - rho * (x_worst - x_centroid)
            f_con = f(x_con)
            if math.isfinite(f_con) and abs(f_con) < abs(f_worst):
                simplex[-1] = (x_con, f_con)
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    x_sh = x_centroid + sigma * (simplex[i][0] - x_centroid)
                    f_sh = f(x_sh)
                    if math.isfinite(f_sh):
                        simplex[i] = (x_sh, f_sh)
                    else:
                        simplex[i] = (x_centroid, f_centroid if 'f_centroid' in dir() else f(x_centroid))
                break

    # Update best from simplex
    for pt in simplex:
        if math.isfinite(pt[1]) and abs(pt[1]) < best_f:
            best_x = pt[0]
            best_f = abs(pt[1])

    if abs(best_f) <= tol_final:
        return best_x

    # Secant polish from best point
    # Find a second point for secant
    if len(simplex) >= 2:
        # Use the two best points
        simplex.sort(key=lambda p: abs(p[1]))
        xa, fa = simplex[0]
        xb, fb = simplex[1]
    else:
        xa = best_x
        fa = f(xa) if not math.isfinite(fa) else fa
        xb = xa + 1e-6 * max(1.0, abs(xa))
        fb = f(xb)
        if not math.isfinite(fb):
            xb = xa - 1e-6 * max(1.0, abs(xa))
            fb = f(xb)
            if not math.isfinite(fb):
                return best_x

    # Secant iterations for polish
    for _ in range(50):
        if time.time() > deadline - 0.5:
            break
        denom = fb - fa
        if abs(denom) < 1e-15:
            break
        xn = xb - fb * (xb - xa) / denom
        if not math.isfinite(xn):
            break
        fn = f(xn)
        if not math.isfinite(fn):
            break
        if abs(fn) < best_f:
            best_x = xn
            best_f = abs(fn)
        if abs(fn) <= tol_final:
            return xn
        if abs(xn - xb) < 1e-14:
            break
        xa, fa = xb, fb
        xb, fb = xn, fn

    if abs(fb) <= tol_final:
        return xb
    if abs(fa) <= tol_final:
        return xa

    return best_x
