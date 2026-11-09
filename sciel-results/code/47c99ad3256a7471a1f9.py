import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 40.0

    x = float(x0)
    fx = f(x)
    best_x = x
    best_fx = fx
    if abs(fx) <= 1e-8:
        return x

    # Adaptive initial exploration to find a sign change or good bracket
    h = max(1.0, abs(x)) * 1e-3
    x1 = x + h
    fx1 = f(x1)
    if abs(fx1) <= 1e-8:
        return x1
    if abs(fx1) < abs(best_fx):
        best_x = x1
        best_fx = fx1

    x2 = x - h
    fx2 = f(x2)
    if abs(fx2) <= 1e-8:
        return x2
    if abs(fx2) < abs(best_fx):
        best_x = x2
        best_fx = fx2

    # If no sign change found, try larger steps
    if fx * fx1 > 0 and fx * fx2 > 0 and fx1 * fx2 > 0:
        # All same sign - try expanding
        for _ in range(10):
            if time.time() > deadline:
                break
            h *= 10.0
            if h > 1e12:
                break
            x3 = x + h
            fx3 = f(x3)
            if abs(fx3) <= 1e-8:
                return x3
            if abs(fx3) < abs(best_fx):
                best_x = x3
                best_fx = fx3
            if fx * fx3 < 0:
                x2 = x3
                fx2 = fx3
                break
            x4 = x - h
            fx4 = f(x4)
            if abs(fx4) <= 1e-8:
                return x4
            if abs(fx4) < abs(best_fx):
                best_x = x4
                best_fx = fx4
            if fx * fx4 < 0:
                x1 = x4
                fx1 = fx4
                break

    # Choose two points for secant: prefer sign change, else closest to zero
    if fx * fx1 < 0:
        xa, fa = x, fx
        xb, fb = x1, fx1
    elif fx * fx2 < 0:
        xa, fa = x, fx
        xb, fb = x2, fx2
    elif fx1 * fx2 < 0:
        xa, fa = x1, fx1
        xb, fb = x2, fx2
    else:
        points = [(x, fx), (x1, fx1), (x2, fx2)]
        points.sort(key=lambda p: abs(p[1]))
        xa, fa = points[0]
        xb, fb = points[1]

    # Secant method: 1 function call per iteration after initial
    sx0, sf0 = xa, fa
    sx1, sf1 = xb, fb

    for i in range(60):
        if time.time() > deadline:
            break

        denom = sf1 - sf0
        if abs(denom) < 1e-300:
            break

        x_new = sx1 - sf1 * (sx1 - sx0) / denom

        if not math.isfinite(x_new):
            break

        f_new = f(x_new)

        if not math.isfinite(f_new):
            break

        if abs(f_new) < abs(best_fx):
            best_x = x_new
            best_fx = f_new

        if abs(f_new) <= 1e-8:
            return x_new

        # Update secant points
        sx0, sf0 = sx1, sf1
        sx1, sf1 = x_new, f_new

    # Polish phase: use secant with tighter tolerance
    if abs(best_fx) <= 1e-4:
        px = best_x
        pf = best_fx
        # Use a small perturbation for derivative estimate
        ph = 1e-7 * max(1.0, abs(px))
        for j in range(30):
            if time.time() > deadline:
                break
            px2 = px + ph
            pf2 = f(px2)
            if abs(pf2) <= 1e-8:
                return px2
            denom = pf2 - pf
            if abs(denom) < 1e-300:
                break
            pnew = px - pf * ph / denom
            if not math.isfinite(pnew):
                break
            pfnew = f(pnew)
            if not math.isfinite(pfnew):
                break
            if abs(pfnew) < abs(pf):
                px, pf = pnew, pfnew
                ph = 1e-7 * max(1.0, abs(px))
            else:
                ph *= 0.1
                if ph < 1e-16:
                    break
            if abs(pf) <= 1e-8:
                return px

    return best_x
