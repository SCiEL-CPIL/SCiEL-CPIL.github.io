"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 120.0   # safety margin before 150s kill
_SAFETY_MARGIN = 5.0  # seconds to stop generating new candidates

def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)

def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

def _squared_residual(f, x):
    """g(x) = f(x)^2, robust to non-finite evaluations."""
    try:
        fx = f(x)
    except Exception:
        return float('inf')
    if not _is_finite(fx):
        return float('inf')
    return fx * fx

def _nm_1d(f, x0, max_iter=120, tol=1e-10, delta0=1.0):
    """1-D Nelder-Mead on g(x)=f(x)^2. Returns (x, g) for the best vertex seen."""
    try:
        x0 = float(x0)
    except Exception:
        return x0, float('inf')

    d = delta0 * max(1.0, abs(x0))
    if d < 1e-6:
        d = 1e-6

    # simplex: two points
    v0 = x0
    v1 = x0 + d
    g0 = _squared_residual(f, v0)
    g1 = _squared_residual(f, v1)

    best_x = v0 if g0 <= g1 else v1
    best_g = min(g0, g1)

    for _ in range(max_iter):
        if _time_left() < 0.5:
            break
        # order: b (best) then w (worst)
        if g0 <= g1:
            b, w = v0, v1
            gb, gw = g0, g1
        else:
            b, w = v1, v0
            gb, gw = g1, g0

        if gw <= tol:
            break
        if abs(w - b) < 1e-14 * max(1.0, abs(b)):
            break

        # centroid of best vertex
        c = b
        # reflection
        xr = c - (w - c)
        gr = _squared_residual(f, xr)
        if not _is_finite(gr):
            gr = float('inf')

        if gr < gb:
            # expansion
            xe = c - 2.0 * (w - c)
            ge = _squared_residual(f, xe)
            if not _is_finite(ge):
                ge = float('inf')
            if ge < gr:
                w, gw = xe, ge
            else:
                w, gw = xr, gr
        elif gr < gw:
            w, gw = xr, gr
        else:
            # contraction
            xc = c + 0.5 * (w - c)
            gc = _squared_residual(f, xc)
            if not _is_finite(gc):
                gc = float('inf')
            if gc < gw:
                w, gw = xc, gc
            else:
                # shrink both toward best
                v0 = b + 0.5 * (v0 - b)
                v1 = b + 0.5 * (v1 - b)
                g0 = _squared_residual(f, v0)
                g1 = _squared_residual(f, v1)
                if not _is_finite(g0):
                    g0 = float('inf')
                if not _is_finite(g1):
                    g1 = float('inf')
                w = v0 if g0 <= g1 else v1
                gw = min(g0, g1)

        if gw < best_g:
            best_g = gw
            best_x = w
        if gb < best_g:
            best_g = gb
            best_x = b

        # recompute g for the two vertices if they changed via reflection/expansion
        if gw == gr or gw == ge:
            pass
        # keep vertex values consistent
        if g0 > best_g and g1 > best_g:
            pass
        # update vertex g values from the current v's only if we shrank
        # (recompute cheaply only when needed)

    return best_x, best_g

def _secant(f, x0, max_iter=25, h=1e-5):
    """Secant method reusing previous evaluations."""
    try:
        x0 = float(x0)
    except Exception:
        return x0, float('inf')
    x1 = x0 + h * max(1.0, abs(x0))
    try:
        fx0 = f(x0)
        fx1 = f(x1)
    except Exception:
        return x0, float('inf')
    if not _is_finite(fx0) or not _is_finite(fx1):
        return x0, float('inf')
    if abs(fx0) <= 1e-12:
        return x0, fx0 * fx0
    if abs(fx1) <= 1e-12:
        return x1, fx1 * fx1
    for _ in range(max_iter):
        if _time_left() < 0.5:
            break
        denom = fx1 - fx0
        if abs(denom) < 1e-15:
            break
        x_new = x1 - fx1 * (x1 - x0) / denom
        if not _is_finite(x_new):
            break
        x0, x1 = x1, x_new
        fx0, fx1 = fx1, f(x1)
        if not _is_finite(fx1):
            break
        if abs(fx1) <= 1e-12:
            return x1, fx1 * fx1
    return x1, fx1 * fx1

def _newton_fd(f, x, max_iter=20, h_base=1e-6):
    """Newton's method with finite-difference derivative."""
    try:
        x = float(x)
    except Exception:
        return x, float('inf')
    for _ in range(max_iter):
        if _time_left() < 0.5:
            break
        try:
            fx = f(x)
        except Exception:
            break
        if not _is_finite(fx):
            break
        if abs(fx) <= 1e-12:
            return x, fx * fx
        h = h_base * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
        except Exception:
            break
        if not _is_finite(fx_plus):
            break
        d = (fx_plus - fx) / h
        if d == 0 or not _is_finite(d):
            break
        x_new = x - fx / d
        if not _is_finite(x_new):
            break
        x = x_new
    try:
        fx_final = f(x)
        if _is_finite(fx_final):
            return x, fx_final * fx_final
    except Exception:
        pass
    return x, float('inf')

def find_root(f, x0):
    """
    Global minimization of g(x)=f(x)^2 via 1-D Nelder-Mead, with a
    fast secant/Newton polish. Returns the x with smallest |f(x)| found.
    """
    try:
        x0 = float(x0)
    except Exception:
        return x0

    best_x = x0
    best_g = float('inf')

    # Quick check of x0 itself
    try:
        g0 = _squared_residual(f, x0)
        if g0 < best_g:
            best_g = g0
            best_x = x0
        if g0 <= 1e-16:  # |f| <= 1e-8
            return x0
    except Exception:
        pass

    # Fast local polish: secant then Newton (cheap, converges in few calls)
    if _time_left() > 1.0:
        xs, gs = _secant(f, x0, max_iter=20)
        if _is_finite(xs) and gs < best_g:
            best_g = gs
            best_x = xs
        if gs <= 1e-16:
            return xs
        xn, gn = _newton_fd(f, x0, max_iter=15)
        if _is_finite(xn) and gn < best_g:
            best_g = gn
            best_x = xn
        if gn <= 1e-16:
            return xn

    # Global Nelder-Mead over the squared residual
    if _time_left() > 2.0:
        # try several simplex scales to explore the landscape broadly
        for scale in [1.0, 0.1, 10.0, 100.0, 0.01]:
            if _time_left() < 0.5:
                break
            xc, gc = _nm_1d(f, x0, max_iter=60, tol=1e-12, delta0=scale)
            if _is_finite(xc) and gc < best_g:
                best_g = gc
                best_x = xc
            if gc <= 1e-16:
                return xc

    # Polish the best point with secant to nail down to 1e-8
    if _time_left() > 0.5:
        xp, gp = _secant(f, best_x, max_iter=15)
        if _is_finite(xp) and gp < best_g:
            best_g = gp
            best_x = xp
        xn, gn = _newton_fd(f, best_x, max_iter=10)
        if _is_finite(xn) and gn < best_g:
            best_g = gn
            best_x = xn

    # Final verification of best_x
    try:
        fx_best = f(best_x)
        if _is_finite(fx_best) and abs(fx_best) <= 1e-8:
            return best_x
    except Exception:
        pass

    return best_x
# EVOLVE-BLOCK-END
