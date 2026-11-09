# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_N = 8
_N2 = _N * _N
_I = np.arange(_N2) // _N
_J = np.arange(_N2) % _N
_INV_N = 1.0 / _N

def _union_1d(intervals):
    """intervals: (M,2) array of [lo,hi]. Return total length of union."""
    if intervals.shape[0] == 0:
        return 0.0
    order = np.argsort(intervals[:, 0])
    lo = intervals[order, 0]
    hi = intervals[order, 1]
    total = 0.0
    cur_lo = lo[0]
    cur_hi = hi[0]
    for k in range(1, len(lo)):
        if lo[k] <= cur_hi:
            if hi[k] > cur_hi:
                cur_hi = hi[k]
        else:
            total += cur_hi - cur_lo
            cur_lo = lo[k]
            cur_hi = hi[k]
    total += cur_hi - cur_lo
    return total

def _exact_volume(params, n_quad=40):
    """Exact union volume via Gauss-Legendre quadrature in c in [0,1]."""
    x = params[0:_N2]
    y = params[_N2:2 * _N2]
    t, w = np.polynomial.legendre.leggauss(n_quad)
    c = 0.5 * (t + 1.0)
    vol = 0.0
    for ck, wk in zip(c, w):
        xoff = ck * _I * _INV_N
        yoff = ck * _J * _INV_N
        xl = x + xoff
        xr = xl + _INV_N
        yl = y + yoff
        yr = yl + _INV_N
        lx = _union_1d(np.stack([xl, xr], axis=1))
        ly = _union_1d(np.stack([yl, yr], axis=1))
        vol += wk * 0.5 * lx * ly
    return vol

def _vol_x_only(x, y, n_quad=40):
    """Volume treating x as variable, y fixed."""
    params = np.concatenate([x, y])
    return _exact_volume(params, n_quad)

def _vol_y_only(x, y, n_quad=40):
    """Volume treating y as variable, x fixed."""
    params = np.concatenate([x, y])
    return _exact_volume(params, n_quad)

def _obj_1d(x, n_quad=80):
    """
    Objective for the 1D problem: minimize the average union length of 8 intervals.
    x: shape (8,)
    """
    t, w = np.polynomial.legendre.leggauss(n_quad)
    c = 0.5 * (t + 1.0)
    i_idx = np.arange(_N)
    shifts = np.outer(c, i_idx * _INV_N)
    lo = x[None, :] + shifts
    hi = lo + _INV_N
    intervals = np.stack([lo, hi], axis=2)
    
    total_len = 0.0
    for k in range(n_quad):
        total_len += w[k] * 0.5 * _union_1d(intervals[k])
    return total_len

def kakeya_needle_3d8() -> np.ndarray:
    start = time.time()
    deadline = start + 580.0

    def time_left():
        return deadline - time.time()

    # Step 1: Solve the 1D problem to find optimal x positions
    N = _N
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(bin(idx)[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    best_1d_vol = np.inf
    best_1d_x = keich.copy()
    
    rng = np.random.default_rng(42)
    # Generate more diverse starting points
    starts = [keich.copy(), np.zeros(N), rng.normal(0, 0.5, N)]
    # Add shifted/scaled variants
    for shift in [-0.3, -0.1, 0.1, 0.3]:
        starts.append(keich.copy() + shift)
    for scale in [0.5, 1.5, 2.0]:
        starts.append(keich.copy() * scale)
    
    for s in starts:
        if time_left() < 20.0:
            break
        try:
            res = minimize(
                _obj_1d,
                s,
                method="Nelder-Mead",
                options={
                    "maxiter": 10000,
                    "xatol": 1e-14,
                    "fatol": 1e-14,
                    "adaptive": True,
                    "disp": False,
                },
            )
            if res.fun < best_1d_vol:
                best_1d_vol = res.fun
                best_1d_x = res.x.copy()
        except Exception:
            pass

    # Step 2: Initialize the 128D problem with the best 1D solution
    p0 = np.zeros(2 * _N2)
    for k in range(_N2):
        i = k // _N
        j = k % _N
        p0[k] = best_1d_x[i]
        p0[k + _N2] = best_1d_x[j]
    
    best_params = p0.copy()
    best_vol = _exact_volume(best_params)

    # Step 3: Coordinate descent on x and y separately with more iterations
    if time_left() > 80.0:
        # Optimize x coordinates
        x0 = best_params[0:_N2].copy()
        y0 = best_params[_N2:2 * _N2].copy()
        
        def obj_x(x):
            return _vol_x_only(x, y0)
        
        try:
            res_x = minimize(
                obj_x,
                x0,
                method="Powell",
                options={
                    "maxiter": 100,
                    "maxfev": 3000,
                    "ftol": 1e-15,
                    "xtol": 1e-15,
                    "disp": False,
                },
            )
            if res_x.fun < best_vol:
                best_vol = res_x.fun
                best_params[0:_N2] = res_x.x
        except Exception:
            pass

        # Optimize y coordinates
        x1 = best_params[0:_N2].copy()
        y1 = best_params[_N2:2 * _N2].copy()
        
        def obj_y(y):
            return _vol_y_only(x1, y)
        
        if time_left() > 40.0:
            try:
                res_y = minimize(
                    obj_y,
                    y1,
                    method="Powell",
                    options={
                        "maxiter": 100,
                        "maxfev": 3000,
                        "ftol": 1e-15,
                        "xtol": 1e-15,
                        "disp": False,
                    },
                )
                if res_y.fun < best_vol:
                    best_vol = res_y.fun
                    best_params[_N2:2 * _N2] = res_y.x
            except Exception:
                pass

    # Step 4: Joint polish with Powell (more iterations)
    if time_left() > 30.0:
        try:
            res = minimize(
                _exact_volume,
                best_params,
                method="Powell",
                options={
                    "maxiter": 50,
                    "maxfev": 1000,
                    "ftol": 1e-16,
                    "xtol": 1e-16,
                    "disp": False,
                },
            )
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Step 5: Nelder-Mead final polish for escaping shallow local minima
    if time_left() > 15.0:
        try:
            res = minimize(
                _exact_volume,
                best_params,
                method="Nelder-Mead",
                options={
                    "maxiter": 5000,
                    "xatol": 1e-15,
                    "fatol": 1e-15,
                    "adaptive": True,
                    "disp": False,
                },
            )
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Step 6: Final coordinate polish for highest precision
    if time_left() > 10.0:
        # One more x pass
        x_final = best_params[0:_N2].copy()
        y_final = best_params[_N2:2 * _N2].copy()
        
        def obj_x_final(x):
            return _vol_x_only(x, y_final)
        
        try:
            res_xf = minimize(
                obj_x_final,
                x_final,
                method="Powell",
                options={
                    "maxiter": 30,
                    "maxfev": 500,
                    "ftol": 1e-17,
                    "xtol": 1e-17,
                    "disp": False,
                },
            )
            if res_xf.fun < best_vol:
                best_vol = res_xf.fun
                best_params[0:_N2] = res_xf.x
        except Exception:
            pass

        # One more y pass
        if time_left() > 5.0:
            x_final2 = best_params[0:_N2].copy()
            y_final2 = best_params[_N2:2 * _N2].copy()
            
            def obj_y_final(y):
                return _vol_y_only(x_final2, y)
            
            try:
                res_yf = minimize(
                    obj_y_final,
                    y_final2,
                    method="Powell",
                    options={
                        "maxiter": 30,
                        "maxfev": 500,
                        "ftol": 1e-17,
                        "xtol": 1e-17,
                        "disp": False,
                    },
                )
                if res_yf.fun < best_vol:
                    best_vol = res_yf.fun
                    best_params[_N2:2 * _N2] = res_yf.x
            except Exception:
                pass

    x = best_params[0:_N2]
    y = best_params[_N2:2 * _N2]
    positions = np.stack([x, y], axis=1)
    return positions

# EVOLVE-BLOCK-END
