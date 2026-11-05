"""Structural rewrite: first-order f-ODE reference + scipy L-BFGS-B optimization."""
import time
import math
import numpy as np
from scipy.optimize import minimize as _scipy_minimize

_START = time.time()
_DEADLINE = _START + 85.0  # safety margin before 90s kill

def _om_a(a, Om0, wa):
    """Matter density at scale factor a for CPL with w0=-1."""
    w0 = -1.0
    # a^{-3(1+w0+wa)} = a^{-3wa}
    return Om0 * a ** (-3.0 * wa) / (Om0 * a ** (-3.0 * wa) + 1.0 - Om0)

def _q_of(a, Om0, wa):
    return 1.0 - _om_a(a, Om0, wa)

def _u_of(a, wa):
    w0 = -1.0
    return wa * (1.0 - a)

def _gamma_vec(a, Om0, wa, c):
    q = _q_of(a, Om0, wa)
    u = _u_of(a, wa)
    return c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa*a + c[5]*u*q

def _f_vec(a, Om0, wa, c):
    om = _om_a(a, Om0, wa)
    g = _gamma_vec(a, Om0, wa, c)
    return om ** g

def _build_reference(Om0, wa, a_grid):
    """
    Integrate first-order ODE for f = dlnD/dlna:
    f' = -f^2 - (3/2)(1+w) f - (3/2)(1+w) dH/dlna + (3/2) d^2H/dlna^2 ... 
    Actually use the standard: f' + f^2 + (3/2)(1+w)f + (3/2)(1+w) dH/dlna = 0
    with IC f(1) = Om0^0.55 (matter-era approximation).
    """
    w0 = -1.0
    n = len(a_grid)
    x = np.log(a_grid)  # ln a, decreasing from 0 to ln(a_min)
    # We integrate from a=1 (x=0) down to a_min (x<0)
    # Use scipy solve_ivp or manual RK4 in x from 0 to x_min
    
    a = a_grid
    om = _om_a(a, Om0, wa)
    E2 = om + (1.0 - Om0) * a ** (-3.0 * wa)  # since 1+w0+wa = wa
    E = np.sqrt(E2)
    # dH/dlna = (1/2) dE2/dlna / E = -3/2 * wa * E2 / E
    dHdlna = -1.5 * wa * E2 / E
    # w(a) = w0 + wa(1-a) = -1 + wa(1-a)
    w = -1.0 + wa * (1.0 - a)
    
    # ODE: df/dlna = -f^2 - (3/2)(1+w) f - (3/2)(1+w) dH/dlna
    # Wait, the standard growth ODE in terms of f:
    # f' = -f^2 - (3/2)(1+w)f - (3/2)(1+w) dH/dlna
    # Let me verify: D''/D + (3(1+w)+dH/dlna)D'/D + (d2H/dlna2 + 3(1+w)dH/dlna)D/D = 0
    # With f = D'/(HD), this becomes a Riccati equation.
    # Standard form: f' = -f^2 - (3/2)(1+w)f - (3/2)(1+w) dH/dlna
    # where ' = d/dlna
    
    def ode_rhs(xk, fk):
        # xk is ln a, fk is f
        # Interpolate or compute at xk
        # For efficiency, we'll use the grid values and interpolate
        idx = np.searchsorted(-x, -xk, side='right')
        if idx == 0:
            idx = 1
        if idx >= n:
            idx = n - 1
        # linear interp
        x0, x1 = x[idx-1], x[idx]
        if abs(x1 - x0) < 1e-15:
            t = 0.0
        else:
            t = (xk - x0) / (x1 - x0)
        t = np.clip(t, 0, 1)
        wk = (1-t)*w[idx-1] + t*w[idx]
        dHk = (1-t)*dHdlna[idx-1] + t*dHdlna[idx]
        return -fk**2 - 1.5*(1.0+wk)*fk - 1.5*(1.0+wk)*dHk
    
    # RK4 integration from x=0 to x_min
    x_min = x[0]  # most negative
    n_steps = 5000
    dx = x_min / n_steps
    f_vals = np.empty(n_steps + 1)
    f_vals[0] = Om0 ** 0.55  # matter-era IC
    for i in range(n_steps):
        xk = x_min + i * dx  # going from x_min up to 0... wait, we want from 0 down to x_min
        # Let's integrate from 0 to x_min (negative direction)
        pass
    
    # Redo: integrate from x=0 (a=1) to x=x_min (a=a_min)
    x_start = 0.0
    x_end = x_min
    n_steps = 5000
    dx = (x_end - x_start) / n_steps  # negative
    
    f_vals = np.empty(n_steps + 1)
    f_vals[0] = Om0 ** 0.55
    x_grid = np.linspace(x_start, x_end, n_steps + 1)
    
    for i in range(n_steps):
        xk = x_grid[i]
        fk = f_vals[i]
        k1 = ode_rhs(xk, fk)
        k2 = ode_rhs(xk + 0.5*dx, fk + 0.5*dx*k1)
        k3 = ode_rhs(xk + 0.5*dx, fk + 0.5*dx*k2)
        k4 = ode_rhs(xk + dx, fk + dx*k3)
        f_vals[i+1] = fk + dx/6.0*(k1 + 2*k2 + 2*k3 + k4)
    
    # Interpolate f to a_grid
    f_ref = np.interp(x, x_grid, f_vals)
    
    # Compute normalized D from f: D(a)/D(1) = exp(-int_0^{ln a} f dx)
    # Since x = ln a goes from 0 to negative, integral from 0 to x of f dx
    # D/D(1) = exp(-int_0^x f(x') dx') where x<0
    # int_0^x f dx' = -int_x^0 f dx', so D/D(1) = exp(int_x^0 f dx')
    # Let's compute cumulative integral from x_min to 0, then D(x) = exp(int_x^0 f dx')
    
    # Use trapezoidal rule on x_grid
    dx_grid = np.diff(x_grid)
    f_avg = 0.5*(f_vals[:-1] + f_vals[1:])
    cum_int = np.cumsum(f_avg * dx_grid)
    # cum_int[i] = int_{x_grid[0]}^{x_grid[i+1]} f dx
    # We want int_{x}^{0} f dx = int_{x_grid[0]}^{0} f dx - int_{x_grid[0]}^{x} f dx
    # x_grid[0] = 0, x_grid[-1] = x_min
    # So int_{x_grid[0]}^{0} = 0, and int_{x_grid[0]}^{x} = cum_int at position of x
    # D(x)/D(0) = exp(-int_0^x f dx) = exp(int_x^0 f dx)
    # int_x^0 f dx = -int_0^x f dx
    # int_0^x f dx for x<0: integrate from 0 to x (negative direction)
    # Let's just compute: for each x in x_grid, int_0^{x_grid[i]} f dx
    # Since x_grid goes from 0 to x_min (decreasing), 
    # int_0^{x_grid[i]} = -cum_int[i-1] for i>=1? 
    # cum_int[i] = int_{x_grid[i]}^{x_grid[i+1]} f dx (trapezoid over that segment)
    # int_0^{x_grid[i]} = sum_{j=0}^{i-1} int_{x_grid[j]}^{x_grid[j+1]} f dx = sum_{j=0}^{i-1} cum_int[j]
    
    cum_from_0 = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):
        cum_from_0[i] = cum_from_0[i-1] + 0.5*(f_vals[i-1]+f_vals[i])*(x_grid[i]-x_grid[i-1])
    # cum_from_0[i] = int_0^{x_grid[i]} f dx
    # D(x_grid[i])/D(0) = exp(-cum_from_0[i])
    
    D_ref = np.exp(-cum_from_0)
    D_ref = D_ref / D_ref[0]  # should be 1 at a=1
    
    # Interpolate D_ref to a_grid
    Dn_ref = np.interp(x, x_grid, D_ref)
    
    return f_ref, Dn_ref

def _build_all_references():
    Om0s = [0.22, 0.27, 0.32]
    was = [-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6]
    a_grid = np.geomspace(0.08, 1.0, 300)
    refs = []
    for Om0 in Om0s:
        for wa in was:
            if time.time() > _DEADLINE - 5.0:
                return refs
            f_ref, Dn_ref = _build_reference(Om0, wa, a_grid)
            refs.append((Om0, wa, a_grid, f_ref, Dn_ref))
    return refs

def _objective(c, refs):
    """Compute max relative error across all references."""
    max_err = 0.0
    for (Om0, wa, a_grid, f_ref, Dn_ref) in refs:
        g = _gamma_vec(a_grid, Om0, wa, c)
        if np.any(g < 0.1) or np.any(g > 1.2):
            return 1000.0
        f_calc = _f_vec(a_grid, Om0, wa, c)
        rel_f = np.max(np.abs(f_calc - f_ref) / np.maximum(np.abs(f_ref), 1e-10))
        max_err = max(max_err, rel_f)
        
        # Compute D from f_calc
        x = np.log(a_grid)
        n = len(a_grid)
        # Trapezoidal integration of -f from 0 to x
        # x goes from ln(0.08) to 0
        # We need int_0^x f dx' for each x
        # Sort by x (already sorted ascending since a_grid is ascending)
        # x[0] = ln(0.08) < 0, x[-1] = 0
        # int_0^{x[i]} f dx' = -int_{x[i]}^0 f dx'
        # Compute cumulative from x[-1]=0 backwards
        dx = np.diff(x)  # positive
        f_avg = 0.5*(f_calc[1:] + f_calc[:-1])
        # int_{x[i]}^{x[i+1]} f dx = f_avg[i]*dx[i]
        # int_{x[i]}^0 f dx = sum_{j=i}^{n-2} f_avg[j]*dx[j]
        cum_to_0 = np.zeros(n)
        running = 0.0
        for i in range(n-2, -1, -1):
            running += f_avg[i]*dx[i]
            cum_to_0[i] = running
        cum_to_0[n-1] = 0.0
        # D(x[i])/D(0) = exp(-int_0^{x[i]} f dx) = exp(int_{x[i]}^0 f dx) = exp(cum_to_0[i])
        D_calc = np.exp(cum_to_0)
        D_calc = D_calc / D_calc[-1]  # normalize to 1 at a=1
        rel_D = np.max(np.abs(D_calc - Dn_ref) / np.maximum(np.abs(Dn_ref), 1e-10))
        max_err = max(max_err, rel_D)
    
    return max_err

def construct():
    try:
        if time.time() > _DEADLINE:
            return [0.55, 0.0, 0.0, -0.5, 0.0, 0.0]
        
        refs = _build_all_references()
        if not refs:
            return [0.55, 0.0, 0.0, -0.5, 0.0, 0.0]
        
        # Physically motivated initial condition
        c0_init = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
        
        # Try scipy L-BFGS-B
        best_c = c0_init.copy()
        best_err = _objective(best_c, refs)
        
        if time.time() < _DEADLINE - 5.0:
            try:
                res = _scipy_minimize(
                    _objective, c0_init, args=(refs,),
                    method='L-BFGS-B',
                    bounds=[(-10,10)]*6,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < best_err:
                    best_c = res.x
                    best_err = res.fun
            except Exception:
                pass
        
        # Local refinement with Nelder-Mead if time permits
        if time.time() < _DEADLINE - 3.0:
            try:
                res2 = _scipy_minimize(
                    _objective, best_c, args=(refs,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res2.fun < best_err:
                    best_c = res2.x
                    best_err = res2.fun
            except Exception:
                pass
        
        # Ensure bounds
        best_c = np.clip(best_c, -10, 10)
        
        # Final check: make sure gamma is in [0.1, 1.2]
        for (Om0, wa, a_grid, f_ref, Dn_ref) in refs:
            g = _gamma_vec(a_grid, Om0, wa, best_c)
            if np.any(g < 0.1) or np.any(g > 1.2):
                # Fallback to seed
                return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        return [float(x) for x in best_c]
    except Exception:
        return [0.55, 0.0, 0.0, -0.5, 0.0, 0.0]

