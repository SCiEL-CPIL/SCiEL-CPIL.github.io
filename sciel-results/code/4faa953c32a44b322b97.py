import math
import time

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Uses DFT-based spectral prediction for oscillatory functions:
    sample equally-spaced points, fit sinusoid via DFT, predict zero crossing,
    then refine with secant/Newton.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    tol_final = 1e-8
    max_calls = 480  # Safety margin below 500

    x0 = float(x0)
    call_count = 0

    def fc(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    f0 = fc(x0)
    if math.isfinite(f0) and abs(f0) <= tol_final:
        return x0

    # Handle non-finite f0
    if not math.isfinite(f0):
        base_step = 1e-4 * max(1.0, abs(x0))
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64]:
            if time.time() > deadline or call_count >= max_calls:
                break
            x_try = x0 + s * base_step
            f_try = fc(x_try)
            if math.isfinite(f_try) and abs(f_try) <= tol_final:
                return x_try
            if math.isfinite(f_try):
                x0, f0 = x_try, f_try
                break
        if not math.isfinite(f0):
            return x0

    best_x = x0
    best_f = abs(f0)

    # DFT-based spectral root finding
    # Sample n points over a window, compute DFT, find dominant frequency,
    # predict nearest zero crossing analytically
    n = 16  # number of sample points (power of 2 for clean DFT)
    
    # Try several window sizes
    for wf in [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
        if time.time() > deadline or call_count >= max_calls:
            break
        hw = wf * max(1.0, abs(x0))
        if hw < 1e-10:
            hw = 1e-10
        if hw > 1e5:
            hw = 1e5

        xs = []
        fs = []
        for i in range(n):
            if time.time() > deadline or call_count >= max_calls:
                break
            t = (i / (n - 1)) * 2.0 - 1.0  # -1 to 1
            xi = x0 + t * hw
            fi = fc(xi)
            if math.isfinite(fi):
                xs.append(xi)
                fs.append(fi)
                if abs(fi) < best_f:
                    best_x = xi
                    best_f = abs(fi)
                if abs(fi) <= tol_final:
                    return xi

        if len(xs) < 4:
            continue

        # Check for sign changes and do linear interpolation
        for i in range(len(xs) - 1):
            if time.time() > deadline or call_count >= max_calls:
                break
            if fs[i] * fs[i + 1] < 0:
                t = fs[i] / (fs[i] - fs[i + 1])
                x_interp = xs[i] + t * (xs[i + 1] - xs[i])
                f_interp = fc(x_interp)
                if math.isfinite(f_interp):
                    if abs(f_interp) < best_f:
                        best_x = x_interp
                        best_f = abs(f_interp)
                    if abs(f_interp) <= tol_final:
                        return x_interp

        # Fit sinusoid: f(x) ≈ p*sin(ω(x-x0)) + q*cos(ω(x-x0)) + C
        # Try several candidate frequencies (cycles in window)
        for ncyc in [1, 2, 3, 4, 6, 8]:
            if time.time() > deadline or call_count >= max_calls:
                break
            omega = 2.0 * math.pi * ncyc / (2.0 * hw)
            
            # Least squares: minimize sum (p*s_i + q*c_i + C - f_i)^2
            # Normal equations (3x3):
            # [Sss Ssc Ss1][p]   [Ssf]
            # [Ssc Scc Sc1][q] = [Scf]
            # [Ss1 Sc1 S11][C]   [S1f]
            Sss = Ssc = Ss1 = 0.0
            Scc = Sc1 = S11 = 0.0
            Ssf = Scf = S1f = 0.0
            for i in range(len(xs)):
                dx = xs[i] - x0
                s = math.sin(omega * dx)
                c = math.cos(omega * dx)
                Sss += s * s
                Ssc += s * c
                Ss1 += s
                Scc += c * c
                Sc1 += c
                S11 += 1.0
                Ssf += s * fs[i]
                Scf += c * fs[i]
                S1f += fs[i]

            # Solve 3x3 via Cramer's rule
            det = (Sss*(Scc*S11 - Sc1*Sc1) - Ssc*(Ssc*S11 - Sc1*Ss1) + Ss1*(Ssc*Sc1 - Scc*Ss1))
            if abs(det) < 1e-30:
                continue

            # p: replace col 0
            p = (Ssf*(Scc*S11 - Sc1*Sc1) - Ssc*(Ssf*S11 - Sc1*S1f) + Ss1*(Ssf*Sc1 - Scc*Ssf)) / det
            # q: replace col 1
            q = (Sss*(Ssf*S11 - Sc1*S1f) - Ssf*(Ssc*S11 - Sc1*Ss1) + Ss1*(Ssc*S1f - Scc*Ssf)) / det
            # C: replace col 2
            C = (Sss*(Scc*S1f - Sc1*Scf) - Ssc*(Ssc*S1f - Sc1*Ssf) + Ss1*(Ssc*Scf - Scc*Ssf)) / det

            # Predict zero crossings: p*sin(ω(x-x0)) + q*cos(ω(x-x0)) + C = 0
            # Let R = sqrt(p^2+q^2), phi = atan2(q, p)
            # Then R*sin(ω(x-x0) + phi) + C = 0
            # sin(ω(x-x0) + phi) = -C/R
            R = math.sqrt(p*p + q*q)
            if R < 1e-15:
                continue
            if abs(C) > R:
                continue  # No real zero crossing for this frequency

            # sin(theta) = -C/R
            theta0 = math.asin(-C / R)  # in [-pi/2, pi/2]
            # Solutions: theta = theta0 + 2k*pi or theta = pi - theta0 + 2k*pi
            # theta = omega*(x - x0) + phi
            # x = (theta - phi) / omega + x0
            
            phi = math.atan2(q, p)
            
            # Find nearest zero crossing to x0
            candidates = []
            for k in range(-3, 4):
                for base_theta in [theta0, math.pi - theta0]:
                    theta = base_theta + 2.0 * math.pi * k
                    x_candidate = (theta - phi) / omega + x0
                    candidates.append(x_candidate)
            
            # Evaluate candidates near x0
            for xc in candidates:
                if time.time() > deadline or call_count >= max_calls:
                    break
                if abs(xc - x0) > 2.0 * hw:
                    continue
                fx = fc(xc)
                if math.isfinite(fx):
                    if abs(fx) < best_f:
                        best_x = xc
                        best_f = abs(fx)
                    if abs(fx) <= tol_final:
                        return xc

        # If we improved significantly, stop searching windows
        if best_f < 0.1 * abs(f0):
            break

    # Final secant refinement from best point
    if call_count >= max_calls or time.time() > deadline:
        return best_x

    # Get a second point for secant
    step = 1e-8 * max(1.0, abs(best_x))
    x1 = best_x + step
    f1 = fc(x1)
    if not math.isfinite(f1):
        x1 = best_x - step
        f1 = fc(x1)
        if not math.isfinite(f1):
            return best_x

    if abs(f1) < best_f:
        best_x = x1
        best_f = abs(f1)
    if abs(f1) <= tol_final:
        return x1

    # Secant iterations
    if abs(f0) < abs(f1):
        xa, fa = x0, f0
        xb, fb = x1, f1
    else:
        xa, fa = x1, f1
        xb, fb = x0, f0

    for _ in range(40):
        if time.time() > deadline or call_count >= max_calls:
            break
        denom = fb - fa
        if abs(denom) < 1e-15:
            break
        xn = xb - fb * (xb - xa) / denom
        if not math.isfinite(xn):
            break
        fn = fc(xn)
        if not math.isfinite(fn):
            break
        if abs(fn) < best_f:
            best_x = xn
            best_f = abs(fn)
        if abs(fn) <= tol_final:
            return xn
        if abs(xn - xb) < 1e-15:
            break
        xa, fa = xb, fb
        xb, fb = xn, fn

    if abs(fb) <= tol_final:
        return xb
    if abs(fa) <= tol_final:
        return xa
    return best_x
