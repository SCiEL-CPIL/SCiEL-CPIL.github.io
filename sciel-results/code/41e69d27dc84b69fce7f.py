# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Uses an adaptive multi-strategy approach: secant, cubic interpolation,
    bracketing, and Nelder-Mead optimization of f(x)^2.
    """
    start_time = time.time()
    deadline = start_time + 118.0
    max_evals = 500
    evals_used = 0

    def check_time():
        return time.time() < deadline

    def safe_eval(x):
        nonlocal evals_used
        if not check_time():
            return None
        if evals_used >= max_evals:
            return None
        if not math.isfinite(x):
            return None
        try:
            val = f(x)
            evals_used += 1
            if not isinstance(val, (int, float)):
                return None
            if math.isnan(val) or math.isinf(val):
                return None
            return float(val)
        except Exception:
            evals_used += 1
            return None

    # Track best point seen so far
    best_x = float(x0)
    best_fx = safe_eval(best_x)

    if best_fx is None:
        for dx in [1e-8, -1e-8, 1e-4, -1e-4, 1.0, -1.0, 10.0, -10.0, 0.1, -0.1]:
            if not check_time():
                break
            trial_x = float(x0) + dx
            trial_fx = safe_eval(trial_x)
            if trial_fx is not None:
                if best_fx is None or abs(trial_fx) < abs(best_fx):
                    best_x = trial_x
                    best_fx = trial_fx
                if abs(best_fx) <= 1e-8:
                    return best_x
        if best_fx is None:
            return best_x

    if abs(best_fx) <= 1e-8:
        return best_x

    # ============================================================
    # Strategy 1: Secant with cubic interpolation (Brent-like)
    # ============================================================
    def secant_cubic(x0v, fx0, max_iters=80):
        nonlocal best_x, best_fx, evals_used
        if abs(fx0) <= 1e-8:
            best_x = x0v
            best_fx = fx0
            return x0v

        h = 1e-6 * max(1.0, abs(x0v))
        x1 = x0v + h
        fx1 = safe_eval(x1)
        if fx1 is None:
            x1 = x0v - h
            fx1 = safe_eval(x1)
        if fx1 is None:
            return None

        if abs(fx1) <= 1e-8:
            best_x = x1
            best_fx = fx1
            return x1

        # Use the two points as initial secant
        x_a, fx_a = x0v, fx0
        x_b, fx_b = x1, fx1

        if abs(fx_b) < abs(fx_a):
            x_a, fx_a, x_b, fx_b = x_b, fx_b, x_a, fx_a

        history = [(x_a, fx_a), (x_b, fx_b)]

        for _ in range(max_iters):
            if not check_time() or evals_used >= max_evals - 5:
                break
            if abs(fx_b) <= 1e-8:
                best_x = x_b
                best_fx = fx_b
                return x_b

            denom = fx_b - fx_a
            if abs(denom) < 1e-18:
                break

            # Secant step
            x_sec = x_b - fx_b * (x_b - x_a) / denom

            if not math.isfinite(x_sec):
                break

            max_step = 10.0 * max(1.0, abs(x_b))
            if abs(x_sec - x_b) > max_step:
                x_sec = x_b - math.copysign(max_step, x_sec - x_b)

            fx_sec = safe_eval(x_sec)
            if fx_sec is None:
                break

            if abs(fx_sec) < abs(best_fx):
                best_x = x_sec
                best_fx = fx_sec
                if abs(best_fx) <= 1e-8:
                    return x_sec

            # Try cubic interpolation if we have 3 points
            x_new = x_sec
            fx_new = fx_sec

            if len(history) >= 3:
                x1c, f1c = history[-3]
                x2c, f2c = history[-2]
                x3c, f3c = history[-1]

                if abs(x1c - x2c) > 1e-15 and abs(x2c - x3c) > 1e-15 and abs(x1c - x3c) > 1e-15:
                    den1 = (f1c - f2c) * (f1c - f3c)
                    den2 = (f2c - f1c) * (f2c - f3c)
                    den3 = (f3c - f1c) * (f3c - f2c)

                    if abs(den1) > 1e-18 and abs(den2) > 1e-18 and abs(den3) > 1e-18:
                        l1 = (f2c * f3c) / den1
                        l2 = (f1c * f3c) / den2
                        l3 = (f1c * f2c) / den3
                        x_cubic = x1c * l1 + x2c * l2 + x3c * l3

                        if math.isfinite(x_cubic) and abs(x_cubic - x3c) < 5.0 * max(1.0, abs(x3c)):
                            fx_cubic = safe_eval(x_cubic)
                            if fx_cubic is not None and abs(fx_cubic) < abs(fx_sec):
                                x_new = x_cubic
                                fx_new = fx_cubic
                                if abs(fx_cubic) < abs(best_fx):
                                    best_x = x_cubic
                                    best_fx = fx_cubic
                                    if abs(best_fx) <= 1e-8:
                                        return x_cubic

            history.append((x_new, fx_new))
            if len(history) > 3:
                history.pop(0)

            # Update: keep the point with smaller |f| as the "outer" point
            if abs(fx_new) < abs(fx_a):
                x_a, fx_a = x_b, fx_b
                x_b, fx_b = x_new, fx_new
            else:
                x_a, fx_a = x_new, fx_new

            if abs(x_b - x_a) < 1e-14 * max(1.0, abs(x_b)):
                break

        if abs(fx_b) < abs(best_fx):
            best_x = x_b
            best_fx = fx_b
        return x_b

    secant_cubic(best_x, best_fx)

    if abs(best_fx) <= 1e-8:
        return best_x

    # ============================================================
    # Strategy 2: Bracket search + Brent's method
    # ============================================================
    def brent_method(a, b, fa, fb, max_iters=100):
        nonlocal best_x, best_fx, evals_used
        if abs(fa) <= 1e-8:
            best_x = a
            best_fx = fa
            return a
        if abs(fb) <= 1e-8:
            best_x = b
            best_fx = fb
            return b

        c = b
        fc = fb
        d = b - a
        e = d
        tol_x = 1e-14
        tol_f = 1e-12

        for _ in range(max_iters):
            if not check_time() or evals_used >= max_evals - 5:
                break
            if abs(fb) <= 1e-8:
                best_x = b
                best_fx = fb
                return b

            # Check for exact zero or convergence
            if abs(b - a) < tol_x * max(1.0, abs(b)):
                break

            # Inverse quadratic interpolation
            try:
                if abs(fc) != abs(fb):
                    r = (b - c) * fb
                    s = (b - a) * fc
                    if abs(r - s) > 1e-18:
                        x_new = b - (r * (s - r)) / (2.0 * (s - r) * (fb - fc))
                        # Check if within bounds
                        if not ((x_new < min(a, b) - 1e-15) or (x_new > max(a, b) + 1e-15)):
                            if abs(x_new - b) > 0.5 * abs(b - c):
                                x_new = 0.5 * (b + c)
                            fx_new = safe_eval(x_new)
                            if fx_new is not None and abs(fx_new) < abs(fb):
                                a, b = b, x_new
                                fa, fb = fb, fx_new
                                if abs(fb) < abs(best_fx):
                                    best_x = b
                                    best_fx = fb
                                    if abs(best_fx) <= 1e-8:
                                        return b
                                c, fc = a, fa
                                d = b - c
                                e = d
                                continue
            except (ZeroDivisionError, OverflowError):
                pass

            # Bisection fallback
            x_new = 0.5 * (a + b)
            if not math.isfinite(x_new):
                break
            fx_new = safe_eval(x_new)
            if fx_new is None:
                break

            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
                if abs(best_fx) <= 1e-8:
                    return x_new

            if fa * fx_new <= 0:
                b, fb = x_new, fx_new
            else:
                a, fa = x_new, fx_new
                c, fc = a, fa
                d = b - c
                e = d

            if abs(b - a) < tol_x * max(1.0, abs(b)):
                break

        if abs(fb) < abs(best_fx):
            best_x = b
            best_fx = fb
        return b

    # Try to find a bracket
    base_scale = max(1.0, abs(best_x))
    bracket_found = False
    for scale in [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4]:
        if not check_time() or evals_used >= max_evals - 10:
            break
        delta = scale * base_scale
        xa = best_x - delta
        xb = best_x + delta
        fa = safe_eval(xa)
        if fa is None:
            continue
        fb = safe_eval(xb)
        if fb is None:
            continue
        if abs(fa) < abs(best_fx):
            best_x = xa
            best_fx = fa
            if abs(best_fx) <= 1e-8:
                return best_x
        if abs(fb) < abs(best_fx):
            best_x = xb
            best_fx = fb
            if abs(best_fx) <= 1e-8:
                return best_x
        if fa * fb < 0:
            brent_method(xa, xb, fa, fb)
            bracket_found = True
            break

    if abs(best_fx) <= 1e-8:
        return best_x

    # ============================================================
    # Strategy 3: Nelder-Mead on g(x) = f(x)^2
    # ============================================================
    def nelder_mead(x0_nm, fx0, max_iters=200):
        nonlocal best_x, best_fx, evals_used
        if abs(fx0) <= 1e-8:
            best_x = x0_nm
            best_fx = fx0
            return x0_nm

        h = 1e-4 * max(1.0, abs(x0_nm))
        if h < 1e-8:
            h = 1e-8

        # Initial simplex: [x0, x0+h, x0-h]
        simplex = [
            [x0_nm, fx0 * fx0],
            [x0_nm + h, None],
            [x0_nm - h, None]
        ]

        for i in range(1, 3):
            if not check_time() or evals_used >= max_evals - 3:
                break
            fi = safe_eval(simplex[i][0])
            if fi is None:
                simplex[i][1] = float('inf')
            else:
                simplex[i][1] = fi * fi
                if abs(fi) < abs(best_fx):
                    best_x = simplex[i][0]
                    best_fx = fi
                    if abs(best_fx) <= 1e-8:
                        return best_x

        simplex.sort(key=lambda p: p[1])

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        for _ in range(max_iters):
            if not check_time() or evals_used >= max_evals - 5:
                break

            x0_s, f0 = simplex[0]
            x1_s, f1 = simplex[1]
            x2_s, f2 = simplex[2]

            if abs(math.sqrt(f0)) <= 1e-8:
                best_x = x0_s
                best_fx = math.sqrt(f0)
                return best_x

            # Convergence check
            if abs(f2 - f0) < 1e-18 * max(1.0, abs(f0)):
                break

            # Centroid of two best points
            xc = x0_s + 0.5 * (x1_s - x0_s)

            # Reflection
            xr = xc + alpha * (xc - x2_s)
            fr = safe_eval(xr)
            if fr is None:
                fr = float('inf')
            else:
                fr = fr * fr
                if abs(fr) < abs(best_fx * best_fx):
                    best_x = xr
                    best_fx = math.sqrt(fr)
                    if abs(best_fx) <= 1e-8:
                        return best_x

            if f0 < fr < f1:
                simplex[2] = [xr, fr]
                simplex.sort(key=lambda p: p[1])
                continue

            # Expansion
            xe = xc + gamma * (xc - x2_s)
            fe = safe_eval(xe)
            if fe is None:
                fe = float('inf')
            else:
                fe = fe * fe
                if abs(fe) < abs(best_fx * best_fx):
                    best_x = xe
                    best_fx = math.sqrt(fe)
                    if abs(best_fx) <= 1e-8:
                        return best_x

            if fr < fe and fr < f0:
                simplex[2] = [xe, fe]
                simplex.sort(key=lambda p: p[1])
                continue

            # Contraction
            if fr < f2:
                xc_r = xc + rho * (xc - x2_s)
            else:
                xc_r = xc - rho * (xc - x2_s)

            fc_r = safe_eval(xc_r)
            if fc_r is None:
                fc_r = float('inf')
            else:
                fc_r = fc_r * fc_r
                if abs(fc_r) < abs(best_fx * best_fx):
                    best_x = xc_r
                    best_fx = math.sqrt(fc_r)
                    if abs(best_fx) <= 1e-8:
                        return best_x

            if fc_r < min(fr, f2):
                simplex[2] = [xc_r, fc_r]
                simplex.sort(key=lambda p: p[1])
                continue

            # Shrink
            for i in range(1, 3):
                simplex[i][0] = x0_s + sigma * (simplex[i][0] - x0_s)
                fi = safe_eval(simplex[i][0])
                if fi is None:
                    simplex[i][1] = float('inf')
                else:
                    simplex[i][1] = fi * fi
                    if abs(fi) < abs(best_fx):
                        best_x = simplex[i][0]
                        best_fx = fi
                        if abs(best_fx) <= 1e-8:
                            return best_x

            simplex.sort(key=lambda p: p[1])

        if abs(math.sqrt(simplex[0][1])) < abs(best_fx):
            best_x = simplex[0][0]
            best_fx = math.sqrt(simplex[0][1])
        return best_x

    if check_time() and evals_used < max_evals - 30:
        nelder_mead(best_x, best_fx)

    if abs(best_fx) <= 1e-8:
        return best_x

    # ============================================================
    # Strategy 4: Multi-start secant from different perturbation points
    # ============================================================
    if check_time() and evals_used < max_evals - 20:
        base_scale = max(1.0, abs(best_x))
        perturbations = [
            1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0,
            5.0, -5.0, 0.01, -0.01
        ]
        for scale in perturbations:
            if not check_time() or evals_used >= max_evals - 10:
                break
            dx = scale * base_scale
            trial_x = best_x + dx
            trial_fx = safe_eval(trial_x)
            if trial_fx is None:
                continue
            if abs(trial_fx) < abs(best_fx):
                best_x = trial_x
                best_fx = trial_fx
            if abs(best_fx) <= 1e-8:
                return best_x
            # Quick secant from this point
            h2 = 1e-5 * max(1.0, abs(trial_x))
            x2_trial = trial_x + h2
            fx2_trial = safe_eval(x2_trial)
            if fx2_trial is None:
                x2_trial = trial_x - h2
                fx2_trial = safe_eval(x2_trial)
            if fx2_trial is None:
                continue
            if abs(fx2_trial) < abs(best_fx):
                best_x = x2_trial
                best_fx = fx2_trial
            if abs(best_fx) <= 1e-8:
                return best_x
            # One secant step
            if abs(fx2_trial - trial_fx) > 1e-18:
                x_sec_trial = x2_trial - fx2_trial * (x2_trial - trial_x) / (fx2_trial - trial_fx)
                if math.isfinite(x_sec_trial):
                    fx_sec_trial = safe_eval(x_sec_trial)
                    if fx_sec_trial is not None and abs(fx_sec_trial) < abs(best_fx):
                        best_x = x_sec_trial
                        best_fx = fx_sec_trial
                        if abs(best_fx) <= 1e-8:
                            return best_x

    # ============================================================
    # Strategy 5: Final high-precision polish
    # ============================================================
    if abs(best_fx) <= 1e-4 and check_time() and evals_used < max_evals - 5:
        x = best_x
        fx = best_fx
        for _ in range(40):
            if not check_time() or evals_used >= max_evals - 2:
                break
            if abs(fx) <= 1e-10:
                return x
            h = 1e-9 * max(1.0, abs(x))
            if abs(fx) < 1e-7:
                h = 1e-11 * max(1.0, abs(x))
            x2 = x + h
            fx2 = safe_eval(x2)
            if fx2 is None:
                x2 = x - h
                fx2 = safe_eval(x2)
            if fx2 is None:
                break
            if abs(fx2 - fx) < 1e-20:
                break
            x_new = x - fx * h / (fx2 - fx)
            if not math.isfinite(x_new):
                break
            max_step = 5.0 * max(1.0, abs(x))
            if abs(x_new - x) > max_step:
                x_new = x - math.copysign(max_step, x_new - x)
            fx_new = safe_eval(x_new)
            if fx_new is None:
                break
            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
            if abs(fx_new) <= 1e-9:
                return x_new
            x = x_new
            fx = fx_new

    return best_x
# EVOLVE-BLOCK-END
