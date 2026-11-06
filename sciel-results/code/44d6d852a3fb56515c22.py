"""Direct approach for ill-conditioned root finding with multiple strategies."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 128.0  # safety margin before 150s kill
    max_evals = 500
    TOL = 1e-8
    eval_count = 0
    best_x = float(x0)
    best_fabs = abs(f(best_x))
    eval_count += 1

    def call_f(x):
        nonlocal eval_count, best_x, best_fabs
        if eval_count >= max_evals:
            raise Exception("max evals")
        if time.time() > deadline:
            raise Exception("deadline")
        val = f(x)
        eval_count += 1
        if not (isinstance(val, (int, float)) and math.isfinite(val)):
            val = float('nan')
        fabs = abs(val)
        if math.isfinite(fabs) and fabs < best_fabs:
            best_fabs = fabs
            best_x = x
        return val

    def check_time():
        return time.time() < deadline

    def secant_method(x0, tol=TOL, max_iter=200):
        """Secant method with careful handling of ill-conditioning."""
        x1 = float(x0)
        h_base = 1e-5 * max(1.0, abs(x1))
        
        best_local_x = x1
        best_local_fabs = abs(call_f(x1))
        
        if best_local_fabs <= tol:
            return True
        
        # Try several initial perturbations to find a good secant pair
        for h_factor in [1.0, -1.0, 10.0, -10.0, 0.1, -0.1, 100.0]:
            if eval_count >= max_evals - 5:
                break
            h = h_factor * h_base
            x2 = x1 + h
            if not math.isfinite(x2):
                continue
            try:
                f2 = call_f(x2)
            except Exception:
                continue
            if abs(f2) <= tol:
                return True
            if abs(f2) < best_local_fabs:
                best_local_fabs = abs(f2)
                best_local_x = x2
            
            # If we have a good pair, start secant
            f1 = call_f(x1) if eval_count < max_evals else None
            if f1 is None:
                continue
            if abs(f1) <= tol:
                return True
            denom = f2 - f1
            if abs(denom) > 1e-30 and abs(denom) < 1e10:
                # Good secant pair, proceed
                break
        
        # Standard secant iteration
        x1 = float(x0)
        h = h_base
        x2 = x1 + h
        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
        except Exception:
            return False

        if abs(f1) <= tol:
            return True
        if abs(f2) <= tol:
            return True

        for i in range(max_iter):
            if not check_time():
                return False
            if abs(f2) <= tol:
                return True
            if abs(f1) <= tol:
                return True

            denom = f2 - f1
            if abs(denom) < 1e-30:
                # Try a different perturbation
                h2 = h_base * (1 + 0.1 * i) * (1 if i % 2 == 0 else -1)
                x2_new = x2 + h2
                try:
                    f2 = call_f(x2_new)
                except Exception:
                    return False
                x2 = x2_new
                denom = f2 - f1
                if abs(denom) < 1e-30:
                    return False

            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False

            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False

        return abs(f1) <= tol or abs(f2) <= tol

    def newton_forward(x0, tol=TOL, max_iter=100):
        """Newton's method with forward finite difference for derivative."""
        x = float(x0)

        for i in range(max_iter):
            if not check_time():
                return False
            try:
                fx = call_f(x)
            except Exception:
                return False
            if abs(fx) <= tol:
                return True

            # Forward difference with adaptive step
            h = 1e-6 * max(1.0, abs(x))
            try:
                fp = call_f(x + h)
            except Exception:
                return False

            d = (fp - fx) / h
            if abs(d) < 1e-30:
                # Derivative too small, try larger step
                h2 = 1e-3 * max(1.0, abs(x))
                try:
                    fp2 = call_f(x + h2)
                    d = (fp2 - fx) / h2
                except Exception:
                    return False
                if abs(d) < 1e-30:
                    return False

            step = fx / d
            # Tighter damped step (max 5x) to avoid divergence
            max_step = 5.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)

            x_new = x - step
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False

            x = x_new

        try:
            fx = call_f(x)
        except Exception:
            return False
        return abs(fx) <= tol

    def bisection(x0, tol=TOL, max_iter=200):
        """Bisection with bracket search."""
        x = float(x0)
        try:
            fx = call_f(x)
        except Exception:
            return False
        if abs(fx) <= tol:
            return True

        # Search for a bracket
        step = 1e-3 * max(1.0, abs(x))
        a, b = None, None
        fa, fb = None, None

        # Search right
        for i in range(40):
            x_test = x + step * (2 ** i)
            try:
                f_test = call_f(x_test)
            except Exception:
                break
            if not math.isfinite(f_test):
                break
            if fa is None:
                a, fa = x, fx
            if fa * f_test < 0:
                b, fb = x_test, f_test
                break

        # Search left if not found
        if b is None:
            a, b = None, None
            fa, fb = None, None
            for i in range(40):
                x_test = x - step * (2 ** i)
                try:
                    f_test = call_f(x_test)
                except Exception:
                    break
                if not math.isfinite(f_test):
                    break
                if fb is None:
                    b, fb = x, fx
                if fb * f_test < 0:
                    a, fa = x_test, f_test
                    break

        if a is None or b is None:
            return False

        # Bisection
        for i in range(max_iter):
            if not check_time():
                return False
            mid = (a + b) / 2
            try:
                fm = call_f(mid)
            except Exception:
                return False
            if abs(fm) <= tol:
                return True
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break

        return min(abs(fa), abs(fb)) <= tol

    def bfgs_minimize(x0, tol=TOL, max_iter=100):
        """
        BFGS quasi-Newton method on g(x) = f(x)^2.
        Uses finite-difference gradient and maintains curvature approximation.
        For 1D, BFGS reduces to a secant-like method with adaptive step control.
        """
        x = float(x0)
        
        # Initial gradient estimate via central difference
        h0 = 1e-5 * max(1.0, abs(x))
        try:
            f_plus = call_f(x + h0)
            f_minus = call_f(x - h0)
        except Exception:
            return False
        
        g_plus = f_plus * f_plus
        g_minus = f_minus * f_minus
        grad = (g_plus - g_minus) / (2.0 * h0)
        
        # BFGS Hessian approximation (scalar in 1D)
        # H = 1 / H_inv, initialize H_inv = 1.0
        H_inv = 1.0
        
        best_local_x = x
        best_local_g = g_plus if g_plus < g_minus else g_minus
        
        for i in range(max_iter):
            if not check_time():
                break
            if eval_count >= max_evals - 10:
                break
            
            # Update best
            fx_now = call_f(x) if eval_count < max_evals else None
            if fx_now is not None:
                g_now = fx_now * fx_now
                if g_now < best_local_g:
                    best_local_g = g_now
                    best_local_x = x
            
            if best_local_g <= tol * tol:
                if best_local_g < best_fabs * best_fabs:
                    best_x = best_local_x
                    best_fabs = math.sqrt(best_local_g)
                return True
            
            # Recompute gradient
            h = h0 * (1.0 + 0.1 * (i % 10))
            try:
                fp = call_f(x + h)
                fm = call_f(x - h)
            except Exception:
                break
            
            gp = fp * fp
            gm = fm * fm
            g_new = (gp - gm) / (2.0 * h)
            
            # BFGS update in 1D: H_inv_new = H_inv + (s*s)/(s*grad_new - s*grad_old) style
            # In 1D, BFGS: H_inv = H_inv * (1 + (g_new - g_old)*H_inv / max(|g_old|, eps))
            # Simpler: just use the secant approximation for the inverse Hessian
            if abs(g_new - grad) > 1e-30:
                # BFGS 1D update: H_inv_new = (H_inv + s^2 / (s * (g_new - g_old)))
                # But we haven't taken a step yet. Use gradient ratio instead.
                ratio = g_new / max(abs(grad), 1e-30)
                if ratio > 0.1 and ratio < 10.0:
                    H_inv = 1.0 / ratio
            
            # Step direction
            if abs(grad) < 1e-30:
                # Gradient too small, try larger h
                h_large = 1e-3 * max(1.0, abs(x))
                try:
                    fp2 = call_f(x + h_large)
                    fm2 = call_f(x - h_large)
                except Exception:
                    break
                gp2 = fp2 * fp2
                gm2 = fm2 * fm2
                grad = (gp2 - gm2) / (2.0 * h_large)
                if abs(grad) < 1e-30:
                    break
            
            # Step size with line search (backtracking)
            step = -grad * H_inv
            
            # Damp the step
            max_step = 10.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            
            # Backtracking line search on g(x) = f(x)^2
            alpha = 1.0
            x_trial = x + step
            g_trial = None
            
            for ls in range(20):
                if not check_time():
                    break
                try:
                    ft = call_f(x_trial)
                    g_trial = ft * ft
                except Exception:
                    break
                
                if g_trial is not None and g_trial < best_local_g:
                    break
                
                if g_trial is not None and g_trial < (best_local_g - 0.5 * grad * step * alpha) if abs(grad * step) > 1e-30 else True:
                    break
                
                alpha *= 0.5
                x_trial = x + step * alpha
                if alpha < 1e-12:
                    break
            
            if g_trial is None or not math.isfinite(g_trial):
                break
            
            # Accept step if it improves
            if g_trial < best_local_g * 1.01:
                x = x_trial
                grad = g_new  # approximate
                if g_trial < best_local_g:
                    best_local_g = g_trial
                    best_local_x = x
            
            # Check convergence
            if abs(step * alpha) < 1e-14 * max(1.0, abs(x)):
                break
        
        # Update global best
        if best_local_g < best_fabs * best_fabs:
            best_x = best_local_x
            best_fabs = math.sqrt(best_local_g)
        
        return best_fabs <= tol

    def nelder_mead_minimize(x0, tol=TOL, max_iter=300):
        """
        Nelder-Mead simplex method to minimize |f(x)| directly.
        This is derivative-free and handles ill-conditioning well
        by adaptively scaling the simplex.
        """
        x = float(x0)
        scale = 1e-3 * max(1.0, abs(x))

        # Initial simplex: 2 points in 1D
        x_a = x
        x_b = x + scale

        try:
            fa = abs(call_f(x_a))
            fb = abs(call_f(x_b))
        except Exception:
            return False

        if fa <= tol:
            return True
        if fb <= tol:
            return True

        # Track best across all iterations
        nm_best_x = x_a if fa < fb else x_b
        nm_best_fabs = min(fa, fb)

        no_improve_count = 0
        restart_scale = scale

        for i in range(max_iter):
            if not check_time():
                break

            # Ensure fa <= fb (a is the best point)
            if fa > fb:
                x_a, x_b = x_b, x_a
                fa, fb = fb, fa

            if fa <= tol:
                return True

            # Update global best
            if fa < nm_best_fabs:
                nm_best_fabs = fa
                nm_best_x = x_a
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Adaptive restart: if no improvement for 15 iters, try larger scale
            if no_improve_count >= 15 and i > 10:
                restart_scale *= 3.0
                x_b = x_a + restart_scale * (1 if i % 2 == 0 else -1)
                try:
                    fb = abs(call_f(x_b))
                except Exception:
                    break
                if fb <= tol:
                    return True
                if fb < fa:
                    x_a, x_b = x_b, x_a
                    fa, fb = fb, fa
                no_improve_count = 0
                continue

            # Reflection
            x_r = x_a + (x_a - x_b)
            try:
                fr = abs(call_f(x_r))
            except Exception:
                break

            if fr <= tol:
                return True

            if fr < fa:
                # Reflection improved, expand
                x_e = x_a + 2.0 * (x_a - x_b)
                try:
                    fe = abs(call_f(x_e))
                except Exception:
                    break

                if fe <= tol:
                    return True

                if fe < fr:
                    x_a, x_b = x_e, x_a
                    fa, fb = fe, fr
                else:
                    x_b = x_a
                    x_a = x_r
                    fa, fb = fr, fa
            elif fr < fb:
                # Reflection is between a and b
                x_b = x_r
                fb = fr
            else:
                # Contraction
                x_c = x_a + 0.5 * (x_b - x_a)
                try:
                    fc = abs(call_f(x_c))
                except Exception:
                    break

                if fc <= tol:
                    return True

                if fc < fb:
                    x_b = x_c
                    fb = fc
                else:
                    # Shrink
                    x_b = x_a + 0.5 * (x_b - x_a)
                    try:
                        fb = abs(call_f(x_b))
                    except Exception:
                        break

                    # If simplex is too small, try larger scale
                    if abs(x_b - x_a) < 1e-12 * max(1.0, abs(x_a)):
                        # Restart with larger scale
                        new_scale = 1e-2 * max(1.0, abs(x_a)) * (1 + 0.1 * (i % 10))
                        x_b = x_a + new_scale
                        try:
                            fb = abs(call_f(x_b))
                        except Exception:
                            break

            # Check if converged
            if abs(x_b - x_a) < 1e-14 * max(1.0, abs(x_a)):
                if fb > fa:
                    x_a, x_b = x_b, x_a
                    fa, fb = fb, fa
                break

        # Update global best one more time
        if fa < nm_best_fabs:
            nm_best_fabs = fa
            nm_best_x = x_a
        if fb < nm_best_fabs:
            nm_best_fabs = fb
            nm_best_x = x_b

        # If Nelder-Mead found a good point, use it for final polish
        if nm_best_fabs < best_fabs:
            best_x = nm_best_x
            best_fabs = nm_best_fabs

        return nm_best_fabs <= tol

    def polish_pass(x_start, tol=TOL, max_iter=80):
        """
        High-precision polish using secant method starting from a good initial guess.
        Uses tighter tolerance and more careful step sizing.
        """
        x1 = float(x_start)
        h = 1e-8 * max(1.0, abs(x1))
        x2 = x1 + h

        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
        except Exception:
            return False

        if abs(f1) <= tol:
            return True
        if abs(f2) <= tol:
            return True

        # Keep track of three points for potential IQUO
        x3 = x1 - h
        try:
            f3 = call_f(x3)
        except Exception:
            f3 = None

        if f3 is not None and abs(f3) <= tol:
            return True

        for i in range(max_iter):
            if not check_time():
                break
            if abs(f2) <= tol:
                return True
            if abs(f1) <= tol:
                return True
            if f3 is not None and abs(f3) <= tol:
                return True

            denom = f2 - f1
            x_new = None
            
            # Try Inverse Quadratic Interpolation (IQI) for cubic convergence
            if f3 is not None and i >= 1:
                d1 = x2 - x1
                d2 = x3 - x1
                if abs(d1) > 1e-15 and abs(d2) > 1e-15 and abs(d1 - d2) > 1e-15:
                    # IQI: find x such that the quadratic through (x1,f1),(x2,f2),(x3,f3) is zero
                    try:
                        # Standard IQI formula
                        L1 = d1 * d1 / (f1 * (f2 - f1) * (f3 - f1)) if f1 != 0 and f2 != f1 and f3 != f1 else 0
                        L2 = d2 * d2 / (f2 * (f3 - f2) * (f1 - f2)) if f2 != 0 and f3 != f2 and f1 != f2 else 0
                        L3 = d1 * d2 / (f3 * (f1 - f3) * (f2 - f3)) if f3 != 0 and f1 != f3 and f2 != f3 else 0
                        
                        # Alternative: use the Lagrange-based IQI
                        # x_iq = x1 + 1/((x2-x1)/(f2-f1) - (x3-x1)/(f3-f1))
                        A_val = (x2 - x1) / (f2 - f1) if abs(f2 - f1) > 1e-30 else 0
                        B_val = (x3 - x1) / (f3 - f1) if abs(f3 - f1) > 1e-30 else 0
                        
                        if abs(A_val) > 1e-30 and abs(B_val) > 1e-30 and abs(A_val - B_val) > 1e-30:
                            x_iq = x1 + 1.0 / (A_val - B_val)
                            if math.isfinite(x_iq) and abs(x_iq) < 1e15:
                                # Ensure IQI point is near the bracket
                                if abs(x_iq - x1) < 2.0 * abs(d1) + 2.0 * abs(d2):
                                    x_new = x_iq
                    except (ZeroDivisionError, OverflowError, ValueError):
                        pass
            
            # Fall back to secant
            if x_new is None:
                if abs(denom) < 1e-30:
                    h2 = 1e-8 * max(1.0, abs(x2)) * (1 + 0.01 * i)
                    x2_new = x2 + h2
                    try:
                        f2 = call_f(x2_new)
                    except Exception:
                        break
                    x2 = x2_new
                    denom = f2 - f1
                    if abs(denom) < 1e-30:
                        break
                x_new = x2 - f2 * (x2 - x1) / denom
            
            if not math.isfinite(x_new):
                break
            if abs(x_new) > 1e15:
                break

            # Update points (keep the two closest to the root)
            x3, f3 = x1, f1
            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                break

        return abs(f1) <= tol or abs(f2) <= tol

    def hybrid_ill_conditioned(x0, tol=TOL):
        """
        Budget-aware hybrid approach for ill-conditioned problems:
        1. Secant method (most efficient for smooth functions)
        2. Newton with adaptive damping
        3. Bisection (only if budget remains)
        4. Nelder-Mead minimization (derivative-free)
        5. Final polish pass on best point found
        """
        # Strategy 1: Secant method
        if secant_method(x0):
            return True

        # Strategy 2: Newton with forward difference
        if newton_forward(x0):
            return True

        # Strategy 3: Bisection - only if we have enough budget left
        # Bisection bracket search can use up to ~80 evals, so require 150 remaining
        if eval_count < max_evals - 150:
            if bisection(x0):
                return True

        # Strategy 4: BFGS on g(x) = f(x)^2
        if bfgs_minimize(x0):
            return True

        # Strategy 5: Nelder-Mead minimization (derivative-free)
        if nelder_mead_minimize(x0):
            return True

        # Strategy 6: Polish pass on best point found so far
        if best_fabs < TOL * 100 and best_fabs > 0:
            if polish_pass(best_x):
                return True

        # Strategy 7: Second polish pass from x0 with different initial h
        if best_fabs < TOL * 1000 and best_fabs > TOL:
            if polish_pass(x0):
                return True

        return False

    # Try the hybrid approach
    hybrid_ill_conditioned(x0)

    # Final polish: if we're close but not quite there, do one more polish pass
    if best_fabs < TOL * 1000 and best_fabs > TOL:
        polish_pass(best_x)
    
    # If still not converged, try IQI-enhanced secant from the best point
    if best_fabs > TOL and best_fabs < TOL * 1e6:
        x1 = best_x
        h = 1e-10 * max(1.0, abs(x1))
        x2 = x1 + h
        x3 = x1 - h
        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
            f3 = call_f(x3)
            for _ in range(30):
                if abs(f2) <= TOL or abs(f1) <= TOL or abs(f3) <= TOL:
                    break
                # Try IQI
                A_val = (x2 - x1) / (f2 - f1) if abs(f2 - f1) > 1e-30 else None
                B_val = (x3 - x1) / (f3 - f1) if abs(f3 - f1) > 1e-30 else None
                x_new = None
                if A_val is not None and B_val is not None and abs(A_val - B_val) > 1e-30:
                    x_iq = x1 + 1.0 / (A_val - B_val)
                    if math.isfinite(x_iq) and abs(x_iq) < 1e15:
                        x_new = x_iq
                if x_new is None:
                    denom = f2 - f1
                    if abs(denom) < 1e-30:
                        break
                    x_new = x2 - f2 * (x2 - x1) / denom
                if not math.isfinite(x_new) or abs(x_new) > 1e15:
                    break
                x3, f3 = x1, f1
                x1, f1 = x2, f2
                x2 = x_new
                try:
                    f2 = call_f(x2)
                except Exception:
                    break
        except Exception:
            pass

    # One final attempt: if we're very close, try a high-precision Newton-like step
    if best_fabs > TOL and best_fabs < TOL * 10:
        x = best_x
        for _ in range(15):
            if not check_time():
                break
            if eval_count >= max_evals:
                break
            try:
                fx = call_f(x)
            except Exception:
                break
            if abs(fx) <= TOL:
                break
            # Use a very small step for derivative estimation
            h_small = 1e-12 * max(1.0, abs(x))
            try:
                fxp = call_f(x + h_small)
                fxm = call_f(x - h_small)
            except Exception:
                break
            d = (fxp - fxm) / (2.0 * h_small)
            if abs(d) < 1e-30:
                break
            step = fx / d
            if abs(step) > 1e-6 * max(1.0, abs(x)):
                step = math.copysign(1e-6 * max(1.0, abs(x)), step)
            x_new = x - step
            if not math.isfinite(x_new) or abs(x_new) > 1e15:
                break
            x = x_new

    return best_x
# EVOLVE-BLOCK-END
