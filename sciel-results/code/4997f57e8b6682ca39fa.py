"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder for smooth polynomial equations.
    Strategy:
      1. Secant method (1 call/step after initial) for smooth polynomials.
      2. If secant stalls, use inverse quadratic interpolation with 3 points.
      3. If interpolation fails, bracket via adaptive expansion + Illinois.
    Tracks best |f(x)| seen and returns best if deadline approached.
    """
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def time_ok():
        return (time.time() - start_time) < DEADLINE

    x = float(x0)
    if not math.isfinite(x):
        x = 0.0

    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx) if math.isfinite(best_fx) else float('inf')

    if best_abs <= 1e-8:
        return best_x

    # --- Phase 1: Secant method ---
    x1 = x
    f1 = best_fx
    x2 = x1 + 1e-4 * max(1.0, abs(x1))
    f2 = f(x2)

    if not math.isfinite(f2):
        x2 = x1 - 1e-4 * max(1.0, abs(x1))
        f2 = f(x2)

    # Keep track of last 3 points for interpolation
    points = []  # list of (x, f)
    if math.isfinite(f1):
        points.append((x1, f1))
    if math.isfinite(f2):
        points.append((x2, f2))

    secant_success = False
    if math.isfinite(f1) and math.isfinite(f2) and abs(f2 - f1) > 1e-30:
        for _ in range(30):
            if not time_ok():
                break
            if best_abs <= 1e-8:
                return best_x
            if abs(f2 - f1) < 1e-30:
                break

            x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not math.isfinite(x3) or abs(x3) > 1e15:
                break

            f3 = f(x3)
            if not math.isfinite(f3):
                break

            if abs(f3) < best_abs:
                best_x = x3
                best_fx = f3
                best_abs = abs(f3)

            if best_abs <= 1e-12:
                return x3
            if best_abs <= 1e-8:
                return best_x

            # Check convergence
            if abs(x3 - x2) < 1e-14 * max(1.0, abs(x3)):
                secant_success = True
                break

            # Update points for potential interpolation
            points.append((x3, f3))
            if len(points) > 3:
                points.pop(0)

            x1, f1 = x2, f2
            x2, f2 = x3, f3

    if best_abs <= 1e-8:
        return best_x

    # --- Phase 2: Inverse Quadratic Interpolation ---
    # Use the last 3 points to fit f(x) = ax^2 + bx + c and find root
    # Actually, use inverse: x = A*f^2 + B*f + C, then set f=0
    if not secant_success and len(points) >= 3 and time_ok():
        (xa, fa), (xb, fb), (xc, fc) = points[-3], points[-2], points[-1]
        
        # Check if points are distinct enough
        if (abs(xa - xb) > 1e-15 * max(1.0, abs(xa)) and 
            abs(xb - xc) > 1e-15 * max(1.0, abs(xb)) and
            abs(xa - xc) > 1e-15 * max(1.0, abs(xa))):
            
            # Inverse quadratic: x = A*f^2 + B*f + C
            # Solve for A, B, C from three points
            # fa^2*A + fa*B + C = xa
            # fb^2*A + fb*B + C = xb  
            # fc^2*A + fc*B + C = xc
            
            # Use Cramer's rule or direct solve
            a1, a2, a3 = fa*fa, fb*fb, fc*fc
            b1, b2, b3 = fa, fb, fc
            
            # Determinant of coefficient matrix
            det = a1*(b2*1 - b3*1) - b1*(a2*1 - a3*1) + 1*(a2*b3 - a3*b2)
            # Simplified: det = a1*(b2-b3) - b1*(a2-a3) + (a2*b3-a3*b2)
            
            if abs(det) > 1e-30:
                # C for x when f=0: C = xa*b2*b3 - xb*a2*b3 + xc*a2*b1 - ... 
                # Actually let me just solve the linear system directly
                # Using numpy-free approach
                
                # C = (xa*(fb*1 - fc*1) - xb*(fa*1 - fc*1) + xc*(fa*fb - fb*fa)) / det
                # This is getting complex. Let me use a simpler approach.
                
                # Alternative: fit quadratic f(x) = p(x) to 3 points, then find root
                # p(x) = a*x^2 + b*x + c
                # p(xa) = fa, p(xb) = fb, p(xc) = fc
                
                # Solve for a, b, c
                # a*xa^2 + b*xa + c = fa
                # a*xb^2 + b*xb + c = fb
                # a*xc^2 + b*xc + c = fc
                
                # Subtract first from second: a*(xb^2-xa^2) + b*(xb-xa) = fb-fa
                # Subtract first from third: a*(xc^2-xa^2) + b*(xc-xa) = fc-fa
                
                d1 = xb - xa
                d2 = xc - xa
                if abs(d1) > 1e-15 and abs(d2) > 1e-15:
                    e1 = (fb - fa) / d1  # = a*(xb+xa) + b
                    e2 = (fc - fa) / d2  # = a*(xc+xa) + b
                    
                    if abs(d2 - d1) > 1e-15:
                        a_coeff = (e2 - e1) / (d2 - d1)
                        b_coeff = e1 - a_coeff * (xb + xa)
                        c_coeff = fa - a_coeff * xa * xa - b_coeff * xa
                        
                        # Find roots of a_coeff*x^2 + b_coeff*x + c_coeff = 0
                        if abs(a_coeff) > 1e-30:
                            disc = b_coeff*b_coeff - 4*a_coeff*c_coeff
                            if disc >= 0:
                                sqrt_disc = math.sqrt(disc)
                                r1 = (-b_coeff + sqrt_disc) / (2*a_coeff)
                                r2 = (-b_coeff - sqrt_disc) / (2*a_coeff)
                                
                                # Pick the root closer to current best or with smaller |f|
                                candidates = []
                                for r in [r1, r2]:
                                    if math.isfinite(r) and abs(r) < 1e15:
                                        fr = f(r)
                                        if math.isfinite(fr):
                                            candidates.append((r, abs(fr)))
                                        if abs(fr) < best_abs:
                                            best_x = r
                                            best_fx = fr
                                            best_abs = abs(fr)
                                
                                if candidates:
                                    # Evaluate both, pick best
                                    for r, _ in candidates:
                                        if abs(f(r)) <= 1e-8:
                                            return r
                                    
                                    # Try secant from the two roots
                                    if len(candidates) >= 2:
                                        (r1, _), (r2, _) = candidates[0], candidates[1]
                                        fr1 = f(r1)
                                        fr2 = f(r2)
                                        if abs(fr2 - fr1) > 1e-30:
                                            for _ in range(10):
                                                if not time_ok():
                                                    break
                                                r3 = r2 - fr2 * (r2 - r1) / (fr2 - fr1)
                                                if not math.isfinite(r3) or abs(r3) > 1e15:
                                                    break
                                                fr3 = f(r3)
                                                if not math.isfinite(fr3):
                                                    break
                                                if abs(fr3) < best_abs:
                                                    best_x = r3
                                                    best_fx = fr3
                                                    best_abs = abs(fr3)
                                                if best_abs <= 1e-8:
                                                    return best_x
                                                if abs(r3 - r2) < 1e-14 * max(1.0, abs(r3)):
                                                    break
                                                r1, fr1 = r2, fr2
                                                r2, fr2 = r3, fr3
                    elif abs(b_coeff) > 1e-30:
                        # Linear case
                        r = -c_coeff / b_coeff
                        if math.isfinite(r) and abs(r) < 1e15:
                            fr = f(r)
                            if math.isfinite(fr):
                                if abs(fr) < best_abs:
                                    best_x = r
                                    best_fx = fr
                                    best_abs = abs(fr)
                                if best_abs <= 1e-8:
                                    return best_x

    if best_abs <= 1e-8:
        return best_x

    # --- Phase 3: Bracket the root ---
    found_bracket = False
    for start_pt in [best_x, x0]:
        if not time_ok():
            break
        fs = f(start_pt) if start_pt != best_x else best_fx
        if not math.isfinite(fs):
            continue

        for direction in [1, -1]:
            if not time_ok():
                break
            step = 1.0
            for _ in range(30):
                if not time_ok():
                    break
                b = start_pt + direction * step
                fb = f(b)
                if not math.isfinite(fb):
                    step *= 2
                    if abs(b) > 1e15:
                        break
                    continue

                if fs * fb < 0:
                    if abs(fs) < abs(fb):
                        a, b = start_pt, b
                        fa, fb = fs, fb
                    else:
                        a, b = b, start_pt
                        fa, fb = fb, fs
                    found_bracket = True
                    break

                step *= 2
                if abs(b) > 1e15:
                    break
            if found_bracket:
                break
        if found_bracket:
            break

    if found_bracket and time_ok():
        # --- Phase 4: Illinois method ---
        lo, hi = a, b
        flo, fhi = fa, fb

        for _ in range(80):
            if not time_ok():
                break

            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if not math.isfinite(fmid):
                break

            if abs(fmid) < best_abs:
                best_x = mid
                best_fx = fmid
                best_abs = abs(fmid)

            if best_abs <= 1e-12:
                return mid
            if best_abs <= 1e-8:
                return best_x

            if flo * fmid < 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid

            if abs(flo) > abs(fhi):
                lo = 0.5 * (lo + hi)
                flo = f(lo)
                if not math.isfinite(flo):
                    break
            else:
                hi = 0.5 * (lo + hi)
                fhi = f(hi)
                if not math.isfinite(fhi):
                    break

            if abs(hi - lo) < 1e-14 * max(1.0, abs(mid)):
                break

        if best_abs <= 1e-8:
            return best_x

    return best_x
# EVOLVE-BLOCK-END
