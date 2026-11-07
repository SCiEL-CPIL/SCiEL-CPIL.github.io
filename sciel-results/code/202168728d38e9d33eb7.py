import math
import time

# EVOLVE-BLOCK-START
def _is_finite(x):
    return math.isfinite(x)

def find_root(f, x0):
    deadline = time.monotonic() + 110.0  # safety margin before 120s kill

    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0

    # --- Phase 1: evaluate at x0 ---
    fx0 = f(x0)
    if not _is_finite(fx0):
        fx0 = 0.0
    if abs(fx0) <= 1e-8:
        return x0

    # --- Phase 2: build a bracket via geometric expansion ---
    # Try to find x_lo, x_hi with f of opposite signs.
    # Start from x0, expand outward.
    x_lo = x0
    x_hi = x0
    f_lo = fx0
    f_hi = fx0

    # We already have f(x0). Try to find a sign change by expanding.
    # Strategy: try x0 + 1, x0 - 1, x0 + 2, x0 - 2, x0 + 4, ...
    # Also try small perturbations first in case root is very close.
    scale = 1.0
    found_bracket = False
    best_x = x0
    best_abs = abs(fx0)

    # First, try a few small steps to see if we can quickly bracket
    for trial in range(1, 30):
        if time.monotonic() > deadline:
            break
        s = scale * (1.0 if trial % 2 == 1 else -1.0)
        if trial > 1:
            # grow scale: 1, 1, 2, 2, 4, 4, 8, 8, ...
            if trial % 2 == 0:
                scale *= 2.0
                s = scale * (1.0 if trial % 4 == 0 else -1.0)
                # Actually let me redo: pairs (1, -1, 2, -2, 4, -4, ...)
                pass
        # Simpler: explicit sequence
        # Let me just use a clean loop

    # Redo bracketing cleanly
    scale = 1.0
    bracket_found = False
    for i in range(60):
        if time.monotonic() > deadline:
            break
        d = scale
        if i % 2 == 0:
            x_try = x0 + d
        else:
            x_try = x0 - d

        if not _is_finite(x_try):
            scale *= 2.0
            continue

        f_try = f(x_try)
        if not _is_finite(f_try):
            scale *= 2.0
            continue

        if abs(f_try) < best_abs:
            best_abs = abs(f_try)
            best_x = x_try

        if abs(f_try) <= 1e-8:
            return x_try

        # Check for sign change with x0
        if f_try * fx0 < 0:
            # Bracket: [min(x0, x_try), max(x0, x_try)]
            if x_try > x0:
                x_lo, x_hi = x0, x_try
                f_lo, f_hi = fx0, f_try
            else:
                x_lo, x_hi = x_try, x0
                f_lo, f_hi = f_try, fx0
            bracket_found = True
            break

        # Also check sign change with the other side if we have a previous point
        # (we only track x0, so just expand)
        if i % 2 == 1:
            scale *= 2.0

    # If no bracket found, fall back to secant/Newton without bracket
    # Use the best point found as starting point
    if not bracket_found:
        # Try secant method from x0 and x0+1 (or best_x if different)
        x_a = x0
        f_a = fx0
        x_b = best_x if best_x != x0 else x0 + 1.0
        if x_b == x0:
            x_b = x0 + 1.0
        f_b = f(x_b)
        if not _is_finite(f_b):
            f_b = 0.0

        for _ in range(80):
            if time.monotonic() > deadline:
                break
            if abs(f_b) < abs(f_a):
                x_a, f_a = x_b, f_b
            if abs(f_a) <= 1e-8:
                return x_a
            if not _is_finite(x_a) or not _is_finite(f_a):
                break
            # Secant step
            if f_b == f_a:
                # Degenerate; take a small step
                x_new = x_a + 0.1 * max(1.0, abs(x_a))
            else:
                x_new = x_a - f_a * (x_a - x_b) / (f_a - f_b)
            if not _is_finite(x_new):
                x_new = x_a + 0.1 * max(1.0, abs(x_a))
            f_new = f(x_new)
            if not _is_finite(f_new):
                f_new = 0.0
            x_b = x_new
            f_b = f_new
        return x_a if abs(f_a) <= abs(f_b) else x_b

    # --- Phase 3: safeguarded secant with bisection fallback ---
    # We have [x_lo, x_hi] with f_lo * f_hi < 0
    x_prev = x_lo
    f_prev = f_lo
    x_curr = x_hi
    f_curr = f_hi

    for _ in range(200):
        if time.monotonic() > deadline:
            break
        if abs(f_curr) <= 1e-8:
            return x_curr
        if abs(f_prev) <= 1e-8:
            return x_prev

        # Secant prediction
        if f_curr != f_prev:
            x_sec = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        else:
            x_sec = 0.5 * (x_lo + x_hi)

        if not _is_finite(x_sec):
            x_sec = 0.5 * (x_lo + x_hi)

        # Safeguard: if secant step is outside bracket, use bisection
        if x_sec <= x_lo or x_sec >= x_hi:
            x_new = 0.5 * (x_lo + x_hi)
        else:
            x_new = x_sec

        f_new = f(x_new)
        if not _is_finite(f_new):
            # Fallback to bisection
            x_new = 0.5 * (x_lo + x_hi)
            f_new = f(x_new)
            if not _is_finite(f_new):
                break

        # Update bracket
        if f_new * f_lo <= 0:
            x_hi = x_new
            f_hi = f_new
        else:
            x_lo = x_new
            f_lo = f_new

        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        f_curr = f_new

        # Convergence check on interval
        if abs(x_hi - x_lo) < 1e-14 * max(1.0, abs(x_lo), abs(x_hi)):
            break

    # --- Phase 4: high-precision polish (Newton) ---
    # Use the best point; do a few Newton steps with central difference
    x = x_curr if abs(f_curr) <= abs(f_prev) else x_prev
    fx = f_curr if abs(f_curr) <= abs(f_prev) else f_prev

    for _ in range(10):
        if time.monotonic() > deadline:
            break
        if abs(fx) <= 1e-12:
            return x
        h = 1e-8 * max(1.0, abs(x))
        f_plus = f(x + h)
        f_minus = f(x - h)
        if not _is_finite(f_plus) or not _is_finite(f_minus):
            break
        d = (f_plus - f_minus) / (2.0 * h)
        if d == 0 or not _is_finite(d):
            break
        x_new = x - fx / d
        if not _is_finite(x_new):
            break
        fx_new = f(x_new)
        if not _is_finite(fx_new):
            break
        x = x_new
        fx = fx_new

    # Return best candidate
    if abs(fx) <= 1e-8:
        return x
    # Check a few candidates
    candidates = [x, x_lo, x_hi, x0]
    best = x
    best_val = abs(fx)
    for c in candidates:
        if not _is_finite(c):
            continue
        # We may not have current f for these; use stored values where possible
        pass
    return x
# EVOLVE-BLOCK-END
