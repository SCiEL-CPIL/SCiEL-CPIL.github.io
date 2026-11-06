"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-5 polynomial to the unresolved fast-scale forcing.
    Uses orthogonal polynomial basis (Legendre scaled to data range) to avoid
    multicollarity, then converts back to monomial coefficients.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    n = len(x)
    if n < 6:
        # Not enough data for degree 5; fall back to linear
        if n < 2:
            return np.zeros(6)
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(x, y, 1)[::-1]
        return coeff

    # Scale x to [-1, 1] for stable Legendre polynomial evaluation
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        # All x values are the same; return constant
        return np.array([np.mean(y)] + [0.0]*5)

    # Map to [-1, 1]
    x_scaled = (x - 0.5 * (x_max + x_min)) / (0.5 * x_range)

    # Build Legendre polynomial basis up to degree 5
    # P_0(t) = 1
    # P_1(t) = t
    # P_2(t) = (3t^2 - 1)/2
    # P_3(t) = (5t^3 - 3t)/2
    # P_4(t) = (35t^4 - 30t^2 + 3)/8
    # P_5(t) = (63t^5 - 70t^3 + 15t)/8
    t = x_scaled
    P = np.zeros((n, 6))
    P[:, 0] = 1.0
    P[:, 1] = t
    P[:, 2] = 0.5 * (3.0 * t**2 - 1.0)
    P[:, 3] = 0.5 * (5.0 * t**3 - 3.0 * t)
    P[:, 4] = 0.125 * (35.0 * t**4 - 30.0 * t**2 + 3.0)
    P[:, 5] = 0.125 * (63.0 * t**5 - 70.0 * t**3 + 15.0 * t)

    # Solve least squares in Legendre basis
    # Use np.linalg.lstsq for numerical stability
    a_leg, residuals, rank, sv = np.linalg.lstsq(P, y, rcond=None)

    # Convert from Legendre basis on [-1,1] to monomial basis on original x
    # We need: sum_{k=0}^{5} a_leg[k] * P_k((x - mid)/half) = sum_{i=0}^{5} c_i * x^i
    #
    # P_k(t) where t = (x - mid)/half = (x - mid) * (1/half)
    # Let u = x - mid, s = 1/half
    # t = u * s
    #
    # We'll compute the monomial coefficients by expanding each P_k(u*s) in powers of x.
    # P_k(u*s) = sum_{j=0}^{k} b_{k,j} * (u*s)^j = sum_{j=0}^{k} b_{k,j} * s^j * u^j
    # u^j = (x - mid)^j = sum_{i=0}^{j} binom(j,i) * x^i * (-mid)^(j-i)
    #
    # So coefficient of x^i in P_k(u*s) is:
    #   sum_{j=i}^{k} b_{k,j} * s^j * binom(j,i) * (-mid)^(j-i)
    #
    # where b_{k,j} are coefficients of P_k in powers of t.

    mid = 0.5 * (x_max + x_min)
    half = 0.5 * x_range
    s = 1.0 / half

    # Coefficients of Legendre polynomials in powers of t:
    # P_0(t) = 1                  -> [1]
    # P_1(t) = t                  -> [0, 1]
    # P_2(t) = (3t^2-1)/2        -> [-1/2, 0, 3/2]
    # P_3(t) = (5t^3-3t)/2       -> [0, -3/2, 0, 5/2]
    # P_4(t) = (35t^4-30t^2+3)/8 -> [3/8, 0, -30/8, 0, 35/8]
    # P_5(t) = (63t^5-70t^3+15t)/8 -> [0, 15/8, 0, -70/8, 0, 63/8]

    b = np.zeros((6, 6))
    b[0, 0] = 1.0
    b[1, 1] = 1.0
    b[2, 0] = -0.5
    b[2, 2] = 1.5
    b[3, 1] = -1.5
    b[3, 3] = 2.5
    b[4, 0] = 3.0/8.0
    b[4, 2] = -30.0/8.0
    b[4, 4] = 35.0/8.0
    b[5, 1] = 15.0/8.0
    b[5, 3] = -70.0/8.0
    b[5, 5] = 63.0/8.0

    # Compute monomial coefficients c_i for i=0..5
    c = np.zeros(6)
    for i in range(6):
        total = 0.0
        for k in range(6):
            for j in range(i, 6):
                if b[k, j] != 0.0:
                    # binom(j, i) * (-mid)^(j-i) * s^j * b[k,j]
                    from math import comb
                    total += b[k, j] * (s ** j) * comb(j, i) * ((-mid) ** (j - i))
        c[i] = a_leg[k] * total if False else 0.0  # placeholder, fix below

    # Redo properly
    c = np.zeros(6)
    for i in range(6):
        total = 0.0
        for k in range(6):
            if a_leg[k] == 0.0:
                continue
            for j in range(i, 6):
                if b[k, j] != 0.0:
                    from math import comb
                    total += a_leg[k] * b[k, j] * (s ** j) * comb(j, i) * ((-mid) ** (j - i))
        c[i] = total

    # Verify: check that the polynomial reproduces y reasonably
    # (optional sanity check, skip if time is tight)
    elapsed = time.time() - start_time
    if elapsed < 140.0:
        # Verify and potentially refine
        x_test = x[:min(100, n)]
        y_pred = np.zeros(len(x_test))
        for i in range(6):
            y_pred += c[i] * (x_test ** i)
        mse_check = np.mean((y_pred - y[:min(100, n)])**2)
        # If something went wrong, fall back to direct polyfit
        if not np.isfinite(mse_check) or mse_check > 1e10:
            try:
                coeff_direct = np.polynomial.polynomial.polyfit(x, y, 5)
                if np.all(np.isfinite(coeff_direct)):
                    c = coeff_direct
            except Exception:
                pass

    # Ensure all coefficients are finite
    for i in range(6):
        if not np.isfinite(c[i]):
            c[i] = 0.0

    return c
# EVOLVE-BLOCK-END
