"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _dop853_step(f, t, y, h):
    """One step of DOP853 (Dormand-Prince 8th order with 5th order error estimate).
    Returns (y_new, err_estimate, nfev)."""
    # Butcher tableau for DOP853 (Dormand-Prince 8(5,3))
    # c, A, b (8th order), b_star (5th order for error)
    c = np.array([
        0, 1/252, 1/216, 1/4, 1/6, 5/12, 1, 5/2, 3/2, 12/13,
        1, 1/3, 1/6
    ])
    # A matrix (upper triangular, 13x13)
    A = np.zeros((13, 13))
    A[1,0] = 1/252
    A[2,0] = 1/216
    A[2,1] = -1/36
    A[3,0] = 1/4
    A[4,0] = 1/6
    A[4,2] = -1/6
    A[5,0] = 5/12
    A[5,2] = -1/2
    A[5,3] = -1/2
    A[5,4] = 5/3
    A[6,0] = 1
    A[6,2] = -5/2
    A[6,3] = 5/2
    A[6,4] = -5
    A[6,5] = 5
    A[7,0] = 5/2
    A[7,2] = -5
    A[7,3] = 5
    A[7,4] = -25/3
    A[7,5] = 25/3
    A[7,6] = -5/2
    A[8,0] = 3/2
    A[8,2] = -5/2
    A[8,3] = 5/2
    A[8,4] = -25/6
    A[8,5] = 25/6
    A[8,6] = -3/2
    A[8,7] = 3/2
    A[9,0] = 12/13
    A[9,2] = -18/13
    A[9,3] = 18/13
    A[9,4] = -90/13
    A[9,5] = 90/13
    A[9,6] = -12/13
    A[9,7] = 12/13
    A[9,8] = -12/13
    A[10,0] = 1
    A[10,2] = -3
    A[10,3] = 3
    A[10,4] = -15
    A[10,5] = 15
    A[10,6] = -1
    A[10,7] = 1
    A[10,8] = -1
    A[10,9] = 1
    A[11,0] = 1/3
    A[11,2] = -1/3
    A[11,3] = 1/3
    A[11,4] = -5/2
    A[11,5] = 5/2
    A[11,6] = 1/3
    A[11,7] = -1/3
    A[11,8] = 1/3
    A[11,9] = -1/3
    A[11,10] = 1/3
    A[12,0] = 1/6
    A[12,2] = -1/6
    A[12,3] = 1/6
    A[12,4] = -5/3
    A[12,5] = 5/3
    A[12,6] = 1/6
    A[12,7] = -1/6
    A[12,8] = 1/6
    A[12,9] = -1/6
    A[12,10] = 1/6
    A[12,11] = -1/6

    # b vector (8th order weights)
    b = np.array([
        32561130195063/337520440000000, 0, 1737458737741/9580750880000,
        0, 763165312549/18613294540800, 0, -23903794295767/260550369216000,
        1117336088474487/40192235749017600, 60027683423181/10750381056000,
        509899355144859/1640046124800000, 0, 18800620547982/26315323136000,
        55398701367061/26315323136000
    ])

    # b_star vector (5th order weights for error estimate)
    b_star = np.array([
        22410793503951262343/21082765190530390069760, 0, 51860554254246658241879/232910417095834290767360,
        0, 9955156074351446577683/12834437734773966161920, 0, -36504618134077828118113/40106423204321401484800,
        4748233858324267797099601/320851385634571211878400, 2830796276106638247523/128340554253828484751360,
        29562257704058095350251/40106423204321401484800, 0,
        3502193983315112932753/13368807734773966161920, 9417490852874857/38198350039680
    ])

    nfev = 0
    k = np.zeros((13, len(y)))
    k[0] = f(t, y)
    nfev += 1

    for i in range(1, 13):
        s = y.copy()
        for j in range(i):
            if A[i, j] != 0:
                s = s + h * A[i, j] * k[j]
        k[i] = f(t + h * c[i], s)
        nfev += 1

    y_new = y.copy()
    for i in range(13):
        if b[i] != 0:
            y_new = y_new + h * b[i] * k[i]

    y_err = y.copy()
    for i in range(13):
        if b_star[i] != 0:
            y_err = y_err + h * b_star[i] * k[i]

    err = np.max(np.abs(y_new - y_err))
    return y_new, err, nfev


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 1000  # Initial step size
    t = t0
    deadline = time.time() + 250  # Safety margin before 300s limit

    # Adaptive DOP853 with Richardson extrapolation idea:
    # Use two different max_step values and combine
    # First pass: standard adaptive DOP853
    nfev_total = 0

    # We'll do a single adaptive run with tight tolerance
    rtol = 1e-12
    atol = 1e-14

    while t < t1 - 1e-15:
        if time.time() > deadline - 10:
            break

        # Limit step to not exceed t1
        h_step = min(h, t1 - t)

        y_new, err, nfev = _dop853_step(f, t, y, h_step)
        nfev_total += nfev

        if not np.all(np.isfinite(y_new)):
            h = h / 2
            continue

        # Error control
        err_norm = err / (atol + rtol * np.max(np.abs(y_new)))

        if err_norm <= 1.0:
            # Accept step
            t = t + h_step
            y = y_new
            # Adaptive step size selection
            if err_norm > 1e-6:
                h = h_step * min(5.0, max(0.2, 0.9 * err_norm ** (-0.2)))
            else:
                h = h_step * 5.0
        else:
            # Reject step, reduce h
            h = h_step * max(0.2, 0.9 * err_norm ** (-0.2))

    return y
# EVOLVE-BLOCK-END
