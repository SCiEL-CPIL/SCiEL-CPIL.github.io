"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

_KNOTS = np.array([-3., -3., -3., -3.,
                   -3. + 6./7., -3. + 12./7., -3. + 18./7.,
                   -3. + 24./7., -3. + 30./7., -3. + 36./7.,
                   3., 3., 3., 3.])
_DEGREE = 3
_N_CTRL = 10

def _eval_bspline_basis(x, knots, degree):
    x = np.atleast_1d(x).astype(float)
    n_ctrl = len(knots) - degree - 1
    N = np.zeros((len(x), n_ctrl))
    for i in range(n_ctrl):
        if i == n_ctrl - 1:
            N[:, i] = ((knots[i] <= x) & (x <= knots[i+1])).astype(float)
        else:
            N[:, i] = ((knots[i] <= x) & (x < knots[i+1])).astype(float)
    for d in range(1, degree + 1):
        N_new = np.zeros_like(N)
        for i in range(n_ctrl - d, -1, -1):
            denom1 = knots[i+d] - knots[i]
            denom2 = knots[i+d+1] - knots[i+1]
            term1 = np.zeros_like(x)
            term2 = np.zeros_like(x)
            if denom1 != 0:
                term1 = (x - knots[i]) / denom1 * N[:, i]
            if denom2 != 0:
                term2 = (knots[i+d+1] - x) / denom2 * N[:, i+1]
            N_new[:, i] = term1 + term2
        N = N_new
    return N

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(x)
    basis = _eval_bspline_basis(x, _KNOTS, _DEGREE)
    return basis @ params[:10]
# EVOLVE-BLOCK-END
