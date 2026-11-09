"""Custom variable-step Radau IIA(5) integrator with analytic Jacobian for Van der Pol."""
import numpy as np

_c = np.array([0.2567697113820848, 0.3983112382018402,
               0.5868706956534012, 0.8375752590595543, 1.0])
_A = np.array([
    [0.0082125121405976, 0.0090768027405854, 0.0116859234476484, 0.0192157327717625, 0.0117916395370991],
    [0.0037211719588516, 0.0120355593981311, 0.0184139727516236, 0.0329260703292546, 0.0205063132490060],
    [0.0005139241835037, 0.0029391899976074, 0.0114518146002667, 0.0265875261087223, 0.0229585759104684],
    [0.0000124800044879, 0.0000972444855677, 0.0008960114921567, 0.0065895233439289, 0.0095574133265804],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
])
_b = _A[4, :]
_s = 5


def _vd_pol_jac(t, y):
    y0, y1 = y[0], y[1]
    mu = 5.0
    return np.array([
        [0.0, 1.0],
        [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0 * y0) * y1]
    ])


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    h = 0.01
    t = t0
    
    while t < t1 - 1e-15:
        if t + h > t1:
            h = t1 - t
        if h < 1e-14:
            break
        
        Y = np.tile(y, (_s, 1))
        for _ in range(80):
            F = np.array([f(t + _c[i] * h, Y[i]) for i in range(_s)])
            R = Y - y[None, :] - h * (_A @ F)
            rmax = np.max(np.max(np.abs(R), axis=1))
            if rmax < 1e-15:
                break
            
            J_full = np.zeros((_s * n, _s * n))
            for i in range(_s):
                Ji = _vd_pol_jac(t + _c[i] * h, Y[i])
                for j in range(_s):
                    block = -h * _A[i, j] * Ji
                    if i == j:
                        block += np.eye(n)
                    J_full[i*n:(i+1)*n, j*n:(j+1)*n] = block
            
            try:
                delta = np.linalg.solve(J_full, -R.flatten())
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(J_full, -R.flatten(), rcond=None)[0]
            
            for i in range(_s):
                Y[i] += delta[i*n:(i+1)*n]
            if np.linalg.norm(delta) < 1e-15 * (1.0 + np.linalg.norm(Y.flatten())):
                break
        
        F = np.array([f(t + _c[i] * h, Y[i]) for i in range(_s)])
        y_new = y + h * np.dot(_b, F)
        
        # Error estimate
        err = np.linalg.norm(y_new - Y[-1])
        scale = 1e-13 + 1e-11 * np.maximum(1.0, np.abs(y_new))
        err_norm = err / np.max(scale)
        
        if err_norm > 1.0:
            h *= 0.8 * (1.0 / err_norm) ** 0.25
            if h < 1e-14:
                h = 1e-14
            continue
        
        t += h
        y = y_new
        if err_norm < 0.1:
            h *= 1.25
        else:
            h *= 1.05
        h = min(h, 0.05)
    
    return y
