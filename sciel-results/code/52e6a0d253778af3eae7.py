import numpy as np
import time
from scipy.optimize import minimize

def _ry(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])

def _apply_ry(state, n, qubit, theta):
    # state is vector of size 2**n
    # reshape to tensor
    # apply gate on qubit
    pass

def _apply_cz(state, n, q1, q2):
    pass

def _energy_and_grad(theta_flat, n, B, J):
    # reshape theta_flat to (layers, n)
    # compute state
    # compute energy
    # compute gradient
    pass

def angles(n, coupling, field, layers):
    # main logic
    pass
