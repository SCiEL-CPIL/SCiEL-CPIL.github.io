<<<<<<< SEARCH
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _normalize(v):
    norms = np.linalg.norm(v, axis=1)
    # Avoid division by zero
    norms[norms < 1e-10] = 1.0
    return v / norms[:, np.newaxis]
=======
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _normalize(v):
    norms = np.linalg.norm(v, axis=1)
    # Avoid division by zero
    norms[norms < 1e-10] = 1.0
    return v / norms[:, np.newaxis]

def _normalize_batch(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return v / norms
>>>>>>> REPLACE

