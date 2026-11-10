<<<<<<< SEARCH
import numpy as np
import time
=======
import numpy as np
import time
from math import comb
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Try degree 5 with regularization
    try:
        coeff = _fit_regularized_poly(x, y, degree=5, lam=1e-8)
    except Exception:
        # Fallback to degree 1
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            coeff = np.zeros(6)
=======
    # Try degree 1 with regularization (linear fit)
    # Inspiration suggests linear is best baseline
    try:
        coeff = _fit_regularized_poly(x, y, degree=1, lam=1e-8)
    except Exception:
        # Fallback to simple linear fit
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            coeff = np.zeros(6)
>>>>>>> REPLACE
