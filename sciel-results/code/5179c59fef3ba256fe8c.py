try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except ImportError:
    HAS_MILP = False
