import numpy as np
from scipy.special import roots_genlaguerre

# Zeros of L_12^{(1)}(x)
zeros = roots_genlaguerre(12, 1)[0]
