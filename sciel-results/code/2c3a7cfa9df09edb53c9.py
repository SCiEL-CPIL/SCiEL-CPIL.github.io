import numpy as np
U_mat = np.array([
    [1,0,0,1],
    [0,0,1,1],
    [1,0,0,0],
    [0,0,0,1],
    [1,1,0,0],
    [-1,0,1,0],
    [0,1,0,-1]
]).T  # 4x7
# Find null space
from scipy.linalg import null_space
ns = null_space(U_mat)
print("Null space dimension:", ns.shape[1])
print(ns)
