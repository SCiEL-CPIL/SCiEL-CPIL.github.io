"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Construct the identity tensor of size n*m*p
    # Shape: (n, m, p, n, m, p)
    # E[i, j, k, i', j', k'] = 1 if i==i', j==j', k==k' else 0
    E = np.eye(rank).reshape(n, m, p, n, m, p)
    
    # U: shape (n*m, n*m*p)
    # U[i*m + j, i'*m*p + j'*p + k'] = 1 if i==i' and j==j'
    # This is obtained by summing over the row 'k' index (axis 2) of E
    U_tensor = np.sum(E, axis=2)  # Shape: (n, m, n, m, p)
    U = U_tensor.reshape(n * m, rank)
    
    # V: shape (m*p, n*m*p)
    # V[j*p + k, i'*m*p + j'*p + k'] = 1 if j==j' and k==k'
    # This is obtained by summing over the row 'i' index (axis 0) of E
    V_tensor = np.sum(E, axis=0)  # Shape: (m, p, n, m, p)
    V = V_tensor.reshape(m * p, rank)
    
    # W: shape (p*n, n*m*p)
    # W[k*n + i, i'*m*p + j'*p + k'] = 1 if k==k' and i==i'
    # We need to sum over the row 'j' index (axis 1) of E
    W_tensor = np.sum(E, axis=1)  # Shape: (n, p, n, m, p)
    # The row indices in W_tensor are (i, p). We want them to be (k, i) i.e. (p, n)
    # Transpose axes 0 and 1
    W_tensor = W_tensor.transpose(1, 0, 2, 3, 4)  # Shape: (p, n, n, m, p)
    W = W_tensor.reshape(p * n, rank)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
