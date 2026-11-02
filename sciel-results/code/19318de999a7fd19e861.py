# Vectorized M construction for U update
# M[k, c] = V[j*p+k, c] * W[k*n+i, c]
# V_slice = V[j*p : j*p+p, :] # shape (p, rank)
# W_slice = W[:, :] # We need W[k*n+i, :] for k in 0..p-1
# This is not a simple slice because of the k*n+i index.
# W_indices = [k*n + i for k in range(p)]
# W_slice = W[W_indices, :]
# M = V_slice * W_slice
