U_reshaped = U.reshape(6, 6, -1)  # U[i, j, r] where the first two indices are...
V_reshaped = V.reshape(6, 6, -1)  # V[j, k, r]
W_reshaped = W.reshape(6, 6, -1)  # W[k, i, r]
reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
