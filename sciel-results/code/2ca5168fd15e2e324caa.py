full = U_3d[:, :, None, :] * V_3d[None, :, :, :] * W_3d[None, None, :, :]
result = np.sum(full, axis=3)
return (result - 1.0).flatten()
