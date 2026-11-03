val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
if abs(val - 1.0) > 1e-9:
    return False
