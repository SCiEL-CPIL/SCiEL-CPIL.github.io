result = np.sum(u_row * v_row * w_row)
if abs(result - 1) > 1e-9: return False
