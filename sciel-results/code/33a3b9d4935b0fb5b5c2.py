# In _pack_objective
# Add: pen += 0.01 * np.sum((rs - np.mean(rs))**2)

# In _pack_objective_grad
# d/d r_i of 0.01 * sum((r_j - mean)^2) = 0.01 * 2 * (r_i - mean)
