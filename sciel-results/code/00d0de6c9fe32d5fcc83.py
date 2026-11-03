# Vectorized best flip selection
k = worst_lag
# v[i] = x[i+k] + x[i-k]
# Handle bounds
v = np.zeros(N)
# i+k < N
mask1 = np.arange(N) + k < N
v[mask1] += x[np.arange(N)[mask1] + k]
# i-k >= 0
mask2 = np.arange(N) - k >= 0
v[mask2] += x[np.arange(N)[mask2] - k]

# delta[i] = -2 * x[i] * v[i]
deltas = -2 * x * v
# new_vals = c[k] + deltas
# We want min abs(new_vals)
# But we also need to consider if this flip affects other peaks?
# Greedy approach: just reduce the current peak.
new_vals = target_val + deltas
min_idx = np.argmin(np.abs(new_vals))

if np.abs(new_vals[min_idx]) < np.abs(target_val):
    # Flip min_idx
    ...
