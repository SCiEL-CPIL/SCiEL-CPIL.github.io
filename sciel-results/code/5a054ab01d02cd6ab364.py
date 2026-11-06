# p_vals: (M,)
# coeffs: (N,)
# powers: (M, N)

# Create (N, M) array of trial polynomials
# p_trials[k, j] = p_vals[j] - 2 * coeffs[k] * powers[j, k]
# p_vals is (M,), broadcast to (N, M)
# coeffs[:, None] is (N, 1)
# powers.T is (N, M)
# term = 2 * coeffs[:, None] * powers.T
# p_trials = p_vals[None, :] - term
# scores = np.max(np.abs(p_trials), axis=1)
# best_k = np.argmin(scores)
