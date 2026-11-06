from scipy.linalg import expm_multiply

v_aug = np.zeros(n + 1)
v_aug[:n] = y0_arr
v_aug[n] = 1.0

y_aug_end = expm_multiply(M * dt, v_aug)
y_end = y_aug_end[:n]
