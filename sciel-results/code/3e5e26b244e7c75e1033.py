    D_approx = np.zeros(len(a_values))
    D_approx[-1] = 1.0
    t_vals = np.log(a_values)
    for i in range(len(a_values) - 2, -1, -1):
        avg_f = 0.5 * (f_approx[i] + f_approx[i+1])
        integral = avg_f * (t_vals[i+1] - t_vals[i])
        D_approx[i] = D_approx[i+1] * np.exp(-integral)
