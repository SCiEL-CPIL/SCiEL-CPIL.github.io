def objective_smooth(z, beta=200.0):
    h = softmax(z)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    Q = np.sum(y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2)
    M = np.logsumexp(beta*g)/beta
    return -Q/(3.0*M)  # minimize negative
