def _smooth_max_and_grad(h, beta=10.0):
    O = _overlaps(h)
    # Include k=0
    max_O = np.max(O)
    val = max_O + np.log(np.sum(np.exp(O - max_O))) / beta
    
    exp_O = np.exp(O - max_O)
    w = exp_O / np.sum(exp_O)
    
    grad = np.zeros(N)
    for k in range(N):
        wk = w[k]
        if wk > 1e-9:
            if k == 0:
                # d O_0 / d h_i = 1 - 2 h_i
                grad += wk * (1.0 - 2.0 * h)
            else:
                grad += wk * ((1.0 - np.roll(h, -k)) - np.roll(h, k))
            
    grad *= beta
    return val, grad
