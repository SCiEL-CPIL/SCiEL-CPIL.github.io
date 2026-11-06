    fails = cov[1:] < 2
    if not fails.any():
        L = L_max
    else:
        L = np.argmax(fails)
    