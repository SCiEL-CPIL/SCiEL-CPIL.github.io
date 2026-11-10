for k in range(2, K + 1):
    v = float(mu[k])
    if v != 0.0:
        g[k::k] += v
