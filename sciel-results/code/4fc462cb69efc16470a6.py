def polynomial_mutation(x, lo, hi, eta=20.0):
    y = x.copy()
    for i in range(len(x)):
        if np.random.rand() < 1.0 / len(x):
            if hi[i] - lo[i] < 1e-10:
                continue
            delta1 = (y[i] - lo[i]) / (hi[i] - lo[i])
            delta2 = (hi[i] - y[i]) / (hi[i] - lo[i])
            rand = np.random.rand()
            if rand <= 0.5:
                xy = (2.0 * rand + (1.0 - 2.0 * rand) * ((1.0 + (eta + 1.0) * delta1) ** (-(eta + 1.0)))) ** (1.0 / (eta + 1.0))
                deltaq = xy - 1.0
            else:
                xy = (2.0 * (1.0 - rand) + (2.0 * rand - 1.0) * ((1.0 + (eta + 1.0) * delta2) ** (-(eta + 1.0)))) ** (1.0 / (eta + 1.0))
                deltaq = 1.0 - xy
            y[i] = y[i] + deltaq * (hi[i] - lo[i])
            y[i] = np.clip(y[i], lo[i], hi[i])
    return y
