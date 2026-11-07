def acquisition(mu, sigma, f_best):
    # Expected Improvement with a small xi to balance exploration/exploitation
    xi = 1e-3
    z = (f_best - mu - xi) / (sigma + 1e-9)
    return (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
