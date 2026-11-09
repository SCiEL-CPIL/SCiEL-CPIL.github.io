nodes, weights = np.polynomial.legendre.leggauss(8)
# nodes are in [-1, 1], map to [0, 1]
nodes = 0.5 * (nodes + 1)
weights = 0.5 * weights
