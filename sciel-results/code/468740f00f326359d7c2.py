def get_hermite_polys(n_max):
    polys = []
    H0 = np.poly1d([1])
    H1 = np.poly1d([2, 0])
    polys.append(H0)
    if n_max >= 1:
        polys.append(H1)
    for n in range(1, n_max):
        Hn = 2 * np.poly1d([1, 0]) * polys[-1] - 2 * n * polys[-2]
        polys.append(Hn)
    return polys
