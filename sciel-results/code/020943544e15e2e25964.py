def score(h):
    h = np.array(h)
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    sup = np.max(conv)
    return l2_sq / (l1 * sup)
