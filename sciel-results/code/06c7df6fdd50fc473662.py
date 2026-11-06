def evaluate_np(A):
    A = np.array(A)
    sums = np.add.outer(A, A)
    prods = np.multiply.outer(A, A)
    s_size = len(np.unique(sums))
    p_size = len(np.unique(prods))
    return max(s_size, p_size)
