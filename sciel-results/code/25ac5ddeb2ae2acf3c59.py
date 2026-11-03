    M_csr = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_k, n_t))
    M_csc = M_csr.tocsc()
