S = A_sym[i] @ D[js] - (A_sym[i] @ D[i]) * ones - np.sum(A_sym[js] * D[js], axis=1) + A_sym[js] @ D[i]
