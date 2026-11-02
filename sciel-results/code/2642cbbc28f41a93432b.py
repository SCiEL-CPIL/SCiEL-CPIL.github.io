def objective(x):
    Ux = x[:n*m*rank].reshape(n*m, rank)
    Vx = x[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
    Wx = x[(n*m+m*p)*rank:].reshape(p*n, rank)
    
    terms = np.sum(Ux[u_idx] * Vx[v_idx] * Wx[w_idx], axis=1)
    return np.sum((terms - 1.0) ** 2)
