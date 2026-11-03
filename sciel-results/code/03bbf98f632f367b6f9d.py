for u in range(nm):
    i = u // m
    j = u % m
    for v in range(mp):
        j_v = v // p
        k = v % p
        if j_v != j:
            uv = U[u, :] * V[v, :]
            for w in range(pn):
                if abs(np.dot(uv, W[w, :])) > 1e-9:
                    return False
        else:
            w_target = k * n + i
            uv = U[u, :] * V[v, :]
            if abs(np.dot(uv, W[w_target, :]) - 1.0) > 1e-9:
                return False
