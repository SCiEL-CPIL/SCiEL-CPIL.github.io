<<<<<<< SEARCH
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return None
=======
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # If verification fails, try to debug by checking which (i,j,k) fail
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_val = 0.0
                    for r in range(rank):
                        tensor_val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(tensor_val - 1.0) > 1e-10:
                        return None
    return None
>>>>>>> REPLACE

