total = 0
for c in range(rank):
    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
if total != 1:
    raise ValueError
