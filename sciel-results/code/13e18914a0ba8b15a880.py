def _merge_terms(U, V, W, n, m, p):
    """Merge terms that are proportional."""
    rank = len(U[0])
    current_U, current_V, current_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    current_rank = rank
    changed = True
    while changed:
        changed = False
        for i in range(current_rank):
            for j in range(i+1, current_rank):
                # Check if U_i and U_j are proportional
                # If U_i is zero, skip? No, if U_i is zero, term is zero, remove it.
                if all(x == 0 for x in current_U[i]):
                    current_U.pop(i)
                    current_V.pop(i)
                    current_W.pop(i)
                    current_rank -= 1
                    changed = True
                    break
                if all(x == 0 for x in current_V[i]):
                    current_U.pop(i)
                    current_V.pop(i)
                    current_W.pop(i)
                    current_rank -= 1
                    changed = True
                    break
                if all(x == 0 for x in current_W[i]):
                    current_U.pop(i)
                    current_V.pop(i)
                    current_W.pop(i)
                    current_rank -= 1
                    changed = True
                    break
                    
                # Check proportionality
                # Find non-zero element in U_i
                alpha = None
                for k in range(len(current_U[i])):
                    if current_U[i][k] != 0:
                        if current_U[j][k] != 0:
                            alpha = current_U[j][k] / current_U[i][k]
                        else:
                            alpha = None
                        break
                    else:
                        if current_U[j][k] != 0:
                            alpha = None
                            break
                if alpha is None:
                    continue
                
                # Check if U_j = alpha * U_i
                match_U = True
                for k in range(len(current_U[i])):
                    if abs(current_U[j][k] - alpha * current_U[i][k]) > 1e-9:
                        match_U = False
                        break
                if not match_U:
                    continue
                
                # Check V and W proportionality
                beta = None
                for k in range(len(current_V[i])):
                    if current_V[i][k] != 0:
                        if current_V[j][k] != 0:
                            beta = current_V[j][k] / current_V[i][k]
                        else:
                            beta = None
                        break
                    else:
                        if current_V[j][k] != 0:
                            beta = None
                            break
                if beta is None:
                    continue
                
                match_V = True
                for k in range(len(current_V[i])):
                    if abs(current_V[j][k] - beta * current_V[i][k]) > 1e-9:
                        match_V = False
                        break
                if not match_V:
                    continue
                
                gamma = None
                for k in range(len(current_W[i])):
                    if current_W[i][k] != 0:
                        if current_W[j][k] != 0:
                            gamma = current_W[j][k] / current_W[i][k]
                        else:
                            gamma = None
                        break
                    else:
                        if current_W[j][k] != 0:
                            gamma = None
                            break
                if gamma is None:
                    continue
                
                match_W = True
                for k in range(len(current_W[i])):
                    if abs(current_W[j][k] - gamma * current_W[i][k]) > 1e-9:
                        match_W = False
                        break
                if not match_W:
                    continue
                
                # Merge: add term j to term i
                # U_i_new = U_i + U_j = (1+alpha) U_i
                # V_i_new = V_i + V_j = (1+beta) V_i
                # W_i_new = W_i + W_j = (1+gamma) W_i
                # But we need to keep the sum of products.
                # Term i is U_i V_i W_i. Term j is U_j V_j W_j.
                # If U_j = a U_i, V_j = b V_i, W_j = c W_i, then Term j = abc U_i V_i W_i.
                # Sum = (1+abc) U_i V_i W_i.
                # We can replace Term i with (1+abc) U_i V_i W_i.
                # This is a single rank-1 term.
                scale = 1 + alpha * beta * gamma
                for k in range(len(current_U[i])):
                    current_U[i][k] = current_U[i][k] * scale
                for k in range(len(current_V[i])):
                    current_V[i][k] = current_V[i][k] * scale
                # Wait, if I scale U, I don't need to scale V and W?
                # No, the term is U \otimes V \otimes W.
                # If I multiply U by scale, the term becomes scale * (U \otimes V \otimes W).
                # So I just need to scale one of the vectors.
                # Let's scale U.
                current_U[i] = [x * scale for x in current_U[i]]
                
                # Remove term j
                current_U.pop(j)
                current_V.pop(j)
                current_W.pop(j)
                current_rank -= 1
                changed = True
                break
            if changed:
                break
    return current_U, current_V, current_W, current_rank
