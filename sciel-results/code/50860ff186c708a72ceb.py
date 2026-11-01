    if best_error < 1e-9:
        U_out = _round_to_half(best_U)
        V_out = _round_to_half(best_V)
        W_out = _round_to_half(best_W)
        # Verify rounded solution
        if _check_decomposition(U_out, V_out, W_out, _build_tensor(n, m, p), n, m, p):
            return U_out.reshape(n*m, r), V_out.reshape(m*p, r), W_out.reshape(p*n, r), True
        else:
            # If rounding breaks it, return original if it's very close
            return best_U.reshape(n*m, r), best_V.reshape(m*p, r), best_W.reshape(p*n, r), True
