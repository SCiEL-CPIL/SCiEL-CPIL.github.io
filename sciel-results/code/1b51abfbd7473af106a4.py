    in_set_full = np.zeros(N, dtype=bool)
    in_set_full[:22] = True
    in_set_full[22:][in_set] = True
    
    result = V[in_set_full]
    return result
