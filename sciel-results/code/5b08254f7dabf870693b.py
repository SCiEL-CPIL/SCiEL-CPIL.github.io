    def construct():
        # ...
        best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
        best_rank = schoolbook_rank
        # ...
        return {"U": U_list, "V": V_list, "W": W_list}
    