<<<<<<< SEARCH
    # Step 1: Try explicit construction
    if time_left() > 10:
        try:
            n_exp, edges_exp = try_explicit_construction()
            if verify_girth(n_exp, edges_exp, g):
                best_n = n_exp
                best_edges = edges_exp
        except Exception:
            pass
=======
    # Step 1: Try Cayley construction (Faster and potentially better for k=5)
    if time_left() > 10:
        try:
            res = try_cayley_construction()
            if res:
                n_cay, edges_cay = res
                if verify_girth(n_cay, edges_cay, g):
                    best_n = n_cay
                    best_edges = edges_cay
        except Exception:
            pass
        
        # Fallback to LUW if Cayley failed or didn't find
        if best_n is None:
             try:
                n_exp, edges_exp = try_explicit_construction()
                if verify_girth(n_exp, edges_exp, g):
                    best_n = n_exp
                    best_edges = edges_exp
             except Exception:
                pass
>>>>>>> REPLACE
