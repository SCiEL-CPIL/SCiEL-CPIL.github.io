<<<<<<< SEARCH
    if best_code is not None and time.time() < deadline - 2.0:
        ls_code = local_search(best_code, 12345, deadline - 2.0)
        if len(ls_code) < best_len:
            best_len = len(ls_code)
            best_code = ls_code

    if best_code is not None:
        final_code = refine(best_code)
    else:
        final_code = []
=======
    if best_code is not None and time.time() < deadline - 2.0:
        ls_code = local_search(best_code, 12345, deadline - 2.0)
        if len(ls_code) < best_len:
            best_len = len(ls_code)
            best_code = ls_code

    # ILS kick-repair phase
    if best_code is not None and time.time() < deadline - 2.0:
        ils_code = kick_repair(best_code, 99999, deadline - 2.0)
        if len(ils_code) < best_len:
            best_len = len(ils_code)
            best_code = ils_code

    if best_code is not None:
        final_code = refine(best_code)
    else:
        final_code = []
>>>>>>> REPLACE
