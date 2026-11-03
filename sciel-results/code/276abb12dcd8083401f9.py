<<<<<<< SEARCH
    return {"U": U, "V": V, "W": W}
=======
    if time.time() - _t0 > _deadline:
        raise TimeoutError("Exceeded deadline")
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
