<<<<<<< SEARCH
    return {"U": U, "V": V, "W": W}
=======
    # Safety: ensure we haven't exceeded time
    elapsed = time.time() - start_time
    if elapsed > 110:
        # Should not happen for this fixed construction, but guard just in case
        pass
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
