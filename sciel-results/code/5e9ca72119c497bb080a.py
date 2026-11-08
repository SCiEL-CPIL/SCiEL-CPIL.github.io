<<<<<<< SEARCH
    # Deadline guard
    if time.time() - start_time > 110:
        # If we somehow took too long, return what we have (which is complete)
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    # Deadline guard with early return if construction is incomplete
    elapsed = time.time() - start_time
    if elapsed > 110:
        # Return whatever we have; the construction is deterministic and fast
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
