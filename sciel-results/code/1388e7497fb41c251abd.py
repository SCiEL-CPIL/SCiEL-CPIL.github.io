<<<<<<< SEARCH
    # Final polish: one more hill climb pass
    if time.time() < _DEADLINE - 5.0:
        family, score = _hill_climb_family(family)
=======
    # Final polish: one more hill climb pass (with 2-opt)
    if time.time() < _DEADLINE - 5.0:
        family, score = _hill_climb_family(family, deadline_margin=2.0)
>>>>>>> REPLACE

