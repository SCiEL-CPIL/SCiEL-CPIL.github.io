def construct():
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 145.0

    known_5 = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
               (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
               (Fraction(16), Fraction(63))]
    
    # Validate the known 5
    ok = True
    for i in range(5):
        for j in range(i + 1, 5):
            if not _valid_distance(known_5[i][0], known_5[i][1], known_5[j][0], known_5[j][1], 1):
                ok = False
                break
        if not ok:
            break
    if ok and _check_general_position(known_5, 1):
        return _build_result(known_5, 1)
    
    # Fallback: return a simple valid set of 3 points if verification fails (should not happen)
    fallback = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4))]
    return _build_result(fallback, 1)
