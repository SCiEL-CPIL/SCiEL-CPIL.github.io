def get_strassen_2x2():
    # Returns U, V, W as lists of lists of coefficients for 7 terms
    # U: 4 rows, V: 4 rows, W: 4 rows
    terms = []
    # M1
    terms.append({
        'U': {0: 1, 3: 1},
        'V': {0: 1, 3: 1},
        'W': {0: 1, 3: 1}
    })
    # M2
    terms.append({
        'U': {2: 1, 3: 1},
        'V': {0: 1},
        'W': {2: 1, 3: -1}
    })
    # M3
    terms.append({
        'U': {0: 1},
        'V': {1: 1, 3: -1},
        'W': {1: 1, 3: 1}
    })
    # M4
    terms.append({
        'U': {3: 1},
        'V': {2: 1, 0: -1},
        'W': {0: 1, 2: 1}
    })
    # M5
    terms.append({
        'U': {0: 1, 1: 1},
        'V': {3: 1},
        'W': {0: -1, 1: 1}
    })
    # M6
    terms.append({
        'U': {2: 1, 0: -1},
        'V': {0: 1, 1: 1},
        'W': {3: 1}
    })
    # M7
    terms.append({
        'U': {1: 1, 3: -1},
        'V': {2: 1, 3: 1},
        'W': {0: 1}
    })
    return terms
