import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, coeff in u_coeffs.items():
            U[idx][col] += coeff
        for idx, coeff in v_coeffs.items():
            V[idx][col] += coeff
        for idx, coeff in w_coeffs.items():
            W[idx][col] += coeff
        col += 1

    # Block 1
    add_term({0: 1, 4: 1}, {0: 1, 5: 1}, {0: 1, 3: 1})
    add_term({3: 1, 4: 1}, {0: 1}, {1: 1, 3: -1})
    add_term({0: 1}, {1: 1, 5: -1}, {2: 1, 3: 1})
    add_term({4: 1}, {4: 1, 0: -1}, {0: 1, 1: 1})
    add_term({0: 1, 1: 1}, {5: 1}, {0: -1, 2: 1})
    add_term({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    add_term({1: 1, 4: -1}, {4: 1, 5: 1}, {0: 1})

    # Block 2
    add_term({0: 1, 4: 1}, {2: 1, 7: 1}, {4: 1, 7: 1})
    add_term({3: 1, 4: 1}, {2: 1}, {5: 1, 7: -1})
    add_term({0: 1}, {3: 1, 7: -1}, {6: 1, 7: 1})
    add_term({4: 1}, {6: 1, 2: -1}, {4: 1, 5: 1})
    add_term({0: 1, 1: 1}, {7: 1}, {4: -1, 6: 1})
    add_term({3: 1, 0: -1}, {2: 1, 3: 1}, {7: 1})
    add_term({1: 1, 4: -1}, {6: 1, 7: 1}, {4: 1})

    # Block 3
    add_term({2: 1}, {8: 1}, {0: 1})
    add_term({2: 1}, {9: 1}, {2: 1})
    add_term({2: 1}, {10: 1}, {4: 1})
    add_term({2: 1}, {11: 1}, {6: 1})
    add_term({5: 1}, {8: 1}, {1: 1})
    add_term({5: 1}, {9: 1}, {3: 1})
    add_term({5: 1}, {10: 1}, {5: 1})
    add_term({5: 1}, {11: 1}, {7: 1})

    return {"U": U, "V": V, "W": W}
