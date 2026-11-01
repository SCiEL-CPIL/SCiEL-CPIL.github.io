def construct():
    n, m, p = 2, 3, 4
    rank = 22

    U = [[0] * rank for _ in range(n * m)]  # 6 x rank
    V = [[0] * rank for _ in range(m * p)]  # 12 x rank
    W = [[0] * rank for _ in range(p * n)]  # 8 x rank

    # A indices: A_{i,j} -> i*m+j
    # B indices: B_{j,k} -> j*p+k
    # C indices: C_{i,k} -> k*n+i (trace convention W[k*n+i, r])

    # Define the 22 terms
    # Each term is (A_dict, B_dict, C_dict)
    # A_dict: {index: coeff}
    # B_dict: {index: coeff}
    # C_dict: {index: coeff}

    terms = [
        # P1
        ({0: 1, 3: 1}, {2: 1, 5: 1}, {0: 1, 4: 1}),
        # P2
        ({0: 1, 4: 1}, {2: 1, 5: 1}, {1: 1, 5: 1}),
        # P3
        ({0: 1, 3: 1}, {3: 1, 6: 1}, {0: 1, 4: 1}),
        # P4
        ({0: 1, 4: 1}, {3: 1, 6: 1}, {1: 1, 5: 1}),
        # P5
        ({1: 1, 3: 1}, {4: 1, 7: 1}, {0: 1, 4: 1}),
        # P6
        ({1: 1, 4: 1}, {4: 1, 7: 1}, {1: 1, 5: 1}),
        # P7
        ({1: 1, 3: 1}, {5: 1, 8: 1}, {0: 1, 4: 1}),
        # P8
        ({1: 1, 4: 1}, {5: 1, 8: 1}, {1: 1, 5: 1}),
        # P9
        ({2: 1}, {9: 1}, {2: 1}),
        # P10
        ({5: 1}, {9: 1}, {6: 1}),
        # P11
        ({2: 1}, {10: 1}, {3: 1}),
        # P12
        ({5: 1}, {10: 1}, {7: 1}),
        # Q1
        ({0: 1, 4: 1}, {2: 1, 7: 1}, {2: 1, 7: 1}),
        # Q2
        ({3: 1, 4: 1}, {2: 1}, {6: 1, 7: -1}),
        # Q3
        ({0: 1}, {3: 1, 7: -1}, {3: 1, 7: 1}),
        # Q4
        ({4: 1}, {6: 1, 2: -1}, {2: 1, 6: 1}),
        # Q5
        ({0: 1, 1: 1}, {7: 1}, {2: -1, 3: 1}),
        # Q6
        ({3: 1, 0: -1}, {2: 1, 3: 1}, {7: 1}),
        # Q7
        ({1: 1, 4: -1}, {6: 1, 7: 1}, {2: 1}),
        # Q8
        ({2: 1}, {10: 1}, {2: 1}),
        # Q9
        ({2: 1}, {11: 1}, {3: 1}),
        # Q10
        ({5: 1}, {10: 1}, {6: 1}),
        # Q11
        ({5: 1}, {11: 1}, {7: 1}),
    ]

    for r, (A_dict, B_dict, C_dict) in enumerate(terms):
        for idx, coeff in A_dict.items():
            U[idx][r] += coeff
        for idx, coeff in B_dict.items():
            V[idx][r] += coeff
        for idx, coeff in C_dict.items():
            W[idx][r] += coeff

    return {"U": U, "V": V, "W": W}
