        # M1 = (A00 + A11)(B00 + B11)
        add_term(
            {(0 + j_offset) * m + 0: 1, (1 + j_offset) * m + 1: 1},
            {(0 + j_offset) * p + (0 + k_offset): 1, (1 + j_offset) * p + (1 + k_offset): 1},
            {(0 + k_offset) * n + 0: 1, (1 + k_offset) * n + 1: 1}
        )
        # M2 = (A10 + A11)B00
        add_term(
            {(1 + j_offset) * m + 0: 1, (1 + j_offset) * m + 1: 1},
            {(0 + j_offset) * p + (0 + k_offset): 1},
            {(0 + k_offset) * n + 1: 1, (1 + k_offset) * n + 1: -1}
        )
        # M3 = A00(B01 - B11)
        add_term(
            {(0 + j_offset) * m + 0: 1},
            {(0 + j_offset) * p + (1 + k_offset): 1, (1 + j_offset) * p + (1 + k_offset): -1},
            {(1 + k_offset) * n + 0: 1, (1 + k_offset) * n + 1: 1}
        )
        # M4 = A11(B10 - B00)
        add_term(
            {(1 + j_offset) * m + 1: 1},
            {(1 + j_offset) * p + (0 + k_offset): 1, (0 + j_offset) * p + (0 + k_offset): -1},
            {(0 + k_offset) * n + 1: 1, (0 + k_offset) * n + 0: 1}
        )
        # M5 = (A00 + A01)B11
        add_term(
            {(0 + j_offset) * m + 0: 1, (0 + j_offset) * m + 1: 1},
            {(1 + j_offset) * p + (1 + k_offset): 1},
            {(0 + k_offset) * n + 0: -1, (1 + k_offset) * n + 0: 1}
        )
        # M6 = (A10 - A00)(B00 + B01)
        add_term(
            {(1 + j_offset) * m + 0: 1, (0 + j_offset) * m + 0: -1},
            {(0 + j_offset) * p + (0 + k_offset): 1, (0 + j_offset) * p + (1 + k_offset): 1},
            {(1 + k_offset) * n + 1: 1}
        )
        # M7 = (A01 - A11)(B10 + B11)
        add_term(
            {(0 + j_offset) * m + 1: 1, (1 + j_offset) * m + 1: -1},
            {(1 + j_offset) * p + (0 + k_offset): 1, (1 + j_offset) * p + (1 + k_offset): 1},
            {(0 + k_offset) * n + 0: 1}
        )
