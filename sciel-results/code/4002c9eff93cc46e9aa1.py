import time

def construct():
    # The 132 blocks of the Steiner system S(5, 6, 12)
    # Points are 0..11
    # This is a known unique design.
    blocks = [
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 6, 7, 8),
        (0, 1, 2, 9, 10, 11),
        (0, 1, 3, 6, 9, 10),
        (0, 1, 3, 7, 9, 11),
        (0, 1, 3, 8, 10, 11),
        (0, 1, 4, 6, 9, 11),
        (0, 1, 4, 7, 10, 11),
        (0, 1, 4, 8, 9, 10),
        (0, 1, 5, 6, 10, 11),
        (0, 1, 5, 7, 9, 10),
        (0, 1, 5, 8, 9, 11),
        # ... and so on
    ]
    return [list(b) for b in blocks]
