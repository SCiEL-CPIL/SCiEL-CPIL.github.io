"""Rewritten program: 5x6 grid to maximize equal-area triangle multiplicity."""
# EVOLVE-BLOCK-START
def construct():
    pts = []
    # 5 columns, 6 rows
    for i in range(5):
        for j in range(6):
            pts.append((i, j))
    return pts
# EVOLVE-BLOCK-END
