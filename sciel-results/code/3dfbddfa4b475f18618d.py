"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 10
    sq = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append((i + j) % n)
        sq.append(row)
    return [sq]
# EVOLVE-BLOCK-END
