"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Set S = [63, 128]. For any distinct a < b < c in S,
    # b + c >= 64 + 65 = 129 > 128, so b+c is never in S.
    # Hence no forbidden triple exists. Size = 128 - 63 + 1 = 66.
    return list(range(63, 129))
# EVOLVE-BLOCK-END
