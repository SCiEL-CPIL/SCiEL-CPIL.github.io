"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 128}

def construct():
    # The set {63, 64, ..., 128} is valid and has size 66.
    # Sums of distinct elements are >= 127.
    # Only 127 and 128 are in the set.
    # We need 3 distinct sums in the set to form a violation.
    # Since only 2 sums are available, no violation is possible.
    return list(range(63, P['n'] + 1))
# EVOLVE-BLOCK-END
