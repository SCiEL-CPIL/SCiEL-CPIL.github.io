"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # We found a solution with max value 240.
    # Integers: 224, 240.
    # 224 * 240 * 210 = 11289600 = 3360^2.
    # Both > 210 and <= 10210.
    result = [224, 240]
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
