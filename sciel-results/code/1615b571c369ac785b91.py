"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    # Based on analysis, the interval [171, 256] is valid.
    # For any a in [171, 256], 3*a > 512 (max possible sum b+c).
    # And 2*a < 2*a+2 <= b+c.
    # So a cannot divide b+c.
    start = 171
    s = list(range(start, n + 1))
    
    # Verify validity to be absolutely sure
    # If invalid, we could try to repair or return a smaller valid set,
    # but the math holds.
    is_valid = True
    for i in range(len(s)):
        a = s[i]
        for j in range(i + 1, len(s)):
            b = s[j]
            for k in range(j, len(s)):
                c = s[k]
                if (b + c) % a == 0:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break
    
    if is_valid:
        return s
    else:
        # Fallback: return a known valid subset, e.g., upper half
        return list(range(192, 257))
# EVOLVE-BLOCK-END
