"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard (redundant for O(1) solution but required by directive structure)
    start_time = time.time()
    deadline = start_time + 58.0  # 2s margin before 60s limit

    # Construct a set of size 67.
    # S = {1} U {62} U {64, ..., 128}
    # Size: 1 + 1 + 65 = 67.
    # Validity check:
    # 1. Triples not involving 1 are in {62, 64, ..., 128}.
    #    We verified this subset is valid (sums in set are 126, 127, 128, 
    #    and no triple has all 3 sums in set).
    # 2. Triples involving 1: {1, b, c} with b,c >= 64.
    #    Sums: 1+b, 1+c, b+c.
    #    1+b, 1+c are in {65, ..., 129}.
    #    For 1+b in S, we need 1+b in {64, ..., 128} (since 63 not in S).
    #    So b >= 63. Since b in S and b != 1, 62, we have b >= 64.
    #    Similarly c >= 64.
    #    Then b+c >= 64+65=129.
    #    129 is not in S. So b+c not in S.
    #    Thus no forbidden triple involving 1.
    
    result = [1, 62] + list(range(64, 129))
    
    # Verify time (should be negligible)
    if time.time() > deadline:
        # Fallback to a smaller known valid set if somehow slow (impossible here)
        return list(range(65, 129))
        
    return result
# EVOLVE-BLOCK-END
