"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    MOD = 1009
    N = 1008

    # Precompute modular inverses for 1..1008
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)

    best = [1, 0]
    L = 1
    bad_count = 0
    
    for R in range(1, N + 1):
        # Add R to window
        if L <= inv[R] <= R:
            if R == inv[R]:
                bad_count += 1
            else:
                bad_count += 2
        
        # Shrink L until window is valid
        while L <= R and bad_count > 0:
            # Remove L from window
            if L <= inv[L] <= R:
                if L == inv[L]:
                    bad_count -= 1
                else:
                    bad_count -= 2
            L += 1
        
        if L <= R:
            length = R - L + 1
            if length > best[1]:
                best = [L, length]
        
        # Safety check for deadline
        if time.time() > deadline:
            break

    return best
# EVOLVE-BLOCK-END
