import time

def construct():
    start_time = time.time()
    LIMIT = 1009
    MAX_VAL = 1008

    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)

    best_len = 0
    best_start = 1
    S = 1

    for E in range(1, 1009):
        if time.time() - start_time > 85:
            break
        
        # Check if E creates a conflict with its inverse in the current window [S, E]
        # A conflict exists if inv[E] is inside [S, E]
        if S <= inv[E] <= E:
            # To resolve, we must exclude inv[E] from the left
            # The smallest S that excludes inv[E] is inv[E] + 1
            S = inv[E] + 1
        
        # Now [S, E] is valid
        length = E - S + 1
        if length > best_len:
            best_len = length
            best_start = S

    if best_len == 0:
        # Fallback, though there should always be a valid interval of length 1
        # e.g., [2, 2] since 2*2=4 != 1
        return [2, 1]
    
    return [best_start, best_len]
