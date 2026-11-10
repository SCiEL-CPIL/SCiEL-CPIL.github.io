import time

def construct():
    deadline = time.time() + 88.0  # safety margin before 90s wall-clock limit

    p = 1009
    # Precompute modular inverses for all residues 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)

    best_start = 2
    best_len = 1  # [2,2] is always valid: 2*2 = 4 != 1 mod 1009

    # Use a boolean array for O(1) membership instead of a set
    in_interval = [False] * (p + 1)

    for s in range(2, p - 1):  # skip s=1 and s=1008 (self-inverse)
        if time.time() > deadline:
            break
        # Reset the boolean array (only need to clear positions >= s)
        # But since s increases, we can just clear what we added
        # Simpler: use a generation counter or just reset all
        # For p=1009, resetting 1009 booleans is cheap
        for i in range(s, p):
            in_interval[i] = False
        
        for e in range(s, p):
            if time.time() > deadline:
                break
            if e == p - 1:  # 1008 is self-inverse
                break
            ie = inv[e]
            if in_interval[ie]:
                break  # e and inv[e] would both be in interval
            in_interval[e] = True
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    return [best_start, best_len]
