import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill

    p = 1009

    # Precompute: for each non-zero residue, store one root
    root_arr = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_arr[r] == 0:
            root_arr[r] = x

    # Build a bitmask where bit v is set if v is a non-zero quadratic residue
    mask = 0
    for v in range(1, p):
        if root_arr[v] != 0:
            mask |= (1 << v)

    # Check for 5 consecutive set bits starting at position s
    # (mask >> s) & 0x1F == 0x1F means bits s, s+1, s+2, s+3, s+4 are all set
    target = 0x1F  # 11111 in binary

    for s in range(1, 1005):
        if time.time() > deadline:
            break
        if ((mask >> s) & target) == target:
            return {'start': s, 'roots': [root_arr[s + i] for i in range(5)]}

    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
