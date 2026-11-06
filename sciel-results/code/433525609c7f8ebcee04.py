import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    p = 1009
    m = 5

    # Precompute quadratic residues mod 1009 using a bitmask
    # Bit i is 1 if i is a non-zero quadratic residue mod p
    residue_mask = 0
    root_of = [0] * p

    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            residue_mask |= (1 << r)
            if root_of[r] == 0:
                root_of[r] = x

    # Find minimum s in [1, 1004] such that s..s+4 are all non-zero QRs
    # Using bitmask: (residue_mask >> s) & 0b11111 == 0b11111
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
        if (residue_mask >> s) & 0b11111 == 0b11111:
            roots = [root_of[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No run found')
