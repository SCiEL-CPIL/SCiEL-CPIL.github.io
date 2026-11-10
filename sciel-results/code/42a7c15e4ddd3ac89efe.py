"""Find the earliest run of 5 consecutive quadratic residues modulo 1009."""
import time

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Precompute: for each residue value, store a root
    # r^2 mod p = residue
    root_of = {}
    for r in range(1, (p - 1) // 2 + 1):
        residue = (r * r) % p
        if residue not in root_of:
            root_of[residue] = r
        if time.time() > deadline:
            break

    # Build a bitset: bit i is set if i is a QR mod p (i in 1..1008)
    bitset = 0
    for res in root_of:
        bitset |= (1 << res)

    # Find s where s, s+1, s+2, s+3, s+4 are all residues
    # Bit s in (bitset & (bitset>>1) & (bitset>>2) & (bitset>>3) & (bitset>>4))
    # is set iff bits s..s+4 in bitset are all set
    and_result = bitset
    for shift in range(1, 5):
        and_result &= (bitset >> shift)
        if time.time() > deadline:
            break

    # Find the lowest set bit in and_result corresponding to s in [1, 1004]
    best_s = None
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        if (and_result >> s) & 1:
            best_s = s
            break

    if best_s is not None:
        return {
            'start': best_s,
            'roots': [root_of[best_s + i] for i in range(5)]
        }

    raise RuntimeError('No run of 5 consecutive quadratic residues found')
