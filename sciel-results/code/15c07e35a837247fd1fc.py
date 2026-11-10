"""Find the earliest run of 5 consecutive quadratic residues modulo 1009."""
import time

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Precompute: for each residue value, store a root
    # r^2 mod p = residue
    # We iterate r from 1 to (p-1)//2 to get distinct residues
    root_of = {}
    for r in range(1, (p - 1) // 2 + 1):
        residue = (r * r) % p
        if residue not in root_of:
            root_of[residue] = r
        if time.time() > deadline:
            break

    # Build a bitset: bit i is set if i is a QR mod p (i in 1..1008)
    # Use an integer where bit position i represents value i
    # We need bits 1 through 1008 to be set for residues
    bitset = 0
    for res in root_of:
        bitset |= (1 << res)

    # To find s where s, s+1, s+2, s+3, s+4 are all residues:
    # Check if bits s, s+1, s+2, s+3, s+4 are all set
    # This is equivalent to: (bitset >> s) & 0b11111 == 0b11111
    # But we need to be careful about bit positions

    # Alternative: shift the bitset and AND with itself shifted
    # For 5 consecutive: we need bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    # to have a bit set at position s

    # Compute the AND of shifted versions
    # For position s to be valid, bits s, s+1, s+2, s+3, s+4 must all be set
    # This means in (bitset >> 0) & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4),
    # bit s is set if and only if original bits s, s+1, s+2, s+3, s+4 are all set

    mask5 = 0b11111  # 5 bits
    and_result = bitset
    for shift in range(1, 5):
        and_result &= (bitset >> shift)
        if time.time() > deadline:
            break

    # Now find the lowest set bit in and_result that corresponds to s in [1, 1004]
    # Bit s in and_result means s, s+1, s+2, s+3, s+4 are all residues
    # We need s >= 1 and s+4 <= 1008, so s <= 1004

    best_s = None
    # Find the lowest set bit
    # We can iterate through bits or use bit tricks
    # Since p=1009, we only need to check up to bit 1004

    # Check bits from 1 to 1004
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
