"""Find the earliest run of 5 consecutive quadratic residues modulo 1009."""

def construct():
    p = 1009
    half = p // 2  # 504

    # Compute all quadratic residues and store one root per residue
    # r^2 mod p for r in 1..504 gives 504 distinct nonzero QRs
    # (since p ≡ 1 mod 4, no r in 1..504 has r^2 ≡ 0 mod p)
    root_map = {}
    for r in range(1, half + 1):
        val = (r * r) % p
        if val not in root_map:
            root_map[val] = r

    # Convert to a lookup array for fast membership test
    # is_qr[x] is True iff x is a nonzero quadratic residue mod p
    is_qr = [False] * p
    for v in root_map:
        is_qr[v] = True

    # Scan s from 1 upward; first s where s..s+4 are all QR is the answer
    # s must satisfy 1 <= s <= 1004 (so s+4 <= 1008, no wraparound, no zero)
    for s in range(1, p - 4):
        if is_qr[s] and is_qr[s + 1] and is_qr[s + 2] and is_qr[s + 3] and is_qr[s + 4]:
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No run of 5 consecutive quadratic residues found')
