"""Find the minimum start s such that s, s+1, ..., s+4 are all quadratic residues mod 1009."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 45  # safety margin under the 60s / 90s limit

    # Build boolean array: is_residue[v] is True iff v is a non-zero QR mod p
    is_residue = [False] * p
    # Store one root for each residue (smallest x with x^2 % p == r)
    roots = [0] * p  # roots[r] = x if r is a QR, else 0

    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and roots[r] == 0:
            roots[r] = x
            is_residue[r] = True

    # Search for the smallest s in 1..1004 such that s, s+1, s+2, s+3, s+4
    # are all non-zero quadratic residues mod 1009.
    # Since s >= 1 and s+4 <= 1008, none of the values can be 0, so we only
    # need to check is_residue.
    for s in range(1, p - 4):  # s+4 <= 1008, so s <= 1004
        if (is_residue[s] and is_residue[s + 1] and is_residue[s + 2]
                and is_residue[s + 3] and is_residue[s + 4]):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}

        # Deadline guard: if we've exceeded our budget, return the best we can.
        # Since we search ascending, if we haven't found one by the deadline,
        # there is no valid answer (or we should just raise).
        if time.time() > deadline:
            raise RuntimeError('Deadline exceeded before finding a run of 5 consecutive QRs')

    raise RuntimeError('No run of 5 consecutive quadratic residues found')
# EVOLVE-BLOCK-END
