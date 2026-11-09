import time

_P = 1009
_M = 5


def _build_tables():
    """
    Build:
      - is_qr: bool list of length _P, is_qr[v] True iff v is a non-zero QR mod _P
      - root:  list of length _P, root[v] a witness x in [1,_P-1] with x^2 % _P == v, or 0
      - qr_bitset: integer whose bit v is 1 iff v is a non-zero QR mod _P
    """
    is_qr = [False] * _P
    root = [0] * _P
    qr_bitset = 0
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            is_qr[r] = True
            if root[r] == 0:
                root[r] = x
            qr_bitset |= (1 << r)
    return is_qr, root, qr_bitset


def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin well within the 90s limit

    # Precompute QR indicators, one root per QR, and a bitset of QRs
    is_qr, root, qr_bitset = _build_tables()

    # We need the smallest s in [1, _P - _M] such that s..s+M-1 are all QRs.
    # s+M-1 <= 1008 => s <= 1004
    best_s = None
    best_roots = None

    # Bitset-based search:
    # A run of _M consecutive QRs starting at position s means bits s, s+1, ..., s+M-1 are all set.
    # We can check this by taking the bitset, AND-ing with itself shifted left by 1, 2, ..., M-1,
    # and checking if bit s is set in the result.
    #
    # More efficiently: compute a "run mask" where bit s is 1 iff s..s+M-1 are all QRs.
    # run_mask = qr_bitset & (qr_bitset >> 1) & (qr_bitset >> 2) & (qr_bitset >> 3) & (qr_bitset >> 4)
    # Then find the lowest set bit in run_mask (within valid range s in [1, 1004]).

    # Compute the run mask: bit s is 1 iff bits s, s+1, s+2, s+3, s+4 are all set in qr_bitset
    run_mask = qr_bitset
    for shift in range(1, _M):
        run_mask &= (qr_bitset >> shift)

    # Now find the smallest s in [1, _P - _M] such that bit s is set in run_mask
    # We only care about s from 1 to _P - _M (i.e., 1 to 1004)
    # Create a mask for valid positions: bits 1 through 1004
    valid_mask = 0
    for s in range(1, _P - _M + 1):
        valid_mask |= (1 << s)

    candidate_mask = run_mask & valid_mask

    if candidate_mask != 0:
        # Find the lowest set bit
        # lowest_set_bit = candidate_mask & (-candidate_mask)
        # s = index of that bit
        lowbit = candidate_mask & (-candidate_mask)
        best_s = lowbit.bit_length() - 1
        best_roots = [root[best_s + i] for i in range(_M)]
    else:
        # Fallback: linear scan (should not happen for P=1009, M=5)
        for s in range(1, _P - _M + 1):
            all_qr = True
            for i in range(_M):
                val = s + i
                if not is_qr[val]:
                    all_qr = False
                    break
            if all_qr:
                best_s = s
                best_roots = [root[s + i] for i in range(_M)]
                break

    # This instance is guaranteed to have a solution; still guard against unexpected failure.
    if best_s is None:
        raise RuntimeError("No solution found")

    # Deadline guard (should be far from triggering for this small instance)
    if time.time() > deadline:
        # If we somehow exceeded the deadline, still return the best valid result found.
        pass

    return {'start': best_s, 'roots': best_roots}
