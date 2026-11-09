import time

_P = 1009
_M = 5


def _build_tables():
    """
    Build:
      - is_qr: bool list of length _P, is_qr[v] True iff v is a non-zero QR mod _P
      - root:  list of length _P, root[v] a witness x in [1,_P-1] with x^2 % _P == v, or 0
    """
    is_qr = [False] * _P
    root = [0] * _P
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            is_qr[r] = True
            if root[r] == 0:
                root[r] = x
    return is_qr, root


def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin well within the 60s limit

    # Precompute QR indicators and one root per QR
    is_qr, root = _build_tables()

    # We need the smallest s in [1, _P - _M] such that s..s+M-1 are all QRs.
    # s+M-1 <= 1008 => s <= 1004
    best_s = None
    best_roots = None

    # Linear scan from the minimum possible s upward; first hit is optimal.
    # Since M=5 is small, we unroll the check for speed, though it's negligible here.
    for s in range(1, _P - _M + 1):
        if time.time() > deadline:
            # If we somehow hit the deadline (extremely unlikely for P=1009),
            # we would need a fallback. But for this specific instance, 
            # the loop is tiny. We raise to signal failure if logic is broken,
            # but practically this never triggers.
            if best_s is not None:
                break
            else:
                raise TimeoutError("Search timed out before finding solution")
                
        # Check if s, s+1, ..., s+4 are all QRs
        if is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]:
            best_s = s
            best_roots = [root[s], root[s+1], root[s+2], root[s+3], root[s+4]]
            break

    # This instance is guaranteed to have a solution; still guard against unexpected failure.
    if best_s is None:
        raise RuntimeError("No solution found")

    return {'start': best_s, 'roots': best_roots}
