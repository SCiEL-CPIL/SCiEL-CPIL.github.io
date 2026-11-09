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
    deadline = start_time + 80.0  # Safety margin

    # Precompute QR indicators and one root per QR
    is_qr, root = _build_tables()

    # We need the smallest s in [1, _P - _M] such that s..s+M-1 are all QRs.
    # s+M-1 <= 1008 => s <= 1004
    best_s = None
    best_roots = None

    # Linear scan from the minimum possible s upward; first hit is optimal.
    for s in range(1, _P - _M + 1):
        # Quick check: all of s..s+M-1 must be QRs
        ok = True
        for i in range(_M):
            v = s + i
            if not is_qr[v]:
                ok = False
                break
        if ok:
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
