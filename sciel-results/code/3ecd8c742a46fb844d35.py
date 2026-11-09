_P = 1009
_M = 5

def _build_tables():
    is_qr = [False] * _P
    root = [0] * _P
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            is_qr[r] = True
            if root[r] == 0:
                root[r] = x
    return is_qr, root

_IS_QR, _ROOT = _build_tables()

def construct():
    for s in range(1, _P - _M + 1):
        if _IS_QR[s] and _IS_QR[s+1] and _IS_QR[s+2] and _IS_QR[s+3] and _IS_QR[s+4]:
            return {'start': s, 'roots': [_ROOT[s], _ROOT[s+1], _ROOT[s+2], _ROOT[s+3], _ROOT[s+4]]}
            
    raise RuntimeError("No solution found")
