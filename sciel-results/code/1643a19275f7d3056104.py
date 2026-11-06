import time

N = 18
MIN_DIST = 6
_DEADLINE = time.time() + 85

_CODEWORDS = None

def _compute_codewords():
    """Compute the 512 codewords using the QR code construction with bitmask XOR."""
    p = 17
    qr = set((i * i) % p for i in range(p))
    
    basis = []
    for i in range((p + 1) // 2):
        mask = 0
        parity = 0
        for j in range(p):
            if (j - i) % p in qr:
                mask |= (1 << j)
                parity ^= 1
        if parity:
            mask |= (1 << 17)
        basis.append(mask)
    
    codewords = [0] * 512
    for msg in range(512):
        word = 0
        for i in range(9):
            if (msg >> i) & 1:
                word ^= basis[i]
        codewords[msg] = word
        
    return codewords

def _verify(words):
    """Verify all pairs have Hamming distance >= 6 using pure Python."""
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if (wi ^ words[j]).bit_count() < MIN_DIST:
                return False
    return True

def construct():
    global _CODEWORDS, _DEADLINE
    _DEADLINE = time.time() + 85
    
    if _CODEWORDS is not None:
        return _CODEWORDS
    
    codewords = _compute_codewords()
    
    if codewords and len(codewords) == 512:
        if _verify(codewords):
            _CODEWORDS = codewords
            return codewords
    
    return [0]
