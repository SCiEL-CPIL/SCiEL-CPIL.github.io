def update(state, item):
    b = bytearray(state)
    for r in range(16):
        p = PRIMES[r]
        s = SALTS[r]
        idx = ((item * p + s) % MOD) % 64
        pos = r * 64 + idx
        b[pos] = (b[pos] + 1) & 0xFF
    return bytes(b)

def query(state, item):
    min_val = 255
    for r in range(16):
        p = PRIMES[r]
        s = SALTS[r]
        idx = ((item * p + s) % MOD) % 64
        pos = r * 64 + idx
        val = state[pos]
        if val < min_val:
            min_val = val
    return min_val
