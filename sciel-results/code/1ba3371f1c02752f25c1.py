import time
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generator polynomial powers for [17, 9, 5] QR code
    g_powers = [0, 3, 4, 5, 8]
    
    basis_18 = []
    for i in range(9):
        word = 0
        for p in g_powers:
            pos = (i + p) % 17
            word |= (1 << pos)
        # Extend to [18, 9, 6] with parity bit
        if word.bit_count() % 2:
            word |= (1 << 17)
        basis_18.append(word)
        
    # Generate 512 codewords
    b = basis_18
    current_code = [0] * 512
    for msg in range(512):
        word = 0
        # Unroll for speed? 9 iterations is small.
        if msg & 1: word ^= b[0]
        if msg & 2: word ^= b[1]
        if msg & 4: word ^= b[2]
        if msg & 8: word ^= b[3]
        if msg & 16: word ^= b[4]
        if msg & 32: word ^= b[5]
        if msg & 64: word ^= b[6]
        if msg & 128: word ^= b[7]
        if msg & 256: word ^= b[8]
        current_code[msg] = word
        
    return current_code
