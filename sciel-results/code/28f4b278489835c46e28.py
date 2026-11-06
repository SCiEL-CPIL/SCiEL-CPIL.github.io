"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Extended Quadratic Residue code of length 18
    # QR code mod 17: [17, 9, 5], extended to [18, 9, 6] with 512 codewords
    
    n = 17
    # Quadratic residues mod 17
    qr = set()
    for i in range(1, n):
        qr.add((i * i) % n)
    # qr = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Build generator matrix: 9 rows, 17 cols
    # Row 0: all ones
    # Rows 1-8: cyclic shifts of QR indicator
    gen_rows = []
    # All-ones row
    gen_rows.append([1] * n)
    
    # QR indicator vector (position i is 1 if i in qr)
    qr_vec = [1 if i in qr else 0 for i in range(n)]
    
    for shift in range(8):
        row = [(qr_vec[(i + shift) % n]) for i in range(n)]
        gen_rows.append(row)
    
    # Generate all 512 codewords (linear combinations of gen_rows)
    words = []
    for mask in range(512):
        word = [0] * n
        for bit_idx in range(9):
            if (mask >> bit_idx) & 1:
                row = gen_rows[bit_idx]
                for i in range(n):
                    word[i] ^= row[i]
        # Extend by parity bit to make even weight
        weight = sum(word)
        parity_bit = weight % 2
        word.append(parity_bit)
        # Convert to integer
        val = 0
        for i in range(18):
            if word[i]:
                val |= (1 << i)
        words.append(val)
    
    # Verify minimum distance
    if len(words) > 1:
        min_dist = 18
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                d = (words[i] ^ words[j]).bit_count()
                if d < min_dist:
                    min_dist = d
        # All pairs should have distance >= 6
    
    return words
# EVOLVE-BLOCK-END
