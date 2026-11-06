import numpy as np
import time

def construct():
    N = 18
    MIN_DIST = 6
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # 1. Try Extended QR Code Construction (deterministic, ~instant)
    try:
        # Quadratic residues mod 17: {0, 1, 2, 4, 8, 9, 13, 15, 16}
        residues = [0, 1, 2, 4, 8, 9, 13, 15, 16]
        v = np.zeros(17, dtype=np.uint8)
        v[residues] = 1
        
        # Build generator matrix: 9 rows x 18 cols (17 QR positions + 1 parity)
        G = np.zeros((9, 18), dtype=np.uint8)
        for i in range(9):
            row = np.roll(v, i)
            G[i, :17] = row
            G[i, 17] = int(row.sum()) % 2
        
        # Precompute int versions of rows for fast XOR
        row_ints = []
        for j in range(9):
            val = 0
            for k in range(18):
                if G[j, k]:
                    val |= (1 << k)
            row_ints.append(val)
        
        # Generate all 512 codewords
        codes = []
        for m in range(512):
            c = 0
            for j in range(9):
                if (m >> j) & 1:
                    c ^= row_ints[j]
            codes.append(c)
        
        # Verify all pairs have distance >= 6
        min_d = N + 1
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                d = (codes[i] ^ codes[j]).bit_count()
                if d < min_d:
                    min_d = d
                if min_d < MIN_DIST:
                    break
            if min_d < MIN_DIST:
                break
        
        if min_d >= MIN_DIST:
            return codes
    except Exception:
        pass
    
    # Fallback: Greedy with conflict tracking
    try:
        codes = [0]  # Start with all-zeros word
        # For each candidate, track which existing codewords it conflicts with (dist < 6)
        # Use a set of forbidden words: any word within distance 5 of an existing codeword
        
        forbidden = set()
        forbidden.add(0)
        # Mark all words at distance <= 5 from 0 as forbidden
        for w in range(1, 2**N):
            if w.bit_count() < MIN_DIST:  # dist from 0 = popcount(w)
                forbidden.add(w)
        
        # Greedy: iterate through all words, add if not forbidden
        for w in range(1, 2**N):
            if time.time() > deadline:
                break
            if w not in forbidden:
                codes.append(w)
                # Mark all words within distance 5 of w as forbidden
                for w2 in range(2**N):
                    if (w ^ w2).bit_count() < MIN_DIST:
                        forbidden.add(w2)
    except Exception:
        pass
    
    return codes if codes else [0]
