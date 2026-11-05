import numpy as np

def construct():
    # Helper to multiply polynomials over GF(2)
    def poly_mul(p1, p2):
        res = [0] * (len(p1) + len(p2) - 1)
        for i, c1 in enumerate(p1):
            if c1:
                for j, c2 in enumerate(p2):
                    if c2:
                        res[i+j] ^= 1
        return res

    # f1(x) = x^8 + x^4 + x^3 + x^2 + 1
    # Coeffs: x^0 ... x^8
    f1 = [1, 0, 0, 1, 1, 1, 0, 0, 1]
    
    # Generate basis for [17, 9, 5] code
    # Basis vectors are f1(x) * x^i for i=0..8
    basis = []
    for i in range(9):
        # Shift f1 by i
        vec = [0] * 17
        for j in range(len(f1)):
            if i + j < 17:
                vec[i+j] = f1[j]
        basis.append(vec)
        
    # Generate all 512 codewords
    words_17 = []
    for mask in range(512):
        word = [0] * 17
        for i in range(9):
            if mask & (1 << i):
                for j in range(17):
                    word[j] ^= basis[i][j]
        words_17.append(word)
        
    # Extend to 18 bits
    words_18 = []
    for w in words_17:
        parity = sum(w) % 2
        words_18.append(w + [parity])
        
    # Shorten: keep words where bit 0 is 0
    final_words = []
    for w18 in words_18:
        if w18[0] == 0:
            # Remove index 0
            new_word = w18[1:]
            # Convert to integer
            val = 0
            for i, bit in enumerate(new_word):
                if bit:
                    val |= (1 << i)
            final_words.append(val)
            
    return final_words
