import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # Generator polynomial for the [17,9,5] cyclic code which is the 
    # unextended quadratic residue code of length 17.
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Coefficients: x^8, x^5, x^4, x^3, x^0
    # Integer value: 2^8 + 2^5 + 2^4 + 2^3 + 2^0
    g = (1 << 8) | (1 << 5) | (1 << 4) | (1 << 3) | (1 << 0)
    
    words = []
    
    # Iterate over all 2^9 = 512 possible messages
    # The code is linear, so we generate all linear combinations of the generator
    for msg in range(512):
        # Compute the 17-bit codeword
        # c(x) = m(x) * g(x) mod (x^17 - 1) is NOT needed because deg(m)<=8, deg(g)=8 => deg(c)<=16
        # So it's just simple polynomial multiplication (XOR of shifted g)
        
        c_val = 0
        m = msg
        shift = 0
        while m:
            if m & 1:
                c_val ^= (g << shift)
            m >>= 1
            shift += 1
            
        # Calculate parity bit for extension to length 18
        # The [17,9,5] code has odd minimum distance.
        # Adding an overall parity bit makes it a [18,9,6] code.
        parity = bin(c_val).count('1') & 1
        
        # Form the 18-bit word
        # c_val occupies bits 0..16
        # parity bit goes to bit 17
        word = c_val
        if parity:
            word |= (1 << 17)
            
        words.append(word)
        
        # Check time
        if time.time() > deadline:
            return words
            
    return words
