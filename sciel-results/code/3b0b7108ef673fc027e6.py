"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a binary error-correcting code of length 18 and minimum distance 6.
    
    The optimal size A_2(18, 6) is known to be 512. We construct this using
    the extended quadratic residue code construction or equivalent algebraic method.
    
    One known construction achieving 512 words of length 18 with distance 6:
    Use the [18, 9, 6] extended quadratic residue code over GF(2).
    
    The quadratic residue code of length 19 (p=19) is a [19, 10, 7] code.
    Extending it by a parity bit gives a [20, 10, 8] code. Shortening appropriately
    or using a different approach...
    
    Actually, let's use a known construction:
    The [18, 9, 6] code can be obtained from the [16, 11, 4] first-order Reed-Muller
    with extensions, but that only gives 2048 words with distance 4.
    
    Better approach: The extended quadratic residue code of length 20 (from p=19)
    has parameters [20, 10, 8]. If we shorten it by 2 positions, we get a [18, 8, 6]
    code with 256 words. That's not enough.
    
    Let me reconsider. A_2(18, 6) = 512 is achieved by a nonlinear code.
    
    One construction: Take the [18, 9, 6] linear code if it exists.
    Actually, there is a [18, 9, 6] linear code (the extended ternary Golay related?).
    
    Let me use a different approach: The code can be constructed as follows:
    Consider the set of all vectors (x, y, z) where x is in F_2^4, y is in F_2^4, 
    z is in F_2^4, and some constraint... 
    
    Actually, the simplest known construction for A_2(18,6) = 512 uses the 
    following: Take the 9-dimensional linear code over F_2 of length 18 with 
    minimum distance 6. Such a code exists (it's the [18, 9, 6] code).
    
    One way to construct it: The [18, 9, 6] code is equivalent to the 
    extended quadratic residue code of length 18? No, QR codes have prime length.
    
    Let me use a concrete construction. The [18, 9, 6] code can be constructed
    from the [16, 11, 4] Reed-Muller code by adding 2 parity bits and selecting
    a subcode... This is getting complex.
    
    Alternative: Use the fact that 512 = 2^9, so we need a 9-dimensional linear
    [18, 9, 6] code. Let me construct its generator matrix.
    
    The [18, 9, 6] code is the extended quadratic residue code? No.
    
    Let me try a different known construction:
    The code C = { (v, v, w) : v in F_2^6, w in F_2^6 } with some structure?
    That would give 2^12 = 4096 words, too many.
    
    Let me use the following known fact: A_2(18, 6) = 512 is achieved by a 
    nonlinear code. One construction is based on the Hadamard matrix of order 16.
    
    Actually, let me just implement a greedy/exhaustive search approach that
    is guaranteed to find 512 words within the time limit, or use a known 
    algebraic construction.
    
    Here's a cleaner approach: The [18, 9, 6] linear code exists. I'll construct
    it using the parity-check matrix. The dual is a [18, 9, 6] code as well (self-dual? No, 18/2=9, so it could be self-dual if the generator matrix is orthogonal).
    
    Actually, let me use a well-known construction. The code can be built from
    the 4x4 Hadamard matrix. Let H be the 16x16 Hadamard matrix (Sylvester).
    
    Construction: For each pair (a, b) in F_2^4 x F_2^4, define a codeword of length 18.
    But 2^8 = 256, not 512.
    
    Let me try: For each (a, b, c) in F_2^4 x F_2^4 x F_2^1, define a codeword.
    That gives 2^9 = 512 words.
    
    The codeword for (a, b, c) is:
    w[i] = (a dot i) + (b dot f(i)) + c for i in some set of 18 positions?
    
    This is getting too speculative. Let me use a simpler, verifiable approach:
    
    I'll implement a greedy algorithm that adds codewords one by one, choosing
    the word that maximizes the minimum distance to existing codewords. Start
    with 0, then greedily add words. This may not reach 512, but let's try a
    smarter initialization.
    
    Actually, the safest approach: Use the known [18, 9, 6] linear code.
    I'll define its generator matrix explicitly.
    
    The [18, 9, 6] code has generator matrix G where the columns correspond to
    the 18 positions. One standard form:
    
    G = [I_9 | P]
    
    where P is a 9x9 matrix chosen so that the resulting code has minimum distance 6.
    
    Let me use the construction from the extended quadratic residue code of length 20,
    shortened by 2. The [20, 10, 8] extended QR code (p=19) shortened by 2 gives
    a [18, 8, 6] code with 256 words. To get 512, I need a different approach.
    
    Let me just do a greedy search with good heuristics, and if it finds 512, great.
    Otherwise, I'll fall back to a smaller code.
    
    Actually, let me use the following known construction for A_2(18,6)=512:
    
    The code is the set of all vectors (x, x, y) where x in F_2^6 and y in F_2^6,
    but that's 2^12. No.
    
    I'll use a brute-force greedy approach with random restarts, but that's risky.
    
    Let me use the simplest correct approach: construct the [18, 9, 6] linear code
    using a known generator matrix.
    
    Here's a generator matrix for the [18, 9, 6] code (one possible form):
    
    The code is equivalent to the [18, 9, 6] code from the extended ternary Golay
    code construction... 
    
    I'll use the following: The [18, 9, 6] code has generator matrix:
    
    Row 0: 1 0 0 0 0 0 0 0 0 | 1 1 1 1 1 1 1 1
    Row 1: 0 1 0 0 0 0 0 0 0 | 1 1 1 1 1 0 0 0
    ... etc.
    
    This is too error-prone to do by hand. Let me use a different strategy:
    
    I'll use the fact that the code can be constructed as follows:
    Take the first-order Reed-Muller code RM(1, 4) of length 16, which has 32 words
    and distance 8. Then extend by 2 bits in a clever way to get distance 6 with
    512 words? No, that doesn't work directly.
    
    Let me just implement a greedy algorithm that is fast and see how many words
    it can find. If it finds 512, we're done.
    
    Greedy algorithm:
    1. Start with word 0.
    2. For each candidate word from 1 to 2^18 - 1, check if it has distance >= 6
       from all existing codewords. If so, add it.
    3. This is O(n * 2^18) which is too slow for n=512.
    
    Better: Use a bitset-based approach or precompute distances.
    
    Actually, 2^18 = 262144. For each of the 262144 candidates, checking against
    up to 512 existing codewords is 262144 * 512 = ~134 million operations, which
    is feasible in Python with optimization? Probably not in 60 seconds.
    
    Let me use a smarter approach: Use the fact that we can generate candidates
    in a structured way.
    
    I'll use the following: Start with the 32 RM(1,4) words extended with two zeros.
    Then try to add more words greedily.
    
    Actually, let me just return the 32-word Reed-Muller code as a fallback, and
    try to improve it with a time-limited greedy search.
    
    Given the complexity, let me use a simple but effective approach:
    """
    import random
    
    N = 18
    MIN_DIST = 6
    
    def hamming_dist(a, b):
        return (a ^ b).bit_count()
    
    # Start with a good initial set
    words = [0]
    
    # Add the Reed-Muller words as a starting point
    rm_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_words.append(word)
    
    # Try to add RM words
    for w in rm_words:
        if w == 0:
            continue
        ok = True
        for existing in words:
            if hamming_dist(w, existing) < MIN_DIST:
                ok = False
                break
        if ok:
            words.append(w)
    
    # Now greedily add more words
    # Use a randomized approach with multiple passes
    random.seed(42)
    candidates = list(range(1, 2**N))
    random.shuffle(candidates)
    
    added = True
    while added and len(words) < 512:
        added = False
        for cand in candidates:
            if len(words) >= 512:
                break
            ok = True
            for existing in words:
                if hamming_dist(cand, existing) < MIN_DIST:
                    ok = False
                    break
            if ok:
                words.append(cand)
                added = True
    
    return words
# EVOLVE-BLOCK-END
