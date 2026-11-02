"""Construct a kissing configuration in R^11 using integer and half-integer vectors."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def _build_integer_weight2():
    """All 11*10/2 * 4 = 220 vectors with two non-zero entries each +1 or -1."""
    vecs = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                vecs.append(v)
    return vecs


def _build_e8_half_integers():
    """The 128 minimal half-integer vectors of E8, embedded in the first 8 coordinates.
    E8 minimal vectors: (x1,...,x8) where all xi are integers with sum even, or
    all xi are half-integers (xi = (2mi+1)/2) with sum integer (sum of 8 half-integers
    is even when an even number are negative... actually sum of 8 half-integers with
    each being ±1/2 gives sum in {-4,-2,0,2,4}, which is even).
    The half-integer vectors of E8 have all coordinates ±1/2 with an even number of
    minus signs (so the sum is an integer, specifically 8/2 - 2k = 4 - 2k for k minuses,
    which is even for any k). Wait, sum = (8-2k)/2 = 4-k. For this to be an integer,
    it always is. The E8 lattice condition for the half-integer coset is that the
    number of minus signs is even. So k is even: k=0,2,4,6,8.
    Number: C(8,0)+C(8,2)+C(8,4)+C(8,6)+C(8,8) = 1+28+70+28+1 = 128.
    """
    vecs = []
    for k in range(0, 9, 2):  # even number of minus signs
        for indices in itertools.combinations(range(8), k):
            v = np.zeros(D, dtype=np.int64)
            # All 8 coordinates are ±1 (we'll scale by 1/2 later, but for now use integers)
            # Set coordinates 0..7 to ±1
            for idx in range(8):
                if idx in indices:
                    v[idx] = -2  # use -2 to represent -1/2 when we divide by 2
                else:
                    v[idx] = 2   # use 2 to represent +1/2 when we divide by 2
            # Coordinates 8,9,10 remain 0
            vecs.append(v)
    return vecs


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin before 120s limit

    # Build the two sets
    int_vecs = _build_integer_weight2()
    half_vecs = _build_e8_half_integers()

    # Combine: integer vectors are already integers, half-integer vectors need to be
    # scaled. But the problem requires integer vectors. So we need to work with
    # a common scale. Multiply all by 2 to make everything integer.
    # Integer weight-2 vectors scaled by 2: coordinates are 0, ±2. Norm^2 = 8, norm = 2√2.
    # Half-integer vectors scaled by 2: coordinates are 0, ±2. Norm^2 = 8*1 = 8 (wait,
    # 8 coords with value ±2, so norm^2 = 8*4 = 32). That's wrong.
    #
    # Let me reconsider. The problem says "integer vectors". So all vectors must have
    # integer coordinates. The constraint is: min pairwise distance >= max vector norm.
    #
    # Original seed: weight-2 vectors, norm = √2, min dist = √2. Works.
    #
    # For E8 half-integers embedded: if I use them as half-integers, they're not integer
    # vectors. I need to multiply by 2 to make them integers.
    #
    # Weight-2 int vectors: v has two ±1s. Norm = √2.
    # E8 half-int * 2: h has eight ±2s (in first 8 dims). Norm = √(8*4) = √32 = 4√2.
    #
    # Max norm = 4√2. Need all pairwise distances >= 4√2.
    # Dist between two weight-2 vectors: min is √2 < 4√2. FAIL.
    #
    # So I can't just scale everything by 2. I need all vectors to have the same norm
    # for the constraint to be tight.
    #
    # Alternative: use only integer vectors. The seed gives 220. Can I find more integer
    # vectors in R^11 with all norms equal and all pairwise distances >= that norm?
    #
    # Let's try weight-2 vectors (norm √2) and see if I can add any other integer vectors
    # with norm √2 that are at distance >= √2 from all 220.
    #
    # Integer vectors with norm √2: only weight-2 vectors (two ±1s, rest 0). There are
    # exactly 220 of them. So that's all.
    #
    # What about norm 2? Integer vectors with norm 2:
    # - Weight-1: one ±2, rest 0. Count: 11*2 = 22.
    # - Weight-2: one ±1, another ±1... no, norm would be √2.
    # - Actually: sum of squares = 4. Options: (4,0,...,0) i.e. one coord is ±2.
    #   Or (1,1,1,1,...) i.e. four ±1s. Count: C(11,4)*16 = 330*16 = 5280.
    #
    # Let's check if all 22 weight-1 (norm 2) vectors are compatible with each other
    # and with weight-4 (norm 2) vectors.
    #
    # This is getting complex. Let me just try to maximize the set of integer vectors
    # with a common norm where all pairwise distances >= that norm.
    #
    # Simplest approach: stick with the seed's 220 vectors. Try to add more by
    # considering a different norm.
    #
    # Actually, let me re-examine: can I use the 220 weight-2 vectors AND 22 weight-1
    # vectors (one ±2)?
    # Norm of weight-2: √2. Norm of weight-1 (±2): 2. Max norm = 2.
    # Need all pairwise distances >= 2.
    # Dist between two weight-2 vectors sharing one index: √2 < 2. FAIL.
    #
    # So mixing norms doesn't work easily. Let me just return the 220 vectors.
    #
    # Wait - let me try a completely different approach. Use vectors with norm √3
    # (weight-3: three ±1s). Count: C(11,3)*8 = 165*8 = 1320.
    # Min dist between two weight-3 vectors: they can share 0, 1, or 2 indices.
    # Share 2: dist = √(1+1) = √2 < √3. FAIL.
    #
    # So weight-3 vectors don't work either as a complete set.
    #
    # Let me just return the 220 vectors from the seed.
    all_vecs = int_vecs
    return np.array(all_vecs, dtype=np.int64)

# EVOLVE-BLOCK-END
