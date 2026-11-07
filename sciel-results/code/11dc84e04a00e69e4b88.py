import math
import time

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _primes_upto(n):
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

def _parity_vectors(maxM):
    # returns list parity[i] for n in 211..maxM as int bitsets over primes<=maxM
    primes = _primes_upto(maxM)
    pindex = {p: i for i, p in enumerate(primes)}
    vecs = []
    for n in range(211, maxM+1):
        x = n
        v = 0
        for i, p in enumerate(primes):
            if p * p > x:
                break
            c = 0
            while x % p == 0:
                x //= p
                c ^= 1
            if c:
                v |= (1 << i)
        if x > 1:
            j = pindex[x]
            v |= (1 << j)
        vecs.append(v)
    return vecs, primes

def _target_parity(primes):
    t = 210
    v = 0
    for i, p in enumerate(primes):
        c = 0
        while t % p == 0:
            t //= p
            c ^= 1
        if c:
            v |= (1 << i)
    return v

def _span_combo(vectors, target, deadline):
    basis = {}
    for pos, v in enumerate(vectors):
        if time.time() > deadline:
            return None
        vec = v
        combo = 1 << pos
        while vec:
            pivot = vec.bit_length() - 1
            if pivot in basis:
                bvec, bcombo = basis[pivot]
                vec ^= bvec
                combo ^= bcombo
            else:
                basis[pivot] = (vec, combo)
                break
    vec = target
    combo = 0
    while vec:
        pivot = vec.bit_length() - 1
        if pivot in basis:
            bvec, bcombo = basis[pivot]
            vec ^= bvec
            combo ^= bcombo
        else:
            return None
    return combo

def _reduce_size(combo, vectors, target, deadline):
    # vectors indexed by position (0-based over 211..)
    # combo: bitset of positions currently selected, XOR = target
    # try to reduce to <=100 by removing zero-xor subsets
    while combo.bit_count() > 100 and time.time() < deadline:
        sel = [pos for pos in range(combo.bit_length()) if (combo >> pos) & 1]
        basis2 = {}
        zero_combo = None
        for pos in sel:
            vec = vectors[pos]
            combo2 = 1 << pos
            while vec:
                pivot = vec.bit_length() - 1
                if pivot in basis2:
                    bv, bc = basis2[pivot]
                    vec ^= bv
                    combo2 ^= bc
                else:
                    basis2[pivot] = (vec, combo2)
                    break
            if vec == 0 and combo2 != 0:
                zero_combo = combo2
                break
        if zero_combo is None or zero_combo == combo:
            break
        combo ^= zero_combo
    return combo

def _find_minimal_window(vecs, target, deadline):
    """Find the minimal M such that target is in span(vecs[0..M-211]).
    Uses binary search. Returns (M, combo) or (None, None)."""
    lo, hi = 1, len(vecs)
    ans_M = None
    ans_combo = None
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        sub = vecs[:mid]
        combo = _span_combo(sub, target, deadline)
        if combo is not None:
            ans_M = mid
            ans_combo = combo
            hi = mid - 1
        else:
            lo = mid + 1
    if ans_M is None:
        return None, None
    return ans_M, ans_combo

def construct():
    start = time.time()
    deadline = start + 85  # safety margin under 90s
    maxM = 10210
    vecs, primes = _parity_vectors(maxM)
    target = _target_parity(primes)
    
    # Find minimal window
    M, combo = _find_minimal_window(vecs, target, deadline)
    
    if M is None:
        # Fallback: try a known valid solution
        # 210 = 2*3*5*7. We need product * 210 to be a square.
        # Simple: pick 210*2=420? No, must be > 210. 
        # Actually 420 * 210 = 420^2 / 2... no.
        # 210 * x = square. x = 210 * k^2 for some k? 
        # 210 * 210 = 210^2. But 210 is not > 210.
        # Try x = 210 * 4 = 840. 840 * 210 = 840^2 / 4... no.
        # 210 * x = (2*3*5*7) * x. For this to be a square, x must have
        # odd parity for 2,3,5,7 and even for others.
        # Smallest x > 210 with this property: x = 2*3*5*7 * (some square)
        # x = 210 * 1^2 = 210 (not > 210)
        # x = 210 * 2^2 = 840. Check: 840 * 210 = 176400. isqrt = 420. 420^2 = 176400. Yes!
        # So [840] works. Raw value = 840 - 210 = 630.
        return [840]
    
    # M is the number of elements from 211..211+M-1 = 210+M
    # So the max element is 210 + M
    combo = _reduce_size(combo, vecs, target, deadline)
    selected = [211 + pos for pos in range(combo.bit_length()) if (combo >> pos) & 1]
    
    # Verify
    prod = 210
    for s in selected:
        prod *= s
    r = math.isqrt(prod)
    if (r * r != prod or len(selected) > 100 or len(set(selected)) != len(selected) 
            or any(s <= 210 or s > 10210 for s in selected) or len(selected) < 1):
        return [840]
    
    return list(selected)
