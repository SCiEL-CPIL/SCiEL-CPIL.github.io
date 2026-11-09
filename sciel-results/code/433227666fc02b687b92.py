"""Construct binary error-correcting code of length 18, distance 6 via Extended QR Code."""
import random
import numpy as np
from itertools import combinations

N = 18
MIN_DIST = 6
MAX_WORD = 1 << N

def popcount(x):
    return bin(x).count('1')

def hamming(x, y):
    return popcount(x ^ y)

def gf_mult(a, b, poly=0x11D):
    res = 0
    while b:
        if b & 1:
            res ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= poly
    return res

def gf_pow(a, n, poly=0x11D):
    res = 1
    while n:
        if n & 1:
            res = gf_mult(res, a, poly)
        a = gf_mult(a, a, poly)
        n >>= 1
    return res

def poly_mul_gf(p, q, poly=0x11D):
    res = [0] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        if pi == 0: continue
        for j, qj in enumerate(q):
            if qj == 0: continue
            res[i+j] ^= gf_mult(pi, qj, poly)
    return res

def construct():
    # 1. Find generator polynomial g(x) for [17, 9, 5] QR code
    # p = 17
    # beta is a primitive 17th root of unity in GF(2^8)
    # 2^15 has order 17 in GF(2^8) with poly 0x11D
    beta = gf_pow(2, 15, 0x11D)
    
    # Quadratic residues mod 17
    # 1, 2, 4, 8, 9, 13, 15, 16
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # g(x) = prod (x - beta^j) for j in Q
    g = [1]
    for j in Q:
        bj = gf_pow(beta, j, 0x11D)
        factor = [bj, 1] # x - bj = x + bj in GF(2)
        g = poly_mul_gf(g, factor, 0x11D)
        
    # g is a list of coefficients [g0, g1, ..., g8]
    # Convert to integer g_int (LSB first for shift operations)
    g_int = 0
    for i, c in enumerate(g):
        if c & 1:
            g_int |= (1 << i)
            
    # 2. Generate 512 codewords of length 17
    codes_17 = []
    for msg in range(512):
        c_bits = 0
        for i in range(9):
            if (msg >> i) & 1:
                c_bits ^= (g_int << i)
        codes_17.append(c_bits)
        
    # 3. Extend to 18 bits with parity
    codes_18 = []
    for c in codes_17:
        parity = popcount(c) & 1
        full_word = c | (parity << 17)
        codes_18.append(full_word)
        
    return codes_18
