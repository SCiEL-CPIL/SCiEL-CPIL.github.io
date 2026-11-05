import time
from math import gcd
import random

def construct():
    deadline = time.time() + 85.0
    n = 128
    nums = list(range(1, n + 1))
    
    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Identify bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, n + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    bad_triples.append((a, b, c))
                    
    # Precompute conflict structure
    # conflict[v][a] is bitmask of b such that {a, b, v} is bad
    conflict = [[0] * (n + 1) for _ in range(n + 1)]
    for a, b, c in bad_triples:
        conflict[a][b] |= (1 << c)
        conflict[a][c] |= (1 << b)
        conflict[b][a] |= (1 << c)
        conflict[b][c] |= (1 << a)
        conflict[c][a] |= (1 << b)
        conflict[c][b] |= (1 << a)
        
    # Precompute static degree for heuristic
    static_degree = [0] * (n + 1)
    for a, b, c in bad_triples:
        static_degree[a] += 2
        static_degree[b] += 2
        static_degree[c] += 2

    def is_compatible(v, S_mask):
        m = S_mask
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            if conflict[v][a] & S_mask:
                return False
            m ^= lsb
        return True

    def remove_incompatible(C_mask, S_mask):
        new_C = 0
        m = C_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if is_compatible(v, S_mask):
                new_C |= lsb
            m ^= lsb
        return new_C

    def popcount(x):
        return bin(x).count("1")

    best_size = 0
    best_set_mask = 0
    
    # Greedy initialization
    for _ in range(200):
        order = nums[:]
        random.shuffle(order)
        S_greedy = 0
        C_greedy = (1 << (n + 1)) - 2
        for v in order:
            if (1 << v) & C_greedy and is_compatible(v, S_greedy):
                S_greedy |= (1 << v)
                C_greedy &= ~(1 << v)
                C_greedy = remove_incompatible(C_greedy, S_greedy)
        sz = popcount(S_greedy)
        if sz > best_size:
            best_size = sz
            best_set_mask = S_greedy

    def bnb(S_mask, C_mask, size_S):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline:
            return
            
        ub = size_S + popcount(C_mask)
        if ub <= best_size:
            return
            
        if C_mask == 0:
            if size_S > best_size:
                best_size = size_S
                best_set_mask = S_mask
            return
            
        # Pick v with highest static degree in C
        m = C_mask
        best_v = -1
        best_deg = -1
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if static_degree[v] > best_deg:
                best_deg = static_degree[v]
                best_v = v
            m ^= lsb
            
        v = best_v
        
        # Branch 1: Include v
        if is_compatible(v, S_mask):
            new_S = S_mask | (1 << v)
            new_C = C_mask & ~(1 << v)
            new_C = remove_incompatible(new_C, new_S)
            bnb(new_S, new_C, size_S + 1)
            
        # Branch 2: Exclude v
        new_C = C_mask & ~(1 << v)
        bnb(S_mask, new_C, size_S)

    full_mask = (1 << (n + 1)) - 2
    bnb(0, full_mask, 0)
    
    result = []
    m = best_set_mask
    while m:
        lsb = m & -m
        result.append(lsb.bit_length() - 1)
        m ^= lsb
        
    if not result:
        return [1]
        
    return result
