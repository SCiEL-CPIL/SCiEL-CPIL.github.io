import time
import math
import random

N = 128

# Precompute exclusion masks
# exclusion_masks[u][v] is a bitmask of w such that {u+1, v+1, w+1} is a forbidden triple
exclusion_masks = [[0] * N for _ in range(N)]

def precompute_masks():
    # Find all forbidden triples
    # a < b < c
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            # c must divide L_ab
            # and lcm(a,c) == L_ab and lcm(b,c) == L_ab
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    # Triple (a, b, c) is forbidden
                    ai, bi, ci = a - 1, b - 1, c - 1
                    exclusion_masks[ai][bi] |= (1 << ci)
                    exclusion_masks[bi][ai] |= (1 << ci)
                    exclusion_masks[ai][ci] |= (1 << bi)
                    exclusion_masks[ci][ai] |= (1 << bi)
                    exclusion_masks[bi][ci] |= (1 << ai)
                    exclusion_masks[ci][bi] |= (1 << ai)

precompute_masks()

def popcount(x):
    return bin(x).count('1')

def get_bits(x):
    bits = []
    i = 0
    while x:
        if x & 1:
            bits.append(i)
        x >>= 1
        i += 1
    return bits

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_size = 0
    best_set = 0  # Bitmask
    
    # Greedy to get a good initial solution
    # Try multiple random orders
    def greedy(order):
        S = 0
        excl = 0
        S_bits = []
        for x in order:
            if (S >> x) & 1: continue
            if (excl >> x) & 1: continue
            # It is safe to add x
            S |= (1 << x)
            S_bits.append(x)
            # Update excl
            for u in S_bits[:-1]:
                excl |= exclusion_masks[x][u]
        return S, excl

    best_greedy = 0
    # Natural order
    S, _ = greedy(range(N))
    if popcount(S) > popcount(best_greedy):
        best_greedy = S
    
    # Reverse order
    S, _ = greedy(range(N-1, -1, -1))
    if popcount(S) > popcount(best_greedy):
        best_greedy = S
        
    # Random orders
    for _ in range(100):
        if time.time() > deadline - 20:
            break
        order = list(range(N))
        random.shuffle(order)
        S, _ = greedy(order)
        if popcount(S) > popcount(best_greedy):
            best_greedy = S
            
    best_size = popcount(best_greedy)
    best_set = best_greedy

    # B&B with propagation
    # State: S (int), R (int), excl (int), S_bits (list)
    
    def bnb(S, R, excl, S_bits):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        # Propagate: R should not contain any element in excl
        R &= ~excl
        
        if R == 0:
            sz = len(S_bits)
            if sz > best_size:
                best_size = sz
                best_set = S
            return
            
        # Bound
        if len(S_bits) + popcount(R) <= best_size:
            return
            
        # Pick a variable to branch on
        # Heuristic: Pick the one with highest degree in remaining?
        # For now, pick lowest bit in R
        x = (R & -R).bit_length() - 1
        
        # Branch 1: Include x
        new_S = S | (1 << x)
        new_excl = excl
        new_S_bits = S_bits + [x]
        for u in S_bits:
            new_excl |= exclusion_masks[x][u]
        new_R = R & ~(1 << x)
        bnb(new_S, new_R, new_excl, new_S_bits)
        
        # Branch 2: Exclude x
        new_R = R & ~(1 << x)
        bnb(S, new_R, excl, S_bits)

    # Initialize B&B
    initial_R = (1 << N) - 1
    bnb(0, initial_R, 0, [])
    
    # Convert best_set bitmask to list
    result = []
    for i in range(N):
        if (best_set >> i) & 1:
            result.append(i + 1)
            
    if not result:
        result = [1]
        
    return result
