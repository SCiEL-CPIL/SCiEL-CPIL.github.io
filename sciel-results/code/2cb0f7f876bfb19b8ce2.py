import sys
from itertools import combinations
import random
import time

def construct():
    N = 192
    candidates = list(range(1, N + 1))
    
    def bit_count(bits):
        return bin(bits).count('1')
        
    def is_valid_addition(x, S):
        n = len(S)
        # Check 1: x divides sum of any 3 in S
        if n >= 3:
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # Check 2: Any y in S divides x + a + b
        # We need a, b from S \ {y}
        if n >= 3:
            for i in range(n):
                y = S[i]
                # Check pairs (j, k) where j, k != i
                for j in range(n):
                    if j == i:
                        continue
                    a = S[j]
                    for k in range(j + 1, n):
                        if k == i:
                            continue
                        b = S[k]
                        if (x + a + b) % y == 0:
                            return False
        return True

    # Initial Greedy to get a good lower bound
    def greedy_desc():
        S = []
        for x in range(N, 0, -1):
            if is_valid_addition(x, S):
                S.append(x)
        return S

    best_set = greedy_desc()
    best_size = len(best_set)
    
    # Try ascending greedy as well
    def greedy_asc():
        S = []
        for x in range(1, N + 1):
            if is_valid_addition(x, S):
                S.append(x)
        return S
    init2 = greedy_asc()
    if len(init2) > best_size:
        best_set = init2
        best_size = len(init2)

    def solve(current_set, remaining_bits):
        nonlocal best_set, best_size
        
        current_size = len(current_set)
        remaining_count = bit_count(remaining_bits)
        
        if current_size + remaining_count <= best_size:
            return
            
        if remaining_count == 0:
            if current_size > best_size:
                best_set = current_set[:]
                best_size = current_size
            return
            
        # Pick the largest remaining candidate
        # highest set bit
        idx = remaining_bits.bit_length() - 1
        x = idx + 1
        new_remaining = remaining_bits & ~(1 << idx)
        
        # Option 1: Include x
        if is_valid_addition(x, current_set):
            current_set.append(x)
            solve(current_set, new_remaining)
            current_set.pop()
            
        # Option 2: Exclude x
        solve(current_set, new_remaining)

    all_bits = 0
    for x in candidates:
        all_bits |= (1 << (x - 1))
        
    solve([], all_bits)
    
    return best_set
