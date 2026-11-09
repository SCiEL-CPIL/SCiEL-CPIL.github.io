"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    N = 512
    # Weights
    w = [0.0] * N
    for i in range(1, N):
        w[i] = 1.0 / (N - i)
        
    # Precompute conflict masks
    # conflict_mask[i] has bit j set if gcd(i, j) > 1
    conflict_mask = [0] * N
    for i in range(1, N):
        mask = 0
        for j in range(1, N):
            if i != j and gcd(i, j) > 1:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # Greedy Initialization
    S = []
    S_mask = 0
    for a in range(N - 1, 0, -1):
        # Check if a is coprime to all in S
        # Equivalent to: (conflict_mask[a] & S_mask) == 0
        if (conflict_mask[a] & S_mask) == 0:
            S.append(a)
            S_mask |= (1 << a)
            
    current_score = sum(w[a] for a in S)
    best_score = current_score
    best_S = list(S)
    
    # Helper to calculate score
    def calc_score(S_list):
        return sum(w[a] for a in S_list)

    # Local Search
    # We will alternate between Swap and 1-for-2 moves
    # To avoid infinite loops or getting stuck, we can limit iterations or use time.
    
    def get_largest_bit(mask):
        if mask == 0:
            return -1
        return mask.bit_length() - 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try Swap Moves
        # For each a in S, try to swap with best b
        S_copy = list(S)
        for idx_a, a in enumerate(S_copy):
            if time.time() > deadline: break
            
            # Calculate conflicts of S without a
            # C = OR of conflict_mask[x] for x in S, x != a
            C = 0
            for x in S:
                if x != a:
                    C |= conflict_mask[x]
            
            # Valid b must not be in S, and not in C
            # Also b must be coprime to all in S\{a}, which is ensured by b not in C
            # b must not be in S (so we remove S_mask)
            # Note: a is in S, so it is removed from valid candidates.
            valid_mask = ~(C | S_mask)
            # Mask to only consider bits 1..511
            valid_mask &= (1 << N) - 2  # Bits 1 to 511
            
            b = get_largest_bit(valid_mask)
            if b != -1:
                if w[b] > w[a]:
                    # Perform swap
                    S[idx_a] = b
                    S_mask = (S_mask ^ (1 << a)) | (1 << b)
                    current_score = calc_score(S)
                    if current_score > best_score:
                        best_score = current_score
                        best_S = list(S)
                    improved = True
                    break # Restart search from beginning after a change
        
        if improved:
            continue
            
        # 2. Try 1-for-2 Moves
        # For each a in S, try to replace with b, c
        S_copy = list(S)
        for idx_a, a in enumerate(S_copy):
            if time.time() > deadline: break
            
            C = 0
            for x in S:
                if x != a:
                    C |= conflict_mask[x]
            
            valid_mask = ~(C | S_mask)
            valid_mask &= (1 << N) - 2
            
            # Find largest b in valid_mask
            b = get_largest_bit(valid_mask)
            if b == -1:
                continue
                
            # Find largest c in valid_mask, c != b, and c coprime to b
            # c must be in valid_mask and not in conflict_mask[b]
            valid_c_mask = valid_mask & ~conflict_mask[b]
            # Remove b itself
            valid_c_mask &= ~(1 << b)
            
            c = get_largest_bit(valid_c_mask)
            if c != -1:
                if w[b] + w[c] > w[a]:
                    # Perform 1-for-2 swap
                    # Remove a, add b, c
                    S[idx_a] = b
                    S.append(c)
                    S_mask = (S_mask ^ (1 << a)) | (1 << b) | (1 << c)
                    current_score = calc_score(S)
                    if current_score > best_score:
                        best_score = current_score
                        best_S = list(S)
                    improved = True
                    break
        
    return best_S
# EVOLVE-BLOCK-END
