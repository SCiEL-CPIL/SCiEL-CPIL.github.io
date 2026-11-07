import time
import sys
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin: 85s out of 90s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    num_candidates = len(candidates)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Precompute prime factorization
    factors = [None] * num_candidates
    for i, v in enumerate(candidates):
        x = v
        f = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                f[d] = f.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            f[x] = f.get(x, 0) + 1
        factors[i] = f
    
    # Precompute: for each a, for each b, whether a | (b * c) for each c
    # This is a 3D table: div_table[a][b][c] = True if a | (b*c)
    # But 255^3 is ~16M entries, each a bool. That's 16MB, manageable.
    # Actually, let's precompute div_pair[a][b] = set of c such that a | (b*c)
    # Or better: precompute a 2D array: for each a, a bitset of pairs (b,c) that conflict.
    
    # Let's precompute: conflict[a][b] = bitset of c where a | (b*c) and c != a, c != b
    # Using Python integers as bitsets.
    
    # Actually, for speed, let's precompute:
    # For each a, a list of (b, c) pairs such that a | (b*c), b != c, b != a, c != a
    # But that's too many.
    
    # Better: for each a and b, precompute the set of c (as a bitmask) such that a | (b*c)
    # Then when checking if adding x to S creates a violation:
    #   For each pair (b, c) in S (b != c, b != x, c != x):
    #     check if x | (b*c) or b | (x*c) or c | (b*x)
    
    # Let's precompute a 2D list: divs[a][b] = bitmask of c values where a | (b*c)
    # c ranges over 0..254 (indices), so we need 255-bit integers.
    
    divs = [[0] * num_candidates for _ in range(num_candidates)]
    
    for a_idx in range(num_candidates):
        a_val = candidates[a_idx]
        fa = factors[a_idx]
        for b_idx in range(num_candidates):
            if b_idx == a_idx:
                continue
            b_val = candidates[b_idx]
            fb = factors[b_idx]
            mask = 0
            for c_idx in range(num_candidates):
                if c_idx == a_idx or c_idx == b_idx:
                    continue
                fc = factors[c_idx]
                # Check if a_val | (b_val * c_val)
                # i.e., for each prime p in fa, v_p(b) + v_p(c) >= v_p(a)
                ok = True
                for p, exp_a in fa.items():
                    exp_bc = fb.get(p, 0) + fc.get(p, 0)
                    if exp_a > exp_bc:
                        ok = False
                        break
                if ok:
                    mask |= (1 << c_idx)
            divs[a_idx][b_idx] = mask
    
    # Now, can_add(x, S) where S is a set of indices:
    # For each pair (b, c) in S with b != c:
    #   Check if x | (b*c): equivalent to c in divs[x][b] or b in divs[x][c]
    #   Check if b | (x*c): equivalent to c in divs[b][x]
    #   Check if c | (b*x): equivalent to x in divs[c][b]
    
    # To speed this up, represent S as a bitmask.
    
    def can_add_bitmask(x, S_mask):
        # S_mask is a bitmask of indices in S
        # Check all pairs (b, c) in S, b < c, b != x, c != x
        # For each such pair, check the three conditions.
        
        # Iterate over b in S
        b = 0
        while b < num_candidates:
            if S_mask & (1 << b):
                b += 1
                continue
            # b is in S
            if b != x:
                # For each c in S with c > b, c != x
                c_mask = S_mask & (~((1 << (b + 1)) - 1))  # bits above b
                c = b + 1
                while c < num_candidates:
                    if c_mask & (1 << c):
                        c += 1
                        continue
                    # c is in S, c > b
                    if c != x:
                        # Check x | (b*c): c in divs[x][b]
                        if divs[x][b] & (1 << c):
                            return False
                        # Check b | (x*c): c in divs[b][x]
                        if divs[b][x] & (1 << c):
                            return False
                        # Check c | (b*x): x in divs[c][b]
                        if divs[c][b] & (1 << x):
                            return False
                    c += 1
            b += 1
        return True
    
    # Hmm, this is still O(n^2) per check. Let me think of a faster way.
    
    # Alternative: for a given x, precompute the "forbidden pairs" - pairs (b,c) in S that would violate.
    # For x to be addable, S must not contain any forbidden pair for x, and also for each b in S, the pair (x, c) must not be forbidden for b, etc.
    
    # Let me define: for a given x, the set of pairs (b, c) that conflict with x being in S:
    #   Type 1: x | (b*c) -> pairs (b, c) where c in divs[x][b]
    #   Type 2: b | (x*c) for b in S -> for each b in S, c in divs[b][x]
    #   Type 3: c | (b*x) for c in S -> for each c in S, b in divs[c][x]... wait, this is symmetric to type 2.
    
    # Actually types 2 and 3 are the same condition: for each pair (b, c) in S, check if b | (x*c) or c | (b*x).
    # b | (x*c) means c in divs[b][x].
    # c | (b*x) means x in divs[c][b], i.e., b in divs[c][x]... no.
    
    # Let me re-derive:
    # For triple (x, b, c) with x, b, c distinct:
    #   x | (b*c): c in divs[x][b] (since divs[x][b] gives c where x | b*c)
    #   b | (x*c): c in divs[b][x]
    #   c | (x*b): x in divs[c][b], equivalently b in divs[c][x]? No.
    #   divs[c][b] is the set of d where c | (b*d). So x in divs[c][b] means c | (b*x).
    
    # So for a fixed x and a pair (b, c) in S:
    #   Violation if: (c in divs[x][b]) or (c in divs[b][x]) or (x in divs[c][b])
    
    # To check can_add(x, S):
    #   For each pair (b, c) in S, b != c, b != x, c != x:
    #     if (divs[x][b] & (1 << c)) or (divs[b][x] & (1 << c)) or (divs[c][b] & (1 << x)):
    #       return False
    #   return True
    
    # This is still O(|S|^2). But with bit operations, each check is fast.
    
    # Let me optimize by precomputing for each x:
    #   conflict_mask[x] = for each b, the set of c such that (c in divs[x][b]) or (c in divs[b][x]) or (x in divs[c][b])
    # But the third term depends on c, so it's not separable.
    
    # Actually, let's just do the O(n^2) check but with fast bit operations.
    # n is at most 255, so n^2 ~ 65000, and each check is a few bit ops. Should be fast enough.
    
    def can_add_fast(x, S_indices):
        """Check if x can be added to S_indices (list of indices)."""
        S_set = set(S_indices)
        n = len(S_indices)
        for i in range(n):
            b = S_indices[i]
            if b == x:
                continue
            for j in range(i + 1, n):
                c = S_indices[j]
                if c == x:
                    continue
                # Check three conditions
                if divs[x][b] & (1 << c):
                    return False
                if divs[b][x] & (1 << c):
                    return False
                if divs[c][b] & (1 << x):
                    return False
        return True
    
    # Even faster: use a bitmask for S
    def can_add_bitmask_fast(x, S_mask):
        """Check if x can be added, S represented as bitmask."""
        # Iterate over all b in S
        temp = S_mask
        while temp:
            # Get lowest set bit
            lsb = temp & (-temp)
            b = lsb.bit_length() - 1
            temp ^= lsb
            if b == x:
                continue
            # For each c in S with c != b, c != x
            temp2 = S_mask & ~(1 << b) & ~(1 << x)
            while temp2:
                lsb2 = temp2 & (-temp2)
                c = lsb2.bit_length() - 1
                temp2 ^= lsb2
                if divs[x][b] & (1 << c):
                    return False
                if divs[b][x] & (1 << c):
                    return False
                if divs[c][b] & (1 << x):
                    return False
        return True
    
    # Precompute a "risk" score: for each x, how many pairs (b,c) in the full candidate set conflict with x.
    # This helps with ordering.
    risk = [0] * num_candidates
    for x in range(num_candidates):
        count = 0
        for b in range(num_candidates):
            if b == x:
                continue
            # Number of c where conflict occurs
            mask = divs[x][b] & ~(1 << x) & ~(1 << b)
            count += bin(mask).count('1')
            mask2 = divs[b][x] & ~(1 << x) & ~(1 << b)
            # But we'd double count... let's just use the first term as a proxy
        risk[x] = count
    
    # Sort by risk ascending (safe first)
    sorted_by_risk = sorted(range(num_candidates), key=lambda i: risk[i])
    sorted_desc = sorted(range(num_candidates), key=lambda i: candidates[i], reverse=True)
    sorted_asc = sorted(range(num_candidates), key=lambda i: candidates[i])
    
    def greedy(order):
        S = []
        S_mask = 0
        for idx in order:
            if can_add_bitmask_fast(idx, S_mask):
                S.append(idx)
                S_mask |= (1 << idx)
        return S, S_mask
    
    def local_search(S, S_mask, max_time):
        not_in_S_mask = ((1 << num_candidates) - 1) & ~S_mask
        not_in_S_list = [i for i in range(num_candidates) if not (S_mask & (1 << i))]
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # Try additions
            random.shuffle(not_in_S_list)
            for idx in not_in_S_list:
                if can_add_bitmask_fast(idx, S_mask):
                    S.append(idx)
                    S_mask |= (1 << idx)
                    not_in_S_list.remove(idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try single swaps
            if S and not_in_S_list:
                for _ in range(100):
                    if time.time() >= max_time:
                        break
                    i = random.choice(S)
                    j = random.choice(not_in_S_list)
                    S_minus_i_mask = S_mask & ~(1 << i)
                    if can_add_bitmask_fast(j, S_minus_i_mask):
                        S.remove(i)
                        S.append(j)
                        S_mask = S_minus_i_mask | (1 << j)
                        not_in_S_list.remove(j)
                        not_in_S_list.append(i)
                        improved = True
                        break
        
        return S, S_mask
    
    best_S = []
    best_size = 0
    rng = random.Random(12345)
    
    # Multiple restarts with different orderings
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if r % 4 == 0:
            order = sorted_by_risk[:]
            k = min(30, len(order))
            rng.shuffle(order[:k])
        elif r % 4 == 1:
            order = sorted_desc[:]
            k = min(30, len(order))
            rng.shuffle(order[:k])
        elif r % 4 == 2:
            order = sorted_asc[:]
            k = min(30, len(order))
            rng.shuffle(order[:k])
        else:
            order = list(range(num_candidates))
            rng.shuffle(order)
        
        S, S_mask = greedy(order)
        
        time_left = deadline - time.time()
        search_time = max(0.5, time_left * 0.3)
        S, S_mask = local_search(S, S_mask, start_time + search_time)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
    
    # Final: try to improve best_S with more local search
    if best_S:
        S_mask = 0
        for idx in best_S:
            S_mask |= (1 << idx)
        time_left = deadline - time.time()
        if time_left > 2:
            best_S, S_mask = local_search(best_S, S_mask, deadline - 1)
    
    result = [candidates[i] for i in best_S]
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
