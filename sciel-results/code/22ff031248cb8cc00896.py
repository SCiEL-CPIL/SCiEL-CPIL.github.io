"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 85.0  # 85s safety margin within 90s limit
    
    n = 128
    # 1. Precompute bad pairs for each vertex
    # bad_pairs[v] is a list of bitmasks. Each bitmask represents a pair {a, b}
    # such that gcd(a,b) == gcd(a,v) == gcd(b,v).
    bad_pairs = [[] for _ in range(n + 1)]
    
    # To speed up precomputation, we can iterate over triples?
    # N=128, N^3 is ~2M. Python loops might be slow but acceptable if optimized or done once.
    # 2M iterations in Python is ~0.1-0.5s. Acceptable.
    
    # However, we can be smarter.
    # For each v, and each a != v, and each b != v, a < b:
    # Check gcd(a,b) == gcd(a,v) == gcd(b,v).
    
    # Let's precompute gcd matrix first.
    gcd_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]
            
    for v in range(1, n + 1):
        pairs = []
        for a in range(1, n + 1):
            if a == v: continue
            g_av = gcd_mat[a][v]
            for b in range(a + 1, n + 1):
                if b == v: continue
                if gcd_mat[a][b] == g_av and gcd_mat[b][v] == g_av:
                    # Pair (a, b) conflicts with v
                    mask = (1 << a) | (1 << b)
                    pairs.append(mask)
        bad_pairs[v] = pairs
        
    # 2. Branch and Bound
    best_size = 0
    best_set_mask = 0
    
    def popcount(x):
        return x.bit_count()
        
    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb
            
    def solve(selected_mask, available_mask, current_size):
        nonlocal best_size, best_set_mask, deadline
        
        if time.time() > deadline:
            return
            
        upper_bound = current_size + popcount(available_mask)
        if upper_bound <= best_size:
            return
            
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = selected_mask
            return
            
        # Heuristic: Pick a vertex from available
        # Let's pick the one with the most bad pairs? Or just the first one?
        # Picking the first one (lowest index) is simple.
        # Let's try to pick the one that reduces the most?
        # For now, lowest index.
        v = (available_mask & -available_mask).bit_length() - 1
        
        # Branch 1: Exclude v
        new_available = available_mask & ~(1 << v)
        solve(selected_mask, new_available, current_size)
        
        # Branch 2: Include v
        # Check validity
        valid = True
        for p_mask in bad_pairs[v]:
            if (selected_mask & p_mask) == p_mask:
                valid = False
                break
                
        if valid:
            new_selected = selected_mask | (1 << v)
            new_available = available_mask & ~(1 << v)
            
            # Propagation:
            # If we include v, any u such that {u, v} is a bad pair with some selected w
            # must be excluded.
            # Iterate over selected w
            sel_bits = iter_bits(selected_mask)
            for w in sel_bits:
                for p_mask in bad_pairs[w]:
                    # Check if p_mask contains v
                    if (p_mask & (1 << v)) != 0:
                        # Find the other element u
                        # p_mask has exactly 2 bits set (a and b). One is v.
                        other = p_mask ^ (1 << v)
                        # If other is in new_available, remove it
                        if (new_available & other) != 0:
                            new_available &= ~other
                            
            # Also, we should check if v conflicts with any pair in selected?
            # We already did that in the validity check.
            
            solve(new_selected, new_available, current_size + 1)
            
    # Initial call
    initial_available = (1 << (n + 1)) - 2  # Bits 1..128 set
    solve(0, initial_available, 0)
    
    # Convert best_set_mask to list
    result = []
    for v in iter_bits(best_set_mask):
        result.append(v)
        
    return result
# EVOLVE-BLOCK-END
