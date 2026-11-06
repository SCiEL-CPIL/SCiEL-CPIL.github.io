import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    n = 128
    # adj[i] is a bitmask of nodes conflicting with i
    adj = [0] * n
    
    # Precompute gcds or just use math.gcd
    # Iterate through all triples to find conflicts
    # This is O(N^3) ~ 2M operations, which is fast in Python if optimized.
    
    # Optimization: Only check triples where LCMs are equal.
    # We can iterate a, b, compute L_ab.
    # Then for c, check if L_ac == L_ab and L_bc == L_ab.
    
    for a in range(1, 129):
        for b in range(a + 1, 129):
            lab = math.lcm(a, b)
            for c in range(b + 1, 129):
                if time.time() > deadline:
                    # Fallback to greedy if precomputation takes too long (unlikely)
                    return _greedy_fallback()
                
                lac = math.lcm(a, c)
                if lac != lab:
                    continue
                lbc = math.lcm(b, c)
                if lbc == lab:
                    ia, ib, ic = a - 1, b - 1, c - 1
                    adj[ia] |= (1 << ib) | (1 << ic)
                    adj[ib] |= (1 << ia) | (1 << ic)
                    adj[ic] |= (1 << ia) | (1 << ib)
    
    # Solve Maximum Independent Set on the conflict graph
    # Using a recursive branch and bound with bitsets
    
    best_set = 0
    best_size = 0
    
    # Sort vertices by degree to improve pruning?
    # Bitset operations don't easily allow reordering without remapping.
    # We'll stick to natural ordering but use the popcount bound.
    
    def dfs(candidates, current_set, current_size):
        nonlocal best_set, best_size
        
        # Pruning
        if current_size + bin(candidates).count('1') <= best_size:
            return
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        # Check time
        if time.time() > deadline:
            return
        
        # Pick a vertex
        # Heuristic: Pick the vertex with the highest degree within candidates
        # For simplicity and speed, pick the lowest set bit
        # A better heuristic: pick v that maximizes the number of removed neighbors
        # Let's try picking the first one for now, N=128 is small.
        v_bit = candidates & -candidates
        v = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        # Remove v and its neighbors from candidates
        new_candidates = candidates & ~adj[v] & ~v_bit
        dfs(new_candidates, current_set | v_bit, current_size + 1)
        
        # Branch 2: Exclude v
        new_candidates = candidates & ~v_bit
        dfs(new_candidates, current_set, current_size)
        
    initial_candidates = (1 << n) - 1
    dfs(initial_candidates, 0, 0)
    
    # Convert bitset to list of elements
    result = []
    for i in range(n):
        if best_set & (1 << i):
            result.append(i + 1)
            
    if not result:
        return [1]
        
    return result

def _greedy_fallback():
    selected = []
    for x in range(1, 129):
        valid = True
        for a, b in combinations(selected, 2):
            if math.lcm(a, b) == math.lcm(a, x) == math.lcm(b, x):
                valid = False
                break
        if valid:
            selected.append(x)
    return selected
