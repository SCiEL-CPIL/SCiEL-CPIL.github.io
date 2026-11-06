import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Keep a safety margin
    
    N = 256
    # Partition into S (small) and L (large)
    S = list(range(2, 129))  # 2 to 128
    L = list(range(129, 257)) # 129 to 256
    
    # Precompute reciprocal sums for L subsets
    # For each s in S, find L_s = {l in L | s divides l}
    L_s = {}
    C_s = {}
    for s in S:
        multiples = [l for l in L if l % s == 0]
        L_s[s] = set(multiples)
        C_s[s] = sum(1.0/l for l in multiples)
    
    # Base score: sum of reciprocals of all L
    base_score = sum(1.0/l for l in L)
    
    # Conflict graph for S: edge if lcm(a,b) <= 256
    # Since a,b <= 128, lcm(a,b) <= 256 is the condition.
    # Note: if a|b, lcm=b <= 128 <= 256. Conflict.
    # Precompute conflicts as a set of sets or adjacency list
    conflicts = {s: set() for s in S}
    for i, a in enumerate(S):
        for b in S[i+1:]:
            # lcm(a,b) = a*b/gcd(a,b)
            if (a * b) // gcd(a, b) <= 256:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    def is_valid_add(s, current_set):
        return all(s not in conflicts[c] for c in current_set)
    
    def calculate_gain(s, current_set, current_union_L):
        # current_union_L is the set of L elements already excluded by current_set
        # Gain = 1/s - sum(1/l for l in L_s[s] not in current_union_L)
        new_excluded = L_s[s] - current_union_L
        cost = sum(1.0/l for l in new_excluded)
        return 1.0/s - cost
    
    def total_score(S_prime, union_L):
        return base_score + sum(1.0/s for s in S_prime) - sum(1.0/l for l in union_L)
        
    # Greedy Initialization
    current_set = set()
    current_union_L = set()
    
    # Sort S by potential standalone gain (1/s - C_s) descending
    # Or just iterate by s? Small s have high 1/s but high C_s.
    # Let's try sorting by s (smallest first) as a heuristic, or by gain.
    # Let's try a simple greedy: pick s that gives max positive gain given current state.
    
    # To be efficient, we might just pick the best single element first, then iterate.
    # However, 127 elements is small. We can do a full greedy pass.
    
    # Strategy: 
    # 1. Start with empty.
    # 2. Repeatedly find s in S \ current_set that is valid and has max gain.
    # 3. Add it.
    # 4. If no positive gain, stop.
    
    # Optimization: Precompute 1/s
    inv_s = {s: 1.0/s for s in S}
    
    # Greedy Construction
    while True:
        best_s = None
        best_gain = 0.0
        for s in S:
            if s in current_set:
                continue
            if not is_valid_add(s, current_set):
                continue
            gain = calculate_gain(s, current_set, current_union_L)
            if gain > best_gain:
                best_gain = gain
                best_s = s
        
        if best_s is None:
            break
            
        current_set.add(best_s)
        current_union_L |= L_s[best_s]
        
        if time.time() - start_time > deadline:
            break

    # Local Search: Hill Climbing
    # Moves:
    # 1. Add a valid s
    # 2. Remove an s
    # 3. Swap: remove one, add one (or more complex swaps if time permits)
    
    current_score = total_score(current_set, current_union_L)
    
    def try_add(s):
        nonlocal current_set, current_union_L, current_score
        if s in current_set: return
        if not is_valid_add(s, current_set): return
        new_union = current_union_L | L_s[s]
        new_score = total_score(current_set | {s}, new_union)
        if new_score > current_score:
            current_set.add(s)
            current_union_L = new_union
            current_score = new_score
            return True
        return False

    def try_remove(s):
        nonlocal current_set, current_union_L, current_score
        if s not in current_set: return
        # Recalculate union_L after removal
        # This is expensive if done naively. 
        # Since |S| is small, we can just rebuild union_L.
        new_set = current_set - {s}
        new_union = set()
        for u in new_set:
            new_union |= L_s[u]
        new_score = total_score(new_set, new_union)
        if new_score > current_score:
            current_set = new_set
            current_union_L = new_union
            current_score = new_score
            return True
        return False

    # Iterative Local Search
    improved = True
    while improved and (time.time() - start_time < deadline):
        improved = False
        # Try adds
        for s in S:
            if try_add(s):
                improved = True
                if time.time() - start_time > deadline: break
        if time.time() - start_time > deadline: break
        
        # Try removes
        for s in list(current_set):
            if try_remove(s):
                improved = True
                if time.time() - start_time > deadline: break
        if time.time() - start_time > deadline: break

    # Final Check
    # The result is current_set U {l in L if l not in current_union_L}
    result = list(current_set)
    for l in L:
        if l not in current_union_L:
            result.append(l)
            
    return result

# EVOLVE-BLOCK-END
