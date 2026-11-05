"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin before 90s limit
    
    n = 256
    # Vertices are 1..n, indexed 0..n-1 in the bitset
    # weights[i] = 1/(i+1)
    weights = [1.0 / (i + 1) for i in range(n)]
    
    # Precompute conflict graph
    # Edge (i, j) exists if there exists k such that {i, j, k} is a 3-AP.
    # A 3-AP is (a, b, c) with a+c=2b.
    # If we pick a and c, b is determined as (a+c)/2.
    # If a+c is even and (a+c)/2 is in [1, n], then a and c cannot both be in the set.
    # Also, if we pick a and b, c is determined as 2b-a.
    # If 2b-a is in [1, n], then a and b cannot both be in the set.
    
    # Bitsets for neighbors
    # neighbors[i] is a set of indices j such that i and j conflict.
    # We use Python integers as bitsets for speed.
    neighbors = [0] * n
    
    # Generate conflicts
    # Case 1: i and j are ends of an AP. i < j, (i+j) is even, b=(i+j)/2 in [1, n].
    # Case 2: i and j are start and middle. i < j, c=2j-i in [1, n].
    # Case 3: i and j are middle and end. i < j, a=2i-j in [1, n].
    # Actually, it's easier to just iterate over all possible APs and add edges.
    
    # All 3-APs (a, b, c) with 1 <= a < b < c <= n
    # For each AP, add edges (a,b), (b,c), (a,c).
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            c = 2 * b - a
            if c > n:
                break
            # AP (a, b, c)
            # Add edge a-b
            ia, ib, ic = a - 1, b - 1, c - 1
            neighbors[ia] |= (1 << ib)
            neighbors[ib] |= (1 << ia)
            
            # Add edge b-c
            neighbors[ib] |= (1 << ic)
            neighbors[ic] |= (1 << ib)
            
            # Add edge a-c
            neighbors[ia] |= (1 << ic)
            neighbors[ic] |= (1 << ia)
            
    # Branch and Bound for Maximum Weight Independent Set
    # State: available vertices (bitset), current weight, current set (list)
    # We want to maximize sum of weights.
    
    best_weight = 0.0
    best_set = []
    
    # Upper bound estimation:
    # Sum of weights of all available vertices.
    # A tighter bound: sort available vertices by weight descending, 
    # take the top k where k is the max independent set size? No, size is not the only factor.
    # Simple bound: sum of weights of available vertices.
    # Better bound: For each vertex v in available, if we pick v, we lose its neighbors.
    # We can use a greedy coloring bound or just the sum of weights.
    # Given n=256, the search space is large, but the graph is sparse.
    # Let's use a recursive search with memoization? No, state space is too big.
    # We will use a depth-first search with pruning.
    
    # To make it efficient, we order vertices by weight (descending).
    # But the bitset operations are faster if we just iterate.
    
    # Let's use a stack-based or recursive DFS.
    # To avoid recursion depth issues, we'll use a recursive function but limit depth or use sys.setrecursionlimit.
    sys.setrecursionlimit(3000)
    
    # Precompute weight sum for any subset? No, too big.
    # We maintain current_weight.
    
    # To prune effectively, we need a good upper bound.
    # UB = current_weight + sum(weights[v] for v in available)
    # This is weak.
    # Better UB:
    # Let's try to find a maximal independent set greedily from 'available' to get a lower bound for the remaining?
    # No, we need an upper bound on the *additional* weight.
    
    # A simple but effective strategy for MWIS on sparse graphs:
    # 1. Pick a vertex v.
    # 2. Branch: Include v (remove v and neighbors) OR Exclude v (remove v).
    # 3. Prune if current_weight + sum(weights[available]) <= best_weight.
    
    # To improve the bound, we can compute the sum of weights of available vertices quickly.
    # However, summing every time is O(n).
    # We can maintain a running sum?
    # When we remove a set of vertices R, we subtract sum(weights[r] for r in R).
    
    # Let's implement a helper to sum weights of a bitset.
    # Since n=256, a bitset fits in 4 64-bit integers or just one Python int.
    # Iterating set bits is fast.
    
    def sum_weights(bitset):
        total = 0.0
        b = bitset
        while b:
            # Get lowest set bit
            lsb = b & -b
            idx = lsb.bit_length() - 1
            total += weights[idx]
            b ^= lsb
        return total
    
    # We need to be careful with floating point precision for the bound.
    # Using Fractions for the bound might be slower but exact.
    # Given the goal is to beat 0.55, and the optimum is likely around 0.6-0.7, 
    # float precision should be fine for pruning if we add a small epsilon.
    
    # Optimization: Precompute weights as Fractions for exactness in final score, 
    # but use floats for search speed.
    
    # Let's try to improve the bound using a greedy packing.
    # Or just use the sum of weights.
    
    # To speed up the search, we can order the vertices in the bitset by weight.
    # But bitset operations don't preserve order easily.
    # Instead, we can pick the vertex with the highest weight in 'available' to branch on.
    
    def get_max_weight_vertex(bitset):
        max_w = -1
        max_idx = -1
        b = bitset
        while b:
            lsb = b & -b
            idx = lsb.bit_length() - 1
            if weights[idx] > max_w:
                max_w = weights[idx]
                max_idx = idx
            b ^= lsb
        return max_idx
    
    # Recursive search
    def search(available, current_weight, current_set):
        nonlocal best_weight, best_set
        
        if time.time() - start_time > timeout:
            return
        
        if available == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = list(current_set)
            return
        
        # Pruning
        # Upper bound: current_weight + sum of weights of available
        # We can tighten this.
        # If we just add the sum, it's weak.
        # Let's check if current_weight + sum_weights(available) <= best_weight:
        # if so, prune.
        
        # To avoid calculating sum_weights(available) which is O(k), 
        # we can maintain a running sum?
        # But 'available' changes by removing a subset.
        
        # Let's calculate the bound.
        bound = current_weight + sum_weights(available)
        if bound <= best_weight + 1e-12:
            return
            
        # Pick a vertex to branch on.
        # Heuristic: Pick the vertex with the highest weight.
        v_idx = get_max_weight_vertex(available)
        
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        v_weight = weights[v_idx]
        
        # Branch 1: Include v
        # New available = available & ~(neighbors[v_idx] | v_bit)
        new_available_include = available & ~(neighbors[v_idx] | v_bit)
        current_set.append(v_idx + 1)
        search(new_available_include, current_weight + v_weight, current_set)
        current_set.pop()
        
        # Branch 2: Exclude v
        # New available = available & ~v_bit
        new_available_exclude = available & ~v_bit
        search(new_available_exclude, current_weight, current_set)
        
    # Initial available set is all vertices
    initial_available = (1 << n) - 1
    initial_set = []
    
    # To improve initial best_weight, run a greedy heuristic first.
    # Greedy: Pick highest weight vertex, remove neighbors, repeat.
    avail = initial_available
    greedy_set = []
    greedy_weight = 0.0
    while avail:
        v_idx = get_max_weight_vertex(avail)
        if v_idx == -1: break
        greedy_set.append(v_idx + 1)
        greedy_weight += weights[v_idx]
        avail &= ~(neighbors[v_idx] | (1 << v_idx))
        
    best_weight = greedy_weight
    best_set = list(greedy_set)
    
    # Now run B&B
    # To make B&B faster, we might want to sort vertices by weight in the initial call?
    # The get_max_weight_vertex function handles the ordering dynamically.
    
    # One issue: The search might be too slow if the bound is weak.
    # Let's add a random restart or a different heuristic if time allows?
    # No, let's stick to one deep search.
    
    # To speed up sum_weights and get_max_weight_vertex, we can precompute 
    # a list of (weight, index) sorted by weight.
    # Then in get_max_weight_vertex, we can iterate this list and check if bit is set.
    
    sorted_vertices = sorted(range(n), key=lambda i: -weights[i])
    
    def get_max_weight_vertex_fast(bitset):
        for idx in sorted_vertices:
            if bitset & (1 << idx):
                return idx
        return -1
        
    # And sum_weights can also use this? No, sum is commutative.
    # But iterating all set bits is fine.
    
    # Let's redefine search to use the fast getter.
    def search_fast(available, current_weight, current_set):
        nonlocal best_weight, best_set
        
        if time.time() - start_time > timeout:
            return
        
        if available == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = list(current_set)
            return
        
        # Pruning
        # We can use a tighter bound here.
        # Sum of weights of available.
        bound = current_weight + sum_weights(available)
        if bound <= best_weight + 1e-12:
            return
            
        v_idx = get_max_weight_vertex_fast(available)
        
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        v_weight = weights[v_idx]
        
        # Branch 1: Include v
        new_available_include = available & ~(neighbors[v_idx] | v_bit)
        current_set.append(v_idx + 1)
        search_fast(new_available_include, current_weight + v_weight, current_set)
        current_set.pop()
        
        # Branch 2: Exclude v
        new_available_exclude = available & ~v_bit
        search_fast(new_available_exclude, current_weight, current_set)
        
    # Run the search
    search_fast(initial_available, 0.0, [])
    
    # If time remains, we could try local search on the result to polish?
    # The B&B should find the global optimum if it completes.
    # If it times out, it returns the best found so far.
    
    # Verify the result is valid (just in case of bugs)
    S = set(best_set)
    valid = True
    for a in S:
        for b in S:
            if a >= b: continue
            c = 2*b - a
            if c in S:
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback to greedy
        S = set(greedy_set)
        
    return list(S)

# EVOLVE-BLOCK-END
