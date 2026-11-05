import time
import math
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Identify vertices that cannot be in the set (self-conflict: 2*v is square)
    forbidden = set()
    for v in vertices:
        if 2 * v in squares:
            forbidden.add(v)
            
    allowed = [v for v in vertices if v not in forbidden]
    
    # Strategy: Partition by residue mod 16.
    # Squares mod 16 are {0, 1, 4, 9}.
    # If a + b is a square, then (a mod 16) + (b mod 16) mod 16 must be in {0, 1, 4, 9}.
    # This creates a conflict graph on the 16 residues.
    # We want to select a subset of residues R such that for any r1, r2 in R (including r1=r2),
    # r1 + r2 is NOT a square mod 16.
    # Note: If r1 + r2 is a square mod 16, it might still be a non-square integer sum?
    # No. If a+b is a square integer, then (a+b) mod 16 is a square mod 16.
    # So if (r1+r2) mod 16 is NOT a square mod 16, then a+b is definitely NOT a square integer.
    # This means residues in a "conflict-free" set of residues can be combined freely.
    
    mod = 16
    residues = list(range(mod))
    
    # Build conflict graph on residues
    # Edge (r1, r2) if (r1 + r2) % mod is a quadratic residue (square) mod mod
    # Wait, squares mod 16: 0^2=0, 1^2=1, 2^2=4, 3^2=9, 4^2=0...
    # QR mod 16: {0, 1, 4, 9}
    qr_mod_16 = {0, 1, 4, 9}
    
    res_conflict = {r: set() for r in residues}
    for r1 in residues:
        for r2 in residues:
            if (r1 + r2) % mod in qr_mod_16:
                res_conflict[r1].add(r2)
                
    # We need to find a subset of residues S such that no two residues in S conflict.
    # This is an Independent Set on the residue conflict graph.
    # The residue graph is small (16 nodes). We can brute force all 2^16 subsets? 
    # 2^16 = 65536. Very fast.
    
    # However, even if residues are conflict-free mod 16, the actual integers a,b in those residues
    # might still sum to a square if the "carry" or higher bits align?
    # Let's check: a = 16k1 + r1, b = 16k2 + r2.
    # a + b = 16(k1+k2) + r1 + r2.
    # If (r1+r2) % 16 is NOT in QR mod 16, then a+b is NOT a square.
    # So if we pick a set of residues S that is an Independent Set in the residue conflict graph,
    # then ANY integers from those residue classes are pairwise sum-non-square.
    # This allows us to take ALL allowed vertices in those residue classes!
    
    # But wait, we can also include some vertices from conflicting residue classes if we prune them carefully.
    # Strategy:
    # 1. Find all Independent Sets of residues.
    # 2. For each IS, calculate the count of allowed vertices in those residues.
    # 3. Pick the best IS. This gives a large base set.
    # 4. The remaining vertices (in non-selected residues) can be added if they don't conflict with the base set or each other.
    #    Actually, if a residue r is NOT in S, it conflicts with some r' in S.
    #    So we can only add vertices from r if they don't conflict with specific vertices in S.
    
    # Let's find the best Independent Set of residues first.
    best_res_set = set()
    best_res_count = 0
    
    # Map residue to list of allowed vertices
    res_verts = {r: [] for r in residues}
    for v in allowed:
        res_verts[v % mod].append(v)
        
    # Brute force all subsets of residues
    # There are 16 residues.
    for mask in range(1 << mod):
        S = set()
        valid = True
        for r in residues:
            if mask & (1 << r):
                S.add(r)
        
        # Check if S is an independent set in res_conflict
        for r in S:
            if S & res_conflict[r]:
                valid = False
                break
        
        if valid:
            count = sum(len(res_verts[r]) for r in S)
            if count > best_res_count:
                best_res_count = count
                best_res_set = S
                
    # Initialize best_set with all allowed vertices in best_res_set
    best_set = set()
    for r in best_res_set:
        best_set.update(res_verts[r])
        
    # Now, try to add vertices from the remaining residues (not in best_res_set)
    # These residues conflict with best_res_set, so we can only add specific vertices.
    # Let's treat the remaining vertices as a pool.
    # We want to add as many as possible such that they don't conflict with best_set or each other.
    
    # To do this efficiently, we can build a graph of conflicts among the "candidate" vertices
    # and also mark which ones conflict with the current best_set.
    
    candidates = [v for v in allowed if v not in best_set]
    
    # Filter candidates: A candidate v can only be added if it doesn't conflict with ANY vertex in best_set.
    # If v + u is a square for some u in best_set, v is invalid.
    
    # Precompute neighbors in best_set for speed?
    # Or just check.
    
    valid_candidates = []
    for v in candidates:
        ok = True
        for u in best_set:
            if v + u in squares:
                ok = False
                break
        if ok:
            valid_candidates.append(v)
            
    # Now we have a set of valid_candidates.
    # We need to find a Max Independent Set among valid_candidates (considering edges between them).
    # The graph on valid_candidates might be small enough for greedy or simple local search.
    
    # Build adjacency for valid_candidates
    # Map value to index
    v_to_idx = {v: i for i, v in enumerate(valid_candidates)}
    n_cand = len(valid_candidates)
    
    # Adjacency list
    adj_cand = {v: set() for v in valid_candidates}
    
    for i in range(n_cand):
        u = valid_candidates[i]
        for j in range(i + 1, n_cand):
            v = valid_candidates[j]
            if u + v in squares:
                adj_cand[u].add(v)
                adj_cand[v].add(u)
                
    # Greedy MIS on valid_candidates
    # Sort by degree (ascending) to pick low-degree nodes first
    # This is a heuristic, but given the structure, it should work well.
    
    remaining = set(valid_candidates)
    added = set()
    
    # Pre-compute degrees for sorting
    # Dynamic degree is better but static is faster to start.
    # Let's do a simple greedy: repeatedly pick node with min degree in remaining.
    
    # To make it faster, we can use a heap or just scan. n_cand is likely small (< 100).
    
    while remaining and time.time() - start_time < DEADLINE:
        # Find min degree
        min_deg = float('inf')
        min_v = None
        for v in remaining:
            deg = len(adj_cand[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                min_v = v
        
        if min_v is None:
            break
            
        added.add(min_v)
        remaining.discard(min_v)
        for u in adj_cand[min_v]:
            if u in remaining:
                remaining.discard(u)
                
    best_set.update(added)
    
    # Local Search: Try to improve by swapping?
    # Given the time, let's try a simple "add if possible" pass again, maybe with randomization?
    # The greedy above was deterministic. Let's try a few random restarts on the candidate set?
    # Or just try to add any remaining valid candidates that might have been blocked by greedy choices?
    # Actually, the greedy removes neighbors. If we pick a different node, we might pick more.
    
    # Let's try a simple local search: 1-swap (remove one, add two?) is complex.
    # Let's try 0-1 swap: remove one from best_set, see if we can add more from candidates.
    
    # However, the base set from residues is very strong. The gain from local search on the small remainder is small.
    # Let's verify the result.
    
    best_list = list(best_set)
    valid = True
    for i in range(len(best_list)):
        a = best_list[i]
        for j in range(i, len(best_list)):
            b = best_list[j]
            if a + b in squares:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback
        return list(range(1, 257, 3))
        
    return best_list
