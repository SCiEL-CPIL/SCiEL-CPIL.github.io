import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
DEADLINE = 110.0  # Safety margin before 120s limit

def construct():
    start_time = time.time()
    best_triples = []
    best_score = -1
    
    # Pre-compute all triples as frozensets for faster lookup? 
    # Actually, for checking K(2,2,2), we need to check if a specific set of 8 triples exists.
    # A K(2,2,2) is defined by 3 disjoint pairs (x1,x2), (y1,y2), (z1,z2).
    # The 8 transversal triples are all combinations (xi, yj, zk) for i,j,k in {1,2}.
    # To check if adding a triple t creates a K(2,2,2), we need to see if there exist 
    # 3 other pairs such that all 8 transversals are present (including t).
    
    # Efficient check for adding a triple t = {a,b,c}:
    # If we add t, a new K(2,2,2) is formed if there exist disjoint pairs (a', a''), (b', b''), (c', c'')
    # such that all 8 transversals are in the current set S union {t}.
    # Since t is new, the other 7 must be in S.
    # The pairs must involve a, b, c in some way? No, the pairs are disjoint from each other, 
    # but the vertices of the pairs can be any 6 distinct vertices.
    # Wait, the definition of K(2,2,2) in 3-graphs usually refers to the complete 3-partite 3-graph 
    # with parts of size 2. Let the parts be X={x1,x2}, Y={y1,y2}, Z={z1,z2}.
    # The edges are all {xi, yj, zk}.
    # So if we add t={a,b,c}, it could be one of the 8 edges.
    # The other 7 edges must be in S.
    # This implies that the 6 vertices involved in the K(2,2,2) must include a, b, c.
    # Let the 6 vertices be V_k222. Since t={a,b,c} is an edge, {a,b,c} subset V_k222.
    # The other 3 vertices in V_k222 \ {a,b,c} must form the "complement" parts.
    # Actually, it's easier to just iterate over all possible K(2,2,2) structures that contain t.
    # How many K(2,2,2)s contain a specific triple t={a,b,c}?
    # The 6 vertices of the K(2,2,2) must include a,b,c. Let the other 3 be d,e,f.
    # The partition into 3 pairs is fixed up to permutation of parts.
    # If a,b,c are in different parts, say a in X, b in Y, c in Z.
    # Then X={a,x}, Y={b,y}, Z={c,z}.
    # The 8 edges are:
    # {a,b,c} (t)
    # {a,b,z}, {a,y,c}, {a,y,z}
    # {x,b,c}, {x,b,z}, {x,y,c}, {x,y,z}
    # So if we add t, we need to check if there exist x,y,z (distinct from a,b,c and each other)
    # such that the other 7 triples are in S.
    # Also, a,b,c could be in the same part? No, a triple has 3 distinct vertices, so they must be in different parts.
    # So for each triple t={a,b,c}, we need to check all choices of x,y,z from V \ {a,b,c}.
    # There are 13*12*11 = 1716 such choices.
    # For each choice, we check if 7 specific triples are in S.
    # This is fast enough for a greedy approach if S is stored as a set of frozensets.

    S = set()
    
    def is_valid_add(t):
        a, b, c = t
        # Check all possible completions to a K(2,2,2)
        # We need to find x, y, z such that {a,b,c} is one edge, and the other 7 are in S.
        # The other 3 vertices x, y, z must be distinct from a,b,c and each other.
        # The structure is:
        # Part 1: {a, x}
        # Part 2: {b, y}
        # Part 3: {c, z}
        # Edges:
        # 1. {a,b,c} (t)
        # 2. {a,b,z}
        # 3. {a,y,c}
        # 4. {a,y,z}
        # 5. {x,b,c}
        # 6. {x,b,z}
        # 7. {x,y,c}
        # 8. {x,y,z}
        
        # Note: The assignment of a,b,c to parts 1,2,3 is fixed by t.
        # But x,y,z are the "partners" of a,b,c respectively.
        # We iterate over all distinct x,y,z from V \ {a,b,c}.
        
        # Optimization: Pre-compute the 7 required triples for a given (a,b,c,x,y,z)?
        # No, just check membership.
        
        for x in range(V):
            if x in (a,b,c): continue
            for y in range(V):
                if y in (a,b,c) or y == x: continue
                for z in range(V):
                    if z in (a,b,c) or z == x or z == y: continue
                    # Check the 7 other edges
                    if (frozenset((a,b,z)) in S and
                        frozenset((a,y,c)) in S and
                        frozenset((a,y,z)) in S and
                        frozenset((x,b,c)) in S and
                        frozenset((x,b,z)) in S and
                        frozenset((x,y,c)) in S and
                        frozenset((x,y,z)) in S):
                        return False
        return True

    # Greedy Randomized Construction
    # Try multiple random orders and pick the best
    # Or just one good randomized greedy?
    # Given time budget, let's do a few restarts.
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > DEADLINE:
            break
            
        random.seed(r) # Deterministic for reproducibility in evaluation? 
                       # Actually, random without seed is fine, but seed helps debugging.
                       # Let's use random without fixed seed for diversity across runs?
                       # The evaluator runs it once. So fixed seed per restart is good.
        
        S.clear()
        triples_list = ALL_TRIPLES[:]
        random.shuffle(triples_list)
        
        for t in triples_list:
            if time.time() > DEADLINE:
                break
            if is_valid_add(t):
                S.add(frozenset(t))
        
        # Local Search / Repair
        # Try to add more triples that were skipped?
        # Or just try to add any triple not in S that is valid.
        # Since we processed in random order, some triples might be addable now if we removed others?
        # No, we only added. So if it wasn't valid then, it's still not valid?
        # Wait, adding triples only makes it harder to add more.
        # So the set S is maximal w.r.t the order? No, not necessarily maximal in size.
        # But we can try to add triples that were NOT considered? No, we considered all.
        # However, if we skipped t because it conflicted with existing S, 
        # can we remove some existing triples to add t and more?
        # That's complex.
        
        # Simpler: Just check if any triple from ALL_TRIPLES not in S can be added.
        # This is O(N^2) if done naively.
        # Let's just keep the best S found so far.
        
        if len(S) > best_score:
            best_score = len(S)
            best_triples = [list(t) for t in S]
            # Update best_triples efficiently? 
            # We just copy.
            
        # If we have a very good score, maybe stop early?
        # Max possible is C(16,3) = 560.
        # Known bounds for ex_3(16, K(2,2,2))?
        # If we get close to 500, we can stop.
        if best_score > 400:
            break

    # Final check: ensure the returned list is valid and distinct
    # The problem asks for a list of distinct triples.
    # Our S is a set of frozensets, so they are distinct.
    # Convert to list of lists.
    
    # One last attempt to add more triples if time permits?
    # The greedy pass already tried all triples in a specific order.
    # If a triple was rejected, it's because it formed a K(2,2,2) with existing triples.
    # Since we only added triples, the condition "forms a K(2,2,2) with existing" remains true.
    # So no new triples can be added to S without removing some.
    # Thus, S is a maximal K(2,2,2)-free set (with respect to the addition order).
    
    return best_triples

# Run construct to test? No, just define it.
# The evaluator will call construct().

# Optimization: 
# The is_valid_add function is the bottleneck.
# 1716 iterations * 7 set lookups = ~12k operations per triple.
# 560 triples * 12k = 6.7M operations per greedy pass.
# 50 restarts * 6.7M = 335M operations.
# In Python, this might be too slow for 110s.
# Let's optimize is_valid_add.

# Better is_valid_add:
# Pre-compute for each triple t, the list of (x,y,z) combinations? No, that's memory heavy.
# Instead, note that the 7 other triples are specific.
# We can pre-compute the "conflict sets" or use a bitset?
# With 560 triples, we can use a bitset of 560 bits (9 integers of 64 bits or just Python ints).
# Let's try a faster check.

def construct():
    start_time = time.time()
    best_triples = []
    best_score = -1
    
    # Map triple to index for bitset operations
    triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}
    N_TRIPLES = len(ALL_TRIPLES)
    
    # For each triple t, pre-compute the list of "conflict groups".
    # A conflict group is a set of 7 triples (indices) that, if all present, 
    # would form a K(2,2,2) with t.
    # Actually, we just need to know if ANY such group is fully present.
    # Let's pre-compute for each t, a list of masks (bitsets) representing the 7 other edges.
    # If (current_mask & conflict_mask) == conflict_mask, then t cannot be added.
    
    conflict_masks = [None] * N_TRIPLES
    
    for i, t in enumerate(ALL_TRIPLES):
        a, b, c = t
        masks = []
        for x in range(V):
            if x in (a,b,c): continue
            for y in range(V):
                if y in (a,b,c) or y == x: continue
                for z in range(V):
                    if z in (a,b,c) or z == x or z == y: continue
                    # The 7 other edges
                    edges = [
                        (a,b,z),
                        (a,y,c),
                        (a,y,z),
                        (x,b,c),
                        (x,b,z),
                        (x,y,c),
                        (x,y,z)
                    ]
                    mask = 0
                    for e in edges:
                        idx = triple_to_idx[frozenset(e)]
                        mask |= (1 << idx)
                    masks.append(mask)
        conflict_masks[i] = masks

    def is_valid_add_idx(idx, current_mask):
        for mask in conflict_masks[idx]:
            if (current_mask & mask) == mask:
                return False
        return True

    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > DEADLINE:
            break
            
        random.seed(r)
        
        current_mask = 0
        triples_list = list(range(N_TRIPLES))
        random.shuffle(triples_list)
        
        for idx in triples_list:
            if time.time() > DEADLINE:
                break
            if is_valid_add_idx(idx, current_mask):
                current_mask |= (1 << idx)
        
        score = bin(current_mask).count("1")
        if score > best_score:
            best_score = score
            # Reconstruct the list of triples
            best_triples = [list(ALL_TRIPLES[i]) for i in range(N_TRIPLES) if (current_mask >> i) & 1]
            
        if best_score > 450: # High threshold to stop early
            break

    return best_triples

# Test locally?
# import time
# t0 = time.time()
# res = construct()
# print(len(res), time.time()-t0)
