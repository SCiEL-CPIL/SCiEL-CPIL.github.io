"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute index mapping and conflict masks
# A triple t=(a,b,c) conflicts with any other triple t' that, together with t,
# forms a K(2,2,2). Specifically, t' must share exactly 2 vertices with t,
# and the 6 vertices {a,b,c} ∪ {t'} must form a 2x2x2 grid where all 8
# transversal triples exist.

def _build_conflict_masks():
    """For each triple index, compute a bitmask of all triples that conflict with it."""
    idx_map = {t: i for i, t in enumerate(ALL_TRIPLES)}
    conflict_masks = [0] * N_TRIPLES
    
    for i, t in enumerate(ALL_TRIPLES):
        a, b, c = t
        used = {a, b, c}
        mask = 0
        for j, t2 in enumerate(ALL_TRIPLES):
            if j == i:
                continue
            # t2 must share exactly 2 vertices with t
            overlap = used & set(t2)
            if len(overlap) != 2:
                continue
            # The 6 vertices are a,b,c + the one new vertex in t2
            # Check if all 8 transversal triples of the 2x2x2 grid exist
            # The grid is defined by 3 pairs of vertices
            # t = {a,b,c}, t2 = {p,q,r} where {p,q,r} shares 2 with {a,b,c}
            # The 6 vertices form 3 pairs (the partition of the 2x2x2 grid)
            # We need to identify which 2 vertices are shared and which is new
            new_v = (set(t2) - used)
            if len(new_v) != 1:
                continue
            new_v = new_v.pop()
            # shared = used & set(t2), which has 2 elements
            shared = overlap  # set of 2 vertices
            
            # The 6 vertices: a, b, c, new_v (plus the 2 shared are among a,b,c)
            # Actually, the 6 vertices are {a,b,c,new_v} only if shared has 2 from {a,b,c}
            # Wait: t has 3 vertices, t2 has 3 vertices, they share 2, so union has 4 vertices.
            # That's not 6 vertices. Let me reconsider.
            
            # K(2,2,2) has 6 vertices: x1,x2,y1,y2,z1,z2
            # The 8 triples are all combinations of one from each pair.
            # If t=(x1,y1,z1) and t2 shares 2 vertices with t, say t2=(x1,y1,z2),
            # then the 6 vertices are x1,x2,y1,y2,z1,z2 where x2 and y2 are the
            # "other" vertices in the grid.
            
            # For t and t2 to be part of the same K(2,2,2), we need to find
            # x2, y2, z2 such that all 8 transversals exist.
            # t = (a,b,c), t2 shares 2 with t.
            
            # Let's say t2 = (a, b, d) where d is new (shares a,b with t).
            # Then the grid has pairs: {a,?}, {b,?}, {c,d}
            # We need a' such that (a',b,c), (a',b,d), (a',?,c), (a',?,d) exist...
            # This is getting complex. Let me just check all 8 transversals directly.
            
            # The 6 vertices are the union of t and t2, which is 4 vertices (since they share 2).
            # Wait, that can't form a K(2,2,2) which needs 6 vertices.
            
            # I think I'm confusing myself. Let me re-read the problem.
            # K(2,2,2) is the complete 3-partite 3-graph with parts of size 2.
            # It has 6 vertices: x1,x2 | y1,y2 | z1,z2
            # And 8 edges: all (xi,yj,zk) for i,j,k in {1,2}
            
            # If I add triple t=(a,b,c) to an existing set, I need to check if there
            # exist a',b',c' such that all 8 of:
            # (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
            # are in the set.
            
            # So t' = (a',b',c') is NOT a single triple that conflicts with t.
            # Rather, the conflict is with a SET of triples.
            
            # My original approach was wrong. The conflict check needs to verify
            # that there's no a',b',c' such that all 8 transversals exist.
            
            # Let me go back to the original approach but optimize it.
            pass
        
        conflict_masks[i] = mask
    
    return conflict_masks

# Actually, let me use the original approach but with bitmasks for the 7 other triples.
# For triple t=(a,b,c), adding it creates a K(2,2,2) if there exist a',b',c'
# (distinct from a,b,c and each other) such that all of these 7 triples are present:
# (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')

def _precompute_conflicts():
    """For each triple, precompute a list of (a',b',c') candidates that would
    complete a K(2,2,2) if the 7 corresponding transversals are all present."""
    idx_map = {t: i for i, t in enumerate(ALL_TRIPLES)}
    candidates = [[] for _ in range(N_TRIPLES)]
    
    for i, t in enumerate(ALL_TRIPLES):
        a, b, c = t
        used = {a, b, c}
        cands = []
        for ap in range(V):
            if ap in used:
                continue
            for bp in range(V):
                if bp in used or bp == ap:
                    continue
                for cp in range(V):
                    if cp in used or cp == ap or cp == bp:
                        continue
                    # The 7 other transversals:
                    t1 = tuple(sorted((a, b, cp)))
                    t2 = tuple(sorted((a, bp, c)))
                    t3 = tuple(sorted((a, bp, cp)))
                    t4 = tuple(sorted((ap, b, c)))
                    t5 = tuple(sorted((ap, b, cp)))
                    t6 = tuple(sorted((ap, bp, c)))
                    t7 = tuple(sorted((ap, bp, cp)))
                    # Store the indices of these 7 triples
                    try:
                        cands.append((idx_map[t1], idx_map[t2], idx_map[t3],
                                     idx_map[t4], idx_map[t5], idx_map[t6], idx_map[t7]))
                    except KeyError:
                        pass
        candidates[i] = cands
    
    return idx_map, candidates

_idx_map, _candidates = _precompute_conflicts()

def _is_valid_add_bitmask(trips_mask, t_idx):
    """Check if adding triple t_idx to trips_mask creates a K(2,2,2).
    trips_mask is a Python int where bit i is set if triple i is present."""
    for cand in _candidates[t_idx]:
        # Check if all 7 indices are set in trips_mask
        all_present = True
        for idx in cand:
            if not (trips_mask >> idx) & 1:
                all_present = False
                break
        if all_present:
            return False
    return True

def _greedy(order, deadline):
    """Greedily add triples in given order using bitmask conflict check."""
    trips_mask = 0
    result = []
    for t in order:
        if time.time() >= deadline:
            break
        t_idx = _idx_map[t]
        if (trips_mask >> t_idx) & 1:
            continue
        if _is_valid_add_bitmask(trips_mask, t_idx):
            trips_mask |= (1 << t_idx)
            result.append(list(t))
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best = None
    best_len = -1
    
    # Base: all triples through vertex 0 (105 triples)
    base = [tuple([0, a, b]) for a in range(1, V) for b in range(a + 1, V)]
    base_set = set(base)
    
    # Other triples
    other_triples = [t for t in ALL_TRIPLES if 0 not in t]
    
    # Try many random restarts (enabled by fast bitmask check)
    for attempt in range(2000):
        if time.time() >= deadline:
            break
        
        # Random order: base triples first, then others
        shuffled_base = base[:]
        random.shuffle(shuffled_base)
        shuffled_other = other_triples[:]
        random.shuffle(shuffled_other)
        
        order = shuffled_base + shuffled_other
        
        result = _greedy(order, deadline)
        
        if len(result) > best_len:
            best_len = len(result)
            best = result
            if time.time() >= deadline - 3:
                break
    
    if best is None:
        return [list(t) for t in base]
    
    # Final validation
    best_set = set(tuple(sorted(t)) for t in best)
    valid = True
    for x1, x2, y1, y2, z1, z2 in combinations(range(V), 6):
        t1 = (x1, y1, z1)
        t2 = (x1, y1, z2)
        t3 = (x1, y2, z1)
        t4 = (x1, y2, z2)
        t5 = (x2, y1, z1)
        t6 = (x2, y1, z2)
        t7 = (x2, y2, z1)
        t8 = (x2, y2, z2)
        
        if all(tuple(sorted(t)) in best_set for t in [t1, t2, t3, t4, t5, t6, t7, t8]):
            valid = False
            break
    
    if not valid:
        return [list(t) for t in base]
    
    return best
