import itertools
import random
import time

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

# Precompute squared distances
# dist[i][j]
dist = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i+1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1-x2)**2 + (y1-y2)**2
        dist[i][j] = d
        dist[j][i] = d

# Precompute pair indices
# Map (u, v) with u < v to an index 0..119
pair_to_idx = {}
idx_to_pair = {}
idx = 0
for u in range(NUM_POINTS):
    for v in range(u+1, NUM_POINTS):
        pair_to_idx[(u, v)] = idx
        idx_to_pair[idx] = (u, v)
        idx += 1
NUM_PAIRS = idx

# Precompute conflict_bits[x]
# conflict_bits[x] is an integer with NUM_PAIRS bits.
# Bit i is 1 if point x and pair i form an isosceles triangle.
conflict_bits = [0] * NUM_POINTS
for x in range(NUM_POINTS):
    mask = 0
    for p_idx in range(NUM_PAIRS):
        u, v = idx_to_pair[p_idx]
        # Check if x, u, v form isosceles triangle
        # Sides: xu, xv, uv
        du = dist[x][u]
        dv = dist[x][v]
        uv = dist[u][v]
        # Isosceles if any two sides are equal
        if du == dv or du == uv or dv == uv:
            mask |= (1 << p_idx)
    conflict_bits[x] = mask

def get_pairs_bits(set_indices):
    """Compute the bitset of pairs in the set."""
    mask = 0
    m = len(set_indices)
    for i in range(m):
        u = set_indices[i]
        for j in range(i+1, m):
            v = set_indices[j]
            if u > v:
                u, v = v, u
            p_idx = pair_to_idx[(u, v)]
            mask |= (1 << p_idx)
    return mask

def can_add(p_idx, set_indices, set_pairs_bits=None):
    if set_pairs_bits is None:
        set_pairs_bits = get_pairs_bits(set_indices)
    return (conflict_bits[p_idx] & set_pairs_bits) == 0

def is_maximal(set_indices, set_pairs_bits=None):
    if set_pairs_bits is None:
        set_pairs_bits = get_pairs_bits(set_indices)
    set_set = set(set_indices)
    for p in range(NUM_POINTS):
        if p not in set_set:
            if can_add(p, set_indices, set_pairs_bits):
                return False
    return True

def greedy_construction(order):
    chosen = []
    chosen_bits = 0
    for p in order:
        if can_add(p, chosen, chosen_bits):
            chosen.append(p)
            # Update chosen_bits
            # Find new pairs involving p
            for q in chosen[:-1]:
                u, v = p, q
                if u > v: u, v = v, u
                p_idx = pair_to_idx[(u, v)]
                chosen_bits |= (1 << p_idx)
    return chosen

def local_search(set_indices):
    current = list(set_indices)
    current_bits = get_pairs_bits(current)
    improved = True
    while improved:
        improved = False
        
        # 1-moves: Try removing a point
        # If we remove p, is the set still maximal?
        # If yes, we reduce size by 1.
        for i in range(len(current)):
            p = current[i]
            new_set = current[:i] + current[i+1:]
            if is_maximal(new_set):
                current = new_set
                current_bits = get_pairs_bits(current)
                improved = True
                break
        
        if improved:
            continue
            
        # 2-moves: Try swapping a point p in current with q not in current
        # This keeps size same but might allow future removals or is just a different local minimum.
        # Actually, we want to minimize size. Swaps don't change size directly.
        # However, a swap might lead to a state where a removal is possible.
        # So we should try swaps if we are stuck.
        
        # To avoid infinite loops or excessive cost, we limit swaps or use them carefully.
        # Let's try a few random swaps if stuck.
        if len(current) > 0:
            current_set = set(current)
            candidates_out = [p for p in range(NUM_POINTS) if p not in current_set]
            
            # Try a limited number of random swaps
            num_swaps_to_try = min(20, len(candidates_out) * len(current))
            for _ in range(num_swaps_to_try):
                if not candidates_out: break
                p_in = random.choice(current)
                p_out = random.choice(candidates_out)
                
                new_set = [x for x in current if x != p_in]
                new_set.append(p_out)
                
                # Check independence
                new_bits = get_pairs_bits(new_set)
                # Check if p_out is compatible with rest of new_set
                # Actually get_pairs_bits just builds the mask. 
                # We need to check if p_out forms isosceles with any pair in new_set \ {p_out}.
                # The can_add check does this.
                # But we also need to check if the rest of the set is independent (it was before).
                # And we need to check maximality.
                
                # Efficient check:
                # 1. Independence: p_out must not conflict with any pair in new_set.
                #    Pairs in new_set are:
                #    - Pairs from old_set \ {p_in}
                #    - Pairs {p_out, x} for x in old_set \ {p_in}
                #    We already know old_set was independent.
                #    So we just need to check if p_out conflicts with any pair in old_set \ {p_in} 
                #    AND if p_out conflicts with any pair {p_out, x}? No, that's not how it works.
                #    The condition is: No triple in new_set is isosceles.
                #    Triples involving p_out: {p_out, x, y} where x, y in old_set \ {p_in}.
                #    So we need to check if p_out conflicts with any pair in old_set \ {p_in}.
                
                # Let's just use the general can_add on the reduced set, then check maximality.
                # This is slower but safer.
                reduced_set = [x for x in current if x != p_in]
                reduced_bits = get_pairs_bits(reduced_set)
                
                if can_add(p_out, reduced_set, reduced_bits):
                    # Independence holds for p_out with respect to reduced_set.
                    # Now check if new_set is maximal.
                    if is_maximal(new_set):
                        current = new_set
                        current_bits = get_pairs_bits(current)
                        improved = True
                        break
                        
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Phase 1: Randomized Greedy + Local Search
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        s = greedy_construction(order)
        s = local_search(s)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s
            
    if best_set is None:
        s = greedy_construction(list(range(NUM_POINTS)))
        best_set = s

    return [POINTS[i] for i in best_set]
