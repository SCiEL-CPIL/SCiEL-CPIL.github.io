import time
import math
import random
import sys
from collections import defaultdict

N = 128
sys.setrecursionlimit(100000)

# Precompute conflicts: for each x, conflicts[x] is a set of (u,v) pairs
# such that {x, u, v} is a forbidden triple (all three pairwise LCMs equal).
conflicts = [set() for _ in range(N + 1)]

def precompute_conflicts():
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    conflicts[a].add((b, c))
                    conflicts[b].add((a, c))
                    conflicts[c].add((a, b))

precompute_conflicts()

# Precompute conflict_bitmasks
conflict_bitmasks = [[] for _ in range(N + 1)]
for x in range(1, N + 1):
    for u, v in conflicts[x]:
        # Use 0-indexed bits: element i -> bit (i-1)
        mask = (1 << (u - 1)) | (1 << (v - 1))
        conflict_bitmasks[x].append(mask)

def can_add_mask(x, cur_mask):
    """Check if x can be added to the set represented by cur_mask (bitmask)."""
    for cm in conflict_bitmasks[x]:
        if (cur_mask & cm) == cm:
            return False
    return True

def mask_to_set(cur_mask):
    """Convert bitmask to set of elements."""
    s = set()
    for i in range(1, N + 1):
        if (cur_mask >> (i - 1)) & 1:
            s.add(i)
    return s

# Global pair conflict matrix
pair_conflict = [[False] * (N + 1) for _ in range(N + 1)]

def precompute_pair_conflicts():
    for a in range(1, N + 1):
        for (b, c) in conflicts[a]:
            pair_conflict[a][b] = True
            pair_conflict[a][c] = True
            pair_conflict[b][c] = True
            pair_conflict[b][a] = True
            pair_conflict[c][a] = True
            pair_conflict[c][b] = True

precompute_pair_conflicts()

# Precompute adjacency masks for pair conflict graph
adj_mask = [0] * (N + 1)
for u in range(1, N + 1):
    mask = 0
    for v in range(1, N + 1):
        if pair_conflict[u][v]:
            mask |= (1 << (v - 1))
    adj_mask[u] = mask

def greedy_construct(deadline=None):
    """Greedy construction with multiple ordering strategies using bitmask."""
    best_mask = 0
    best_size = 0
    
    # Precompute degrees
    degrees = [0] * (N + 1)
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if pair_conflict[u][v]:
                degrees[u] += 1
                degrees[v] += 1
    
    # Strategy 1: Random orderings
    num_random = 200 if deadline is None else max(50, int(200 * min(1.0, (deadline - time.time()) / 10.0)))
    for _ in range(num_random):
        if deadline and time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        cur_mask = 0
        cur_size = 0
        for x in order:
            if can_add_mask(x, cur_mask):
                cur_mask |= (1 << (x - 1))
                cur_size += 1
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
    
    # Strategy 2: Sort by degree (low degree first)
    order_low = sorted(range(1, N + 1), key=lambda x: degrees[x])
    cur_mask = 0
    cur_size = 0
    for x in order_low:
        if can_add_mask(x, cur_mask):
            cur_mask |= (1 << (x - 1))
            cur_size += 1
    if cur_size > best_size:
        best_size = cur_size
        best_mask = cur_mask
    
    # Strategy 3: Sort by degree (high degree first)
    order_high = sorted(range(1, N + 1), key=lambda x: -degrees[x])
    cur_mask = 0
    cur_size = 0
    for x in order_high:
        if can_add_mask(x, cur_mask):
            cur_mask |= (1 << (x - 1))
            cur_size += 1
    if cur_size > best_size:
        best_size = cur_size
        best_mask = cur_mask
    
    # Strategy 4: Sort by value
    for order in [list(range(1, N + 1)), list(range(N, 0, -1))]:
        cur_mask = 0
        cur_size = 0
        for x in order:
            if can_add_mask(x, cur_mask):
                cur_mask |= (1 << (x - 1))
                cur_size += 1
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
    
    return mask_to_set(best_mask)

def local_search(s, deadline=None):
    """
    Improve set s using bitmask operations for speed.
    """
    cur_mask = 0
    for x in s:
        cur_mask |= (1 << (x - 1))
    cur_size = len(s)
    
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try to add any element
        for x in range(1, N + 1):
            if not (cur_mask >> (x - 1)) & 1:
                if can_add_mask(x, cur_mask):
                    cur_mask |= (1 << (x - 1))
                    cur_size += 1
                    improved = True
                    break
        if improved:
            continue
        
        # If no element can be added, try removing one element
        best_removal_mask = None
        best_removal_size = cur_size
        s_list = list(s)
        random.shuffle(s_list)
        
        for x in s_list:
            x_bit = 1 << (x - 1)
            new_mask = cur_mask & ~x_bit
            new_size = cur_size - 1
            # Try to greedily add elements
            for y in range(1, N + 1):
                y_bit = 1 << (y - 1)
                if not (new_mask >> (y - 1)) & 1:
                    if can_add_mask(y, new_mask):
                        new_mask |= y_bit
                        new_size += 1
            if new_size > best_removal_size:
                best_removal_size = new_size
                best_removal_mask = new_mask
                if new_size > cur_size + 1:
                    break
        
        if best_removal_mask is not None:
            cur_mask = best_removal_mask
            cur_size = best_removal_size
            improved = True
            continue
        
        # Try removing two elements
        if cur_size >= 4 and (deadline is None or time.time() < deadline - 3.0):
            best_pair_mask = None
            best_pair_size = cur_size
            s_list2 = list(s)
            random.shuffle(s_list2)
            max_pairs = 30 if deadline is None else max(5, int(30 * max(0, deadline - time.time()) / 10.0))
            pairs_tried = 0
            for i in range(len(s_list2)):
                if pairs_tried >= max_pairs:
                    break
                for j in range(i + 1, len(s_list2)):
                    if pairs_tried >= max_pairs:
                        break
                    x, y = s_list2[i], s_list2[j]
                    x_bit = 1 << (x - 1)
                    y_bit = 1 << (y - 1)
                    new_mask = cur_mask & ~x_bit & ~y_bit
                    new_size = cur_size - 2
                    for z in range(1, N + 1):
                        z_bit = 1 << (z - 1)
                        if not (new_mask >> (z - 1)) & 1:
                            if can_add_mask(z, new_mask):
                                new_mask |= z_bit
                                new_size += 1
                    pairs_tried += 1
                    if new_size > best_pair_size:
                        best_pair_size = new_size
                        best_pair_mask = new_mask
                        if new_size > cur_size:
                            break
                if best_pair_size > cur_size:
                    break
            
            if best_pair_mask is not None:
                cur_mask = best_pair_mask
                cur_size = best_pair_size
                improved = True
    
    return mask_to_set(cur_mask)

def bnb_independent_set(vertices, adjacency, deadline, best_size_ref):
    """
    Branch and Bound for Maximum Independent Set using bitmasks for speed.
    Uses Greedy Coloring Bound on the complement graph.
    """
    best_mask = 0
    
    # adj_mask is already global and precomputed
    
    def popcount(x):
        return bin(x).count('1')
    
    def get_bound(cand_mask):
        """
        Upper bound on MIS size of subgraph induced by cand_mask.
        Equivalent to Max Clique in complement.
        Greedy coloring of complement.
        """
        if cand_mask == 0:
            return 0
        
        # Get list of vertices
        verts = []
        cm = cand_mask
        while cm:
            lsb = cm & (-cm)
            v = lsb.bit_length()
            verts.append(v)
            cm ^= lsb
        
        # Order by degree in complement (descending) to help coloring
        # Degree in complement = (len(verts) - 1) - degree_in_G
        # We can just use original order or sort by degree in G (ascending) which is descending in complement
        # Let's sort by degree in G ascending (so high degree in complement first)
        verts.sort(key=lambda v: popcount(cand_mask & adj_mask[v]))
        
        colors = 0
        uncolored = verts[:]
        while uncolored:
            colors += 1
            color_class = []
            next_uncolored = []
            for v in uncolored:
                # v can be added to color_class if it is NOT adjacent to any u in color_class in Complement
                # NOT adjacent in Complement <=> Adjacent in G
                # So v must be Adjacent in G to ALL u in color_class
                can_add = True
                for u in color_class:
                    if not pair_conflict[v][u]:
                        can_add = False
                        break
                if can_add:
                    color_class.append(v)
                else:
                    next_uncolored.append(v)
            uncolored = next_uncolored
        return colors

    def rec(cand_mask, cur_mask, cur_size):
        nonlocal best_mask, best_size_ref
        
        if time.time() > deadline:
            return
        
        # Upper bound
        bound = get_bound(cand_mask)
        if cur_size + bound <= best_size_ref[0]:
            return
        
        if cand_mask == 0:
            if cur_size > best_size_ref[0]:
                best_size_ref[0] = cur_size
                best_mask = cur_mask
            return
        
        # Pick pivot: vertex with max degree in induced subgraph
        max_deg = -1
        pivot = -1
        cm = cand_mask
        while cm:
            lsb = cm & (-cm)
            v = lsb.bit_length()
            deg = popcount(cand_mask & adj_mask[v])
            if deg > max_deg:
                max_deg = deg
                pivot = v
            cm ^= lsb
        
        if pivot == -1:
            return
        
        pivot_bit = 1 << (pivot - 1)
        
        # Option 1: Include pivot
        new_cur_mask = cur_mask | pivot_bit
        new_cand_mask = cand_mask & ~pivot_bit & ~adj_mask[pivot]
        rec(new_cand_mask, new_cur_mask, cur_size + 1)
        
        # Option 2: Exclude pivot
        new_cand_mask = cand_mask & ~pivot_bit
        rec(new_cand_mask, cur_mask, cur_size)
    
    full_mask = 0
    for v in vertices:
        full_mask |= (1 << (v - 1))
    
    rec(full_mask, 0, 0)
    return mask_to_set(best_mask)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Greedy with multiple strategies + Local Search to get a good lower bound
    initial_set = greedy_construct(deadline)
    initial_set = local_search(initial_set, deadline)
    
    best_size = len(initial_set)
    best_set = initial_set
    
    # 2. Branch and Bound on the Pair-Conflict Graph
    adjacency = defaultdict(set)
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if pair_conflict[u][v]:
                adjacency[u].add(v)
                adjacency[v].add(u)
    
    vertices = list(range(1, N + 1))
    bnb_best_size = [best_size]
    
    # Run B&B with time limit
    remaining = deadline - time.time()
    if remaining > 5.0:
        bnb_result = bnb_independent_set(vertices, adjacency, deadline, bnb_best_size)
        
        if len(bnb_result) > best_size:
            best_size = len(bnb_result)
            best_set = bnb_result
    
    # 3. Final local search polish if time permits
    if time.time() < deadline - 3.0:
        best_set = local_search(best_set, deadline)
        if len(best_set) > best_size:
            best_size = len(best_set)
    
    result = list(best_set)
    if not result:
        result = [1]
        
    return result

# Test locally
if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify validity
    valid = True
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            for k in range(j + 1, len(res)):
                a, b, c = res[i], res[j], res[k]
                if math.lcm(a, b) == math.lcm(a, c) == math.lcm(b, c):
                    valid = False
                    print(f"Invalid triple: {a}, {b}, {c}")
    print(f"Valid: {valid}")
