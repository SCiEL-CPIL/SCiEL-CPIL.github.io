import random
import time
import sys

def construct():
    d = 7
    n_vertices = 1 << d
    # Generate all edges
    edges = []
    edge_map = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_map[(u, v)] = idx
                edges.append((u, v))
                idx += 1
    
    n_edges = len(edges) # 448
    
    # Precompute conflicts: for each edge, list of 3-edge tuples that block it
    conflicts = [[] for _ in range(n_edges)]
    
    for e_idx, (u, v) in enumerate(edges):
        diff = u ^ v
        bit_i = diff.bit_length() - 1
        
        squares = []
        for bit_j in range(d):
            if bit_j == bit_i:
                continue
            mask_j = 1 << bit_j
            u_prime = u ^ mask_j
            v_prime = v ^ mask_j
            
            e1 = edge_map.get((min(u, u_prime), max(u, u_prime)))
            e2 = edge_map.get((min(v, v_prime), max(v, v_prime)))
            e3 = edge_map.get((min(u_prime, v_prime), max(u_prime, v_prime)))
            
            if e1 is not None and e2 is not None and e3 is not None:
                squares.append((e1, e2, e3))
        
        conflicts[e_idx] = squares
    
    # For O(1) add-check: blockers[e_idx] counts how many conflict triples are fully present
    # When we add edge e, for each other edge f that has a conflict triple containing e,
    # we need to increment/decrement blockers[f] if the other two edges of that triple are present.
    # Precompute: for each edge e, list of (f, other1, other2) where {e, other1, other2} is a conflict triple for f
    # i.e., conflicts[f] contains (e, other1, other2)
    edge_to_blockers = [[] for _ in range(n_edges)]
    for f in range(n_edges):
        for (e, a, b) in conflicts[f]:
            edge_to_blockers[e].append((f, a, b))
            # Also need to handle when e is 'a' or 'b' in another conflict
            # Actually, let's be more careful.
            # conflicts[f] = list of triples (x,y,z) meaning if x,y,z all in set, f is blocked.
            # When adding edge e, for each f where e appears in some triple (e,a,b) in conflicts[f],
            # if a and b are both in set, then f becomes blocked (blockers[f] += 1).
            # When removing edge e, for each f where e appears in some triple,
            # if a and b are both in set, then f becomes unblocked (blockers[f] -= 1).
    
    # Build reverse mapping: for each edge e, list of (f, a, b) such that (e,a,b) in conflicts[f]
    # We already have edge_to_blockers built above, but let's verify it's complete.
    # The above loop adds (f, a, b) to edge_to_blockers[e] for each triple (e,a,b) in conflicts[f].
    # But what if the triple is (a, e, b) or (a, b, e)? We need to handle all positions.
    # Let me rebuild properly.
    edge_to_blockers = [[] for _ in range(n_edges)]
    for f in range(n_edges):
        for triple in conflicts[f]:
            for pos, e in enumerate(triple):
                # The other two edges in the triple
                others = [triple[i] for i in range(3) if i != pos]
                edge_to_blockers[e].append((f, others[0], others[1]))
    
    # State: in_set[e] = True if edge e is in current set
    in_set = [False] * n_edges
    # blockers[f] = number of conflict triples for f that are fully satisfied
    blockers = [0] * n_edges
    
    def can_add(e_idx):
        """Check if edge e_idx can be added (O(1) check)."""
        return blockers[e_idx] == 0
    
    def add_edge(e_idx):
        """Add edge e_idx to the set. O(|edge_to_blockers[e_idx]|) update."""
        in_set[e_idx] = True
        for (f, a, b) in edge_to_blockers[e_idx]:
            if in_set[a] and in_set[b]:
                blockers[f] += 1
    
    def remove_edge(e_idx):
        """Remove edge e_idx from the set. O(|edge_to_blockers[e_idx]|) update."""
        in_set[e_idx] = False
        for (f, a, b) in edge_to_blockers[e_idx]:
            if in_set[a] and in_set[b]:
                blockers[f] -= 1
    
    def count_edges():
        return sum(in_set)
    
    # Greedy initialization
    for e_idx in range(n_edges):
        if can_add(e_idx):
            add_edge(e_idx)
    
    best_count = count_edges()
    best_in_set = in_set[:]
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    T = 5.0
    cooling = 0.9998
    min_T = 0.05
    no_improve = 0
    stagnation_limit = 50000
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.35:
            # Try Add
            e_idx = random.randrange(n_edges)
            if not in_set[e_idx] and can_add(e_idx):
                add_edge(e_idx)
                cur = count_edges()
                if cur > best_count:
                    best_count = cur
                    best_in_set = in_set[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        elif move_type < 0.55:
            # Try Remove
            if best_count > 0:
                # Pick a random edge that's in the set
                # Use rejection sampling
                while True:
                    e_idx = random.randrange(n_edges)
                    if in_set[e_idx]:
                        break
                remove_edge(e_idx)
                # Accept with probability exp(-1/T)
                if random.random() > (2.718281828459045 ** (-1.0 / T)):
                    # Reject, revert
                    add_edge(e_idx)
                no_improve += 1
        else:
            # Try Swap
            if best_count > 0 and best_count < n_edges:
                # Pick e_out from set
                while True:
                    e_out = random.randrange(n_edges)
                    if in_set[e_out]:
                        break
                # Try several random e_in candidates
                success = False
                for _ in range(3):
                    e_in = random.randrange(n_edges)
                    if in_set[e_in]:
                        continue
                    # Check if swap is valid: remove e_out, then can we add e_in?
                    # Temporarily remove e_out
                    remove_edge(e_out)
                    if can_add(e_in):
                        add_edge(e_in)
                        # Swap accepted (neutral move)
                        cur = count_edges()
                        if cur > best_count:
                            best_count = cur
                            best_in_set = in_set[:]
                            no_improve = 0
                        success = True
                        break
                    else:
                        # Revert
                        add_edge(e_out)
                if not success:
                    no_improve += 1
        
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
        
        # Reheat if stuck
        if no_improve > stagnation_limit:
            T = min(T * 2.0, 5.0)
            no_improve = 0
    
    # Convert best_in_set to list of [u, v]
    result = []
    for e_idx in range(n_edges):
        if best_in_set[e_idx]:
            u, v = edges[e_idx]
            result.append([u, v])
    
    return result
