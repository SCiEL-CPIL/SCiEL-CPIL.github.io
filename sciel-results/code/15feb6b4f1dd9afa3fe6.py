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
    
    # Precompute conflicts
    # conflicts[e_idx] = list of 6 tuples, each tuple is (e1, e2, e3)
    # If e1, e2, e3 are in the set, e_idx cannot be added.
    conflicts = [[] for _ in range(n_edges)]
    
    for e_idx, (u, v) in enumerate(edges):
        # Determine which bit is flipped
        diff = u ^ v
        bit_i = diff.bit_length() - 1
        
        squares = []
        for bit_j in range(d):
            if bit_j == bit_i:
                continue
            mask_j = 1 << bit_j
            u_prime = u ^ mask_j
            v_prime = v ^ mask_j
            
            # The other 3 edges are (u, u_prime), (v, v_prime), (u_prime, v_prime)
            e1 = edge_map.get((min(u, u_prime), max(u, u_prime)))
            e2 = edge_map.get((min(v, v_prime), max(v, v_prime)))
            e3 = edge_map.get((min(u_prime, v_prime), max(u_prime, v_prime)))
            
            if e1 is not None and e2 is not None and e3 is not None:
                squares.append((e1, e2, e3))
        
        conflicts[e_idx] = squares

    def is_valid_add(e_idx, current_set):
        # Check if adding e_idx creates a C4
        for e1, e2, e3 in conflicts[e_idx]:
            if e1 in current_set and e2 in current_set and e3 in current_set:
                return False
        return True

    # Initialize with a greedy solution
    current_set = set()
    # Randomize edge order for greedy to break symmetry? Or just standard greedy.
    # Standard greedy:
    for e_idx in range(n_edges):
        if is_valid_add(e_idx, current_set):
            current_set.add(e_idx)
    
    best_set = set(current_set)
    best_score = len(current_set)
    
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    T = 10.0
    cooling = 0.9999
    min_T = 0.01
    
    # Lists for fast random selection
    all_edges = list(range(n_edges))
    
    while time.time() < deadline:
        # Propose move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try Add
            e_idx = random.choice(all_edges)
            if e_idx not in current_set:
                if is_valid_add(e_idx, current_set):
                    current_set.add(e_idx)
                    # Always accept improvement
                    if len(current_set) > best_score:
                        best_score = len(current_set)
                        best_set = set(current_set)
                else:
                    # Reject
                    pass
            else:
                # Edge already in set, try something else or skip
                pass
                
        elif move_type < 0.6:
            # Try Remove
            if current_set:
                e_idx = random.choice(list(current_set))
                current_set.remove(e_idx)
                # Accept with probability exp(-delta/T) where delta = 1 (score decreases by 1)
                # We store the removed edge to potentially revert? No, SA state is current_set.
                # We just accept the state.
                # But we need to track if we should "revert" based on SA logic?
                # Standard SA: We propose a move, calculate new energy. If worse, accept with prob.
                # Here, the state IS the set.
                # So we remove it. If we decide to "reject" the move, we must put it back.
                
                # Let's re-structure:
                # 1. Propose move
                # 2. Calculate delta
                # 3. Accept/Reject
                # If Reject, revert state.
                
                # So for Remove:
                # State change: Remove e_idx. Delta = +1 (worse).
                # Accept prob = exp(-1/T).
                if random.random() > (2.718281828 ** (-1.0 / T)):
                    # Reject, revert
                    current_set.add(e_idx)
                    
        else:
            # Try Swap
            if len(current_set) > 0 and len(current_set) < n_edges:
                e_out = random.choice(list(current_set))
                # Pick e_in not in set
                # Efficient way: pick random, if in set, pick another? 
                # Or maintain a list of available edges? 
                # With 448 edges, picking random and checking is fast.
                e_in = random.choice(all_edges)
                if e_in not in current_set:
                    # Remove e_out
                    current_set.remove(e_out)
                    # Check if e_in is valid
                    if is_valid_add(e_in, current_set):
                        current_set.add(e_in)
                        # Delta = 0. Accept with prob 1 (or some prob)
                        # Usually accept neutral moves to allow exploration
                        if random.random() > 0.5: # Sometimes reject to avoid loops? No, accept to explore.
                             pass # Accepted
                    else:
                        # e_in invalid, revert
                        current_set.add(e_out)
                        # Reject move
                
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
            # Optional: Reheat if stuck?
            # For now, just keep going.
            
    # Convert best_set to list of [u, v]
    result = []
    for e_idx in best_set:
        u, v = edges[e_idx]
        result.append([u, v])
        
    return result
