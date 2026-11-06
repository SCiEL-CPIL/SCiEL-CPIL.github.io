import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety

    d = 7
    n_vertices = 2 ** d
    n_edges = n_vertices * d // 2  # 448

    # Edge representation: (u, v) with u < v
    # Map edge to index
    edges = []
    edge_to_idx = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                idx += 1

    # Generate all C4s (faces)
    c4s = []
    c4_set = set()
    for i in range(d):
        for j in range(i+1, d):
            bit_i = 1 << i
            bit_j = 1 << j
            for u in range(n_vertices):
                v1 = u
                v2 = u ^ bit_i
                v3 = u ^ bit_j
                v4 = u ^ bit_i ^ bit_j

                # The 4 edges
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))

                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]

                c4_sorted = tuple(sorted((idx1, idx2, idx3, idx4)))
                if c4_sorted not in c4_set:
                    c4_set.add(c4_sorted)
                    c4s.append(c4_sorted)

    n_c4s = len(c4s)

    # Precompute for each edge, which C4s it hits
    # edge_to_c4s[e] = list of c4 indices
    edge_to_c4s = [[] for _ in range(n_edges)]
    for c4_idx, c4 in enumerate(c4s):
        for e_idx in c4:
            edge_to_c4s[e_idx].append(c4_idx)

    # --- Initialization: Greedy Set Cover ---
    # We want to minimize R (removed edges).
    # A face is "hit" if at least one of its edges is in R.
    
    R = set()
    cover_count = [0] * n_c4s
    uncovered_c4s = set(range(n_c4s))
    
    # Greedy: pick edge that covers most uncovered C4s
    # Optimization: Precompute list of (edge, count) and update? 
    # For 448 edges and ~1344 faces, simple loop is fast enough for init.
    while uncovered_c4s:
        best_edge = -1
        best_count = -1
        for e in range(n_edges):
            if e in R:
                continue
            count = 0
            for c4_idx in edge_to_c4s[e]:
                if c4_idx in uncovered_c4s:
                    count += 1
            if count > best_count:
                best_count = count
                best_edge = e
        
        if best_edge == -1 or best_count == 0:
            break
        
        R.add(best_edge)
        for c4_idx in edge_to_c4s[best_edge]:
            cover_count[c4_idx] += 1
            if cover_count[c4_idx] == 1:
                uncovered_c4s.remove(c4_idx)

    current_size_R = len(R)
    best_size_R = current_size_R
    best_R = set(R)

    # --- Local Search with Simulated Annealing ---
    
    T = 1.0
    cooling_rate = 0.9995
    min_T = 1e-4
    
    # Helper functions

    def try_remove(edge):
        """Returns True if edge can be removed, and updates state if so."""
        nonlocal R, cover_count, current_size_R, best_size_R, best_R
        for c4_idx in edge_to_c4s[edge]:
            if cover_count[c4_idx] == 1:
                return False
        # Safe to remove
        R.remove(edge)
        for c4_idx in edge_to_c4s[edge]:
            cover_count[c4_idx] -= 1
        current_size_R -= 1
        if current_size_R < best_size_R:
            best_size_R = current_size_R
            best_R = set(R)
        return True

    def perform_swap(e_out, e_in):
        """
        Removes e_out and adds e_in.
        Returns True if the swap is valid.
        """
        nonlocal R, cover_count, current_size_R, best_size_R, best_R
        
        e_in_c4s_set = set(edge_to_c4s[e_in])
        
        for c4_idx in edge_to_c4s[e_out]:
            if cover_count[c4_idx] == 1 and c4_idx not in e_in_c4s_set:
                return False
        
        # Perform swap
        R.remove(e_out)
        R.add(e_in)
        
        for c4_idx in edge_to_c4s[e_out]:
            cover_count[c4_idx] -= 1
        for c4_idx in edge_to_c4s[e_in]:
            cover_count[c4_idx] += 1
            
        return True

    def perform_2_for_1_swap(e_out1, e_out2, e_in):
        """
        Removes e_out1 and e_out2, adds e_in.
        Size decreases by 1.
        Returns True if valid.
        """
        nonlocal R, cover_count, current_size_R, best_size_R, best_R
        
        affected_faces = set(edge_to_c4s[e_out1])
        affected_faces.update(edge_to_c4s[e_out2])
        
        for f in affected_faces:
            old_count = cover_count[f]
            dec = 0
            if f in edge_to_c4s[e_out1]: dec += 1
            if f in edge_to_c4s[e_out2]: dec += 1
            inc = 1 if f in edge_to_c4s[e_in] else 0
            new_count = old_count - dec + inc
            if new_count < 1:
                return False
        
        # Perform swap
        R.remove(e_out1)
        R.remove(e_out2)
        R.add(e_in)
        
        for f in edge_to_c4s[e_out1]:
            cover_count[f] -= 1
        for f in edge_to_c4s[e_out2]:
            cover_count[f] -= 1
        for f in edge_to_c4s[e_in]:
            cover_count[f] += 1
            
        current_size_R -= 1
        if current_size_R < best_size_R:
            best_size_R = current_size_R
            best_R = set(R)
            
        return True

    # Main SA Loop
    # Use a list for R for faster random selection if needed, but set is fine for membership.
    # We will alternate between removal, swap, and 2-for-1.
    
    while time.time() < deadline - 10.0 and T > min_T:
        r = random.random()
        
        # 1. Try Removal (Greedy step)
        if r < 0.2 and R:
            # Pick a random edge from R
            e_out = random.choice(list(R))
            if try_remove(e_out):
                continue # Restart loop with new state
            
        # 2. Try 1-for-1 Swap (Exploration)
        elif r < 0.5 and R:
            e_out = random.choice(list(R))
            e_in = random.randint(0, n_edges - 1)
            if e_in in R:
                continue
            if perform_swap(e_out, e_in):
                pass # Accepted
            
        # 3. Try 2-for-1 Swap (Reduction)
        elif len(R) >= 2:
            # Pick 2 edges to remove
            e_out1, e_out2 = random.sample(list(R), 2)
            e_in = random.randint(0, n_edges - 1)
            if e_in in R:
                continue
            if perform_2_for_1_swap(e_out1, e_out2, e_in):
                pass # Accepted, size decreased
        
        T *= cooling_rate

    # Final Polish: Deterministic Greedy Removal
    # This is crucial for squeezing out the last few edges.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate over a snapshot of R
        for e in list(R):
            if try_remove(e):
                changed = True
                # If we removed one, we might be able to remove more, so restart inner loop
                break 

    if current_size_R < best_size_R:
        best_R = set(R)

    # Construct final edge list
    # Valid edges are those NOT in best_R
    valid_edges = []
    for i, (u, v) in enumerate(edges):
        if i not in best_R:
            valid_edges.append([u, v])
            
    return valid_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
