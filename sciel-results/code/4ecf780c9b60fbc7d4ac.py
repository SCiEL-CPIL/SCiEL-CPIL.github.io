"""Four-cycle-free edge sets of Q7: maximize edges with no 2-face fully selected."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    d = 7
    n = 1 << d  # 128
    total_edges = n * d // 2  # 448

    # 1. Generate all edges and map to indices
    all_edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_index[(x, y)] = idx

    # 2. Precompute edge info
    edge_info = []
    for idx, (u, v) in enumerate(all_edges):
        diff = u ^ v
        bit = diff.bit_length() - 1
        edge_info.append((u, v, bit))

    # 3. Precompute all C4s (faces)
    all_c4s = []
    for i in range(d):
        for j in range(i + 1, d):
            mask = (1 << i) | (1 << j)
            for x in range(n):
                if (x & mask) == 0:
                    v1 = x
                    v2 = x ^ (1 << i)
                    v3 = x ^ (1 << j)
                    v4 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_index[(min(v1, v2), max(v1, v2))]
                    e2 = edge_index[(min(v3, v4), max(v3, v4))]
                    e3 = edge_index[(min(v1, v3), max(v1, v3))]
                    e4 = edge_index[(min(v2, v4), max(v2, v4))]
                    
                    all_c4s.append((e1, e2, e3, e4))

    num_c4s = len(all_c4s)
    
    # 4. Precompute for each edge, the list of C4 indices it belongs to
    c4_membership = [[] for _ in range(total_edges)]
    for c4_idx, (e1, e2, e3, e4) in enumerate(all_c4s):
        for e in (e1, e2, e3, e4):
            c4_membership[e].append(c4_idx)

    # 5. Greedy Deletion to get a valid starting state
    # We want to minimize the set of removed edges R.
    # Start with R = empty (all edges present). This is invalid.
    # Greedy: while there is a bad C4 (all 4 edges present), remove one edge from it.
    # We prefer removing edges that are in many bad C4s.
    
    present = [True] * total_edges
    c4_present_count = [4] * num_c4s
    bad_c4s = set(range(num_c4s))
    
    # To speed up greedy, we can just pick edges in bad C4s.
    # Let's use a simple heuristic: repeatedly pick a random bad C4 and remove a random edge from it.
    # Or better: pick the edge in bad C4s that covers the most bad C4s.
    
    # For efficiency in the start, we'll do a few passes of "remove edge with max bad coverage"
    while bad_c4s:
        if time.time() > deadline - 60:
            break
        max_count = -1
        best_e = -1
        # Sample a few bad C4s to find a good candidate edge
        # For simplicity and speed, just iterate all edges? No, that's slow.
        # Let's just pick a random bad C4 and a random edge in it.
        # This is slower to converge but faster per step.
        # Let's try a hybrid: check a few random bad C4s.
        
        # Actually, the original code's greedy was quite fast. Let's stick to a similar logic but optimized.
        # We'll just remove edges from random bad C4s until valid.
        
        # Pick a random bad C4
        c4_idx = random.choice(list(bad_c4s))
        e1, e2, e3, e4 = all_c4s[c4_idx]
        # Pick a random edge from this C4 that is currently present
        candidates = [e for e in (e1, e2, e3, e4) if present[e]]
        if not candidates:
            # Should not happen if c4_present_count is correct
            continue
        e_to_remove = random.choice(candidates)
        
        present[e_to_remove] = False
        for c4_idx2 in c4_membership[e_to_remove]:
            c4_present_count[c4_idx2] -= 1
            if c4_present_count[c4_idx2] < 4:
                bad_c4s.discard(c4_idx2)

    # Now we have a valid solution.
    # R is the set of removed edges.
    removed = [e for e in range(total_edges) if not present[e]]
    current_removed_set = set(removed)
    current_size = len(removed)
    best_size = current_size
    best_removed_set = set(removed)
    
    # Track bad count for SA
    # c4_present_count is 4 - count_in_R
    # Bad C4s are those with c4_present_count == 4 (i.e., count_in_R == 0)
    # Let's maintain count_in_R directly for clarity in SA.
    count_in_R = [0] * num_c4s
    for e in removed:
        for c4_idx in c4_membership[e]:
            count_in_R[c4_idx] += 1
            
    bad_count = sum(1 for c in count_in_R if c == 0)
    # It should be 0.
    
    # 6. Simulated Annealing
    # We want to minimize size of R subject to bad_count == 0.
    # Energy E = alpha * bad_count + size
    # If bad_count > 0, we prioritize reducing it.
    # If bad_count == 0, we minimize size.
    
    alpha = 10.0 # Penalty for invalidity
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # We will perform moves until time limit or T < min_T
    # Move: pick random edge e.
    # If e in R: try to remove (size -1, bad_count might increase)
    # If e not in R: try to add (size +1, bad_count might decrease)
    
    steps = 0
    while time.time() < deadline - 10 and T > min_T:
        steps += 1
        e = random.randint(0, total_edges - 1)
        
        if e in current_removed_set:
            # Try to remove e from R
            # Delta size = -1
            # Delta bad_count: for each C4 in e, if count_in_R[C4] == 1, it becomes 0 -> bad_count +1
            delta_bad = 0
            for c4_idx in c4_membership[e]:
                if count_in_R[c4_idx] == 1:
                    delta_bad += 1
            
            delta_E = alpha * delta_bad - 1.0
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                current_removed_set.remove(e)
                current_size -= 1
                for c4_idx in c4_membership[e]:
                    count_in_R[c4_idx] -= 1
                    if count_in_R[c4_idx] == 0:
                        bad_count += 1
                # Update best if valid
                if bad_count == 0 and current_size < best_size:
                    best_size = current_size
                    best_removed_set = set(current_removed_set)
        else:
            # Try to add e to R
            # Delta size = +1
            # Delta bad_count: for each C4 in e, if count_in_R[C4] == 0, it becomes 1 -> bad_count -1
            delta_bad = 0
            for c4_idx in c4_membership[e]:
                if count_in_R[c4_idx] == 0:
                    delta_bad -= 1
            
            delta_E = alpha * delta_bad + 1.0
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                current_removed_set.add(e)
                current_size += 1
                for c4_idx in c4_membership[e]:
                    count_in_R[c4_idx] += 1
                    if count_in_R[c4_idx] == 1:
                        bad_count -= 1
                # Note: adding edges to R increases size, so it rarely improves best_size immediately,
                # but it might fix bad_count allowing subsequent removals to succeed?
                # Actually, if we are valid (bad_count=0), adding to R keeps it valid but increases size.
                # We should probably bias against this if we are already valid.
                # The delta_E = 1.0 (if bad_count doesn't change) will be rejected with high prob at low T.
                # If bad_count decreases (was >0), delta_E = -alpha + 1 < 0, accepted.
        
        T *= cooling

    # 7. Final Polish: Greedy removal from R
    # Try to remove edges from best_removed_set if possible without creating bad C4s.
    # We need to reconstruct state from best_removed_set
    count_in_R_best = [0] * num_c4s
    for e in best_removed_set:
        for c4_idx in c4_membership[e]:
            count_in_R_best[c4_idx] += 1
            
    # Check if valid
    if all(c > 0 for c in count_in_R_best):
        # Try to remove edges
        for e in list(best_removed_set):
            if time.time() > deadline - 5:
                break
            # Check if removing e keeps it valid
            ok = True
            for c4_idx in c4_membership[e]:
                if count_in_R_best[c4_idx] == 1:
                    ok = False
                    break
            if ok:
                best_removed_set.remove(e)
                for c4_idx in c4_membership[e]:
                    count_in_R_best[c4_idx] -= 1
                best_size -= 1

    # 8. Construct Output
    # Edges NOT in best_removed_set are selected
    out = []
    for idx, (u, v) in enumerate(all_edges):
        if idx not in best_removed_set:
            out.append([u, v])
            
    return out

# EVOLVE-BLOCK-END
