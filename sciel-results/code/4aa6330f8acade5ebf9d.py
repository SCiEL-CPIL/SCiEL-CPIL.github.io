import random
import time

# Precomputation
N_DIMS = 7
N_VERTS = 1 << N_DIMS
N_EDGES = N_DIMS * (N_VERTS // 2) # 7 * 64 = 448

def precompute():
    edges = []
    edge_map = {} # (u, v) with u<v -> index
    for u in range(N_VERTS):
        for i in range(N_DIMS):
            v = u ^ (1 << i)
            if u < v:
                edge_map[(u, v)] = len(edges)
                edges.append((u, v))
    
    edge_squares = [[] for _ in range(len(edges))]
    square_count = 0
    square_edges = [[] for _ in range(512)]  # Max possible squares (C(7,2) * 2^5 = 21*32=672, but actual is less)
    # Actually let's just build it dynamically
    
    # First pass: count squares
    # There are C(7,2) = 21 pairs of dimensions. For each pair, 2^(7-2) = 32 faces. Total 21*32 = 672 faces.
    # Wait, the problem says "No 2-dimensional face may have all four edges selected."
    # A 2D face in Q7 is a square.
    
    for i in range(N_DIMS):
        for j in range(i + 1, N_DIMS):
            other_bits = [k for k in range(N_DIMS) if k != i and k != j]
            for mask in range(1 << len(other_bits)):
                base = 0
                for k in range(len(other_bits)):
                    if (mask >> k) & 1:
                        base |= (1 << other_bits[k])
                
                v00 = base
                v10 = base | (1 << i)
                v01 = base | (1 << j)
                v11 = base | (1 << i) | (1 << j)
                
                e1 = (min(v00, v10), max(v00, v10))
                e2 = (min(v00, v01), max(v00, v01))
                e3 = (min(v10, v11), max(v10, v11))
                e4 = (min(v01, v11), max(v01, v11))
                
                idx1 = edge_map[e1]
                idx2 = edge_map[e2]
                idx3 = edge_map[e3]
                idx4 = edge_map[e4]
                
                # Add to edge_squares
                edge_squares[idx1].append(square_count)
                edge_squares[idx2].append(square_count)
                edge_squares[idx3].append(square_count)
                edge_squares[idx4].append(square_count)
                
                # Add to square_edges
                square_edges[square_count] = [idx1, idx2, idx3, idx4]
                
                square_count += 1
                
    return edges, edge_squares, square_count, square_edges

EDGES, EDGE_SQUARES, NUM_SQUARES, SQUARE_EDGES = precompute()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_count = 0
    best_selected = [False] * N_EDGES
    
    def greedy_start():
        indices = list(range(N_EDGES))
        random.shuffle(indices)
        selected = [False] * N_EDGES
        sq_counts = [0] * NUM_SQUARES
        count = 0
        
        for idx in indices:
            valid = True
            for s in EDGE_SQUARES[idx]:
                if sq_counts[s] >= 3:
                    valid = False
                    break
            if valid:
                selected[idx] = True
                count += 1
                for s in EDGE_SQUARES[idx]:
                    sq_counts[s] += 1
        return selected, count, sq_counts
    
    def local_search(selected, count, sq_counts):
        improved = True
        no_improve_streak = 0
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            no_improve_streak += 1
            
            if no_improve_streak > 10:
                break
            
            # 1. Try to add any unselected edge
            unselected = [e for e in range(N_EDGES) if not selected[e]]
            random.shuffle(unselected)
            
            for e in unselected:
                if time.time() >= deadline - 2.0:
                    break
                sqs = EDGE_SQUARES[e]
                can_add = True
                for s in sqs:
                    if sq_counts[s] >= 3:
                        can_add = False
                        break
                if can_add:
                    selected[e] = True
                    count += 1
                    for s in sqs:
                        sq_counts[s] += 1
                    improved = True
            
            if improved:
                no_improve_streak = 0
                continue
            
            # 2. Try swaps: remove one edge, add one edge (net 0 change in count, but might allow future adds)
            # Actually, let's try: remove edge A, add edge B. If B was blocked by A's square being full, this might work.
            # Better: Try to remove edge A and add edge B where B was previously blocked.
            # Even better: Try to remove edge A and add TWO edges B and C? (Net +1)
            # Let's stick to simple swaps first: remove A, add B.
            # If we remove A, some squares have count 3->2. Then B might become addable.
            
            # To make it efficient, we can just try all pairs (A, B) where A is selected, B is not.
            # But that's O(N^2) = 448^2 ~ 200k. Too slow for each iteration?
            # We can sample.
            
            selected_idx = [e for e in range(N_EDGES) if selected[e]]
            if not selected_idx:
                break
            
            random.shuffle(selected_idx)
            random.shuffle(unselected)
            
            swap_found = False
            for a in selected_idx[:20]: # Sample 20 removals
                if time.time() >= deadline - 2.0:
                    break
                
                # Calculate squares affected by removing a
                affected_sqs = set(EDGE_SQUARES[a])
                
                # Temporarily remove a
                selected[a] = False
                for s in EDGE_SQUARES[a]:
                    sq_counts[s] -= 1
                
                # Try to add any B
                for b in unselected[:20]: # Sample 20 additions
                    if selected[b]:
                        continue
                    sqs_b = EDGE_SQUARES[b]
                    can_add = True
                    for s in sqs_b:
                        if sq_counts[s] >= 3:
                            can_add = False
                            break
                    if can_add:
                        selected[b] = True
                        count += 1
                        for s in sqs_b:
                            sq_counts[s] += 1
                        swap_found = True
                        improved = True
                        break
                
                # Restore a if we didn't swap (or even if we did, wait, if we swapped, we removed a and added b. 
                # If we didn't swap, we must restore a.)
                if not swap_found:
                    # Wait, if swap_found is False for this 'a', we need to put 'a' back.
                    # But what if we tried multiple 'a's?
                    # Let's restructure: for each 'a', try to find a 'b'. If found, commit and break outer loop.
                    # If not found for this 'a', restore 'a'.
                    selected[a] = True
                    for s in EDGE_SQUARES[a]:
                        sq_counts[s] += 1
                else:
                    # We committed to removing 'a' and adding 'b'.
                    # We broke out of inner loop. Now we need to break out of outer loop.
                    # The state is already updated (a removed, b added).
                    break
            
        return selected, count, sq_counts
    
    # Phase 1: Multiple restarts
    for i in range(1000):
        if time.time() >= deadline - 5.0:
            break
        
        selected, count, sq_counts = greedy_start()
        selected, count, sq_counts = local_search(selected, count, sq_counts)
        
        if count > best_count:
            best_count = count
            best_selected = list(selected)
    
    # Phase 2: Intensive local search on best solution
    if best_count > 0:
        # Reconstruct sq_counts
        sq_counts = [0] * NUM_SQUARES
        for e in range(N_EDGES):
            if best_selected[e]:
                for s in EDGE_SQUARES[e]:
                    sq_counts[s] += 1
        
        selected = list(best_selected)
        count = best_count
        selected, count, sq_counts = local_search(selected, count, sq_counts)
        
        if count > best_count:
            best_count = count
            best_selected = list(selected)
    
    return [[u, v] for i, (u, v) in enumerate(EDGES) if best_selected[i]]
