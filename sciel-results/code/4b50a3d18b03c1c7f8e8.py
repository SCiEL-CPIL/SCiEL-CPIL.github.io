import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # Generate all edges and map them to indices
    edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
    
    # Precompute face information
    face_partners = [[] for _ in range(n_edges)]
    
    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        dim = diff.bit_length() - 1
        
        for j in range(d):
            if j == dim:
                continue
            w1 = u ^ (1 << j)
            w2 = v ^ (1 << j)
            
            e1 = edge_to_idx.get((min(u, w1), max(u, w1)))
            e2 = edge_to_idx.get((min(v, w2), max(v, w2)))
            e3 = edge_to_idx.get((min(w1, w2), max(w1, w2)))
            
            face_partners[idx].append((e1, e2, e3))
            
    def calc_block_counts(in_set):
        block_counts = [0] * n_edges
        for x in range(n_edges):
            if not in_set[x]:
                count = 0
                for (a, b, c) in face_partners[x]:
                    if in_set[a] and in_set[b] and in_set[c]:
                        count += 1
                block_counts[x] = count
        return block_counts

    def greedy_fill(in_set, order):
        for idx in order:
            if not in_set[idx]:
                ok = True
                for (a, b, c) in face_partners[idx]:
                    if in_set[a] and in_set[b] and in_set[c]:
                        ok = False
                        break
                if ok:
                    in_set[idx] = True

    best_solution = [False] * n_edges
    best_count = 0
    
    while time.time() < deadline:
        # Random restart
        order = list(range(n_edges))
        random.shuffle(order)
        current_set = [False] * n_edges
        greedy_fill(current_set, order)
        count = sum(current_set)
        
        # Local Search
        block_counts = calc_block_counts(current_set)
        queue = [i for i in range(n_edges) if not current_set[i] and block_counts[i] == 0]
        
        # Allow multiple moves if stuck
        moves_made = 0
        max_moves = 50 # Limit moves per restart to avoid infinite loop on bad restarts
        
        while time.time() < deadline:
            # Process queue
            while queue:
                idx = queue.pop()
                if current_set[idx]:
                    continue
                if block_counts[idx] > 0:
                    continue
                
                # Add idx
                current_set[idx] = True
                count += 1
                
                # Update neighbors
                for (a, b, c) in face_partners[idx]:
                    if not current_set[a]:
                        if current_set[b] and current_set[c]:
                            block_counts[a] += 1
                    if not current_set[b]:
                        if current_set[a] and current_set[c]:
                            block_counts[b] += 1
                    if not current_set[c]:
                        if current_set[a] and current_set[b]:
                            block_counts[c] += 1

            # If queue is empty, we are stuck
            if not queue:
                if moves_made >= max_moves:
                    break
                
                # Make a move: remove a random edge
                in_indices = [i for i in range(n_edges) if current_set[i]]
                if not in_indices:
                    break
                
                rem_idx = random.choice(in_indices)
                current_set[rem_idx] = False
                count -= 1
                moves_made += 1
                
                # Update block counts (decrement)
                for (a, b, c) in face_partners[rem_idx]:
                    if not current_set[a]:
                        if current_set[b] and current_set[c]:
                            block_counts[a] -= 1
                            if block_counts[a] == 0:
                                queue.append(a)
                    if not current_set[b]:
                        if current_set[a] and current_set[c]:
                            block_counts[b] -= 1
                            if block_counts[b] == 0:
                                queue.append(b)
                    if not current_set[c]:
                        if current_set[a] and current_set[b]:
                            block_counts[c] -= 1
                            if block_counts[c] == 0:
                                queue.append(c)
                
                # Loop back to process queue
                continue
            
            # If we exited the while queue loop because of time, break outer too
            break

        if count > best_count:
            best_count = count
            best_solution = current_set[:]
            
    result = []
    for idx, is_in in enumerate(best_solution):
        if is_in:
            result.append(list(edges[idx]))
            
    return result
