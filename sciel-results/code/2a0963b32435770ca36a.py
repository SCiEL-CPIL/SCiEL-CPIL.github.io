import random
import time

def construct():
    d = 7
    n = 1 << d
    # Precompute all edges
    edges = []
    edge_to_idx = {}
    idx_to_edge = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v, i))
                edge_to_idx[(u, v)] = idx
                idx_to_edge[idx] = (u, v)
    
    # Precompute all C4s
    c4s = []
    for u in range(n):
        for i in range(d):
            for j in range(i+1, d):
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    v1 = u ^ (1 << i)
                    v2 = u ^ (1 << j)
                    v3 = u ^ (1 << i) ^ (1 << j)
                    e1 = (min(u, v1), max(u, v1))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v3, v2), max(v3, v2))
                    e4 = (min(v2, u), max(v2, u))
                    c4s.append([edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]])
    
    # Map edge index to list of C4 indices containing it
    edge_c4s = [[] for _ in range(len(edges))]
    for c4_idx, c4 in enumerate(c4s):
        for e_idx in c4:
            edge_c4s[e_idx].append(c4_idx)
    
    best_size = 0
    best_edges = []
    
    start_time = time.time()
    max_time = 110
    
    # Helper to check if adding an edge creates a C4
    def can_add_edge(e_idx, selected):
        for c4_idx in edge_c4s[e_idx]:
            c4 = c4s[c4_idx]
            # Check if all other 3 edges are selected
            if all(selected[e] for e in c4 if e != e_idx):
                return False
        return True

    for _ in range(100): # Fewer iterations, but smarter
        if time.time() - start_time > max_time:
            break
        
        selected = [True] * len(edges)
        
        # Greedy removal
        bad_c4s = set()
        for c4_idx in range(len(c4s)):
            c4 = c4s[c4_idx]
            if all(selected[e] for e in c4):
                bad_c4s.add(c4_idx)
        
        # Count C4s for each edge to prioritize removal
        c4_count = [0] * len(edges)
        for e_idx in range(len(edges)):
            c4_count[e_idx] = len(edge_c4s[e_idx])
        
        while bad_c4s:
            # Pick the bad C4 with the highest sum of c4_counts of its edges
            # This helps break "dense" clusters of C4s
            best_c4_idx = -1
            max_sum = -1
            for c4_idx in bad_c4s:
                c4 = c4s[c4_idx]
                s = sum(c4_count[e] for e in c4)
                if s > max_sum:
                    max_sum = s
                    best_c4_idx = c4_idx
            
            if best_c4_idx == -1:
                break
                
            c4 = c4s[best_c4_idx]
            
            # Remove the edge with the highest c4_count in this C4
            max_cnt = -1
            best_e = -1
            for e_idx in c4:
                if selected[e_idx] and c4_count[e_idx] > max_cnt:
                    max_cnt = c4_count[e_idx]
                    best_e = e_idx
            
            if best_e == -1:
                bad_c4s.remove(best_c4_idx)
                continue
            
            # Remove edge
            selected[best_e] = False
            bad_c4s.remove(best_c4_idx)
            
            # Update counts for other edges in C4s containing best_e
            for c4_idx_2 in edge_c4s[best_e]:
                if c4_idx_2 in bad_c4s:
                    bad_c4s.remove(c4_idx_2)
                    c4_2 = c4s[c4_idx_2]
                    for e2 in c4_2:
                        if e2 != best_e and selected[e2]:
                            c4_count[e2] -= 1
                            
        # Local Search: Try to add back edges
        # Shuffle edges to try different orders
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        
        for e_idx in edge_indices:
            if not selected[e_idx]:
                if can_add_edge(e_idx, selected):
                    selected[e_idx] = True
                    # Update bad_c4s? We don't maintain it during add-back, 
                    # but can_add_edge is O(C4s per edge) which is small.
        
        current_size = sum(selected)
        if current_size > best_size:
            best_size = current_size
            best_edges = [[idx_to_edge[i][0], idx_to_edge[i][1]] for i in range(len(edges)) if selected[i]]
            
    return best_edges
