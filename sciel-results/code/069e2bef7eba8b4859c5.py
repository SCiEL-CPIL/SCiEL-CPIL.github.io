import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    # Precompute all K4s and their edge indices
    k4s = []
    edge_index = {}
    edge_ij = []  # map edge index -> (i, j)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            edge_ij.append((i, j))
            idx += 1
            
    # Total edges = 120
    num_edges = 120
    
    for a,b,c,d in combinations(range(n), 4):
        edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
        edge_idxs = [edge_index[e] for e in edges]
        k4s.append(edge_idxs)
        
    num_k4s = len(k4s)
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edge_idxs in enumerate(k4s):
        for ei in edge_idxs:
            edge_to_k4s[ei].append(k4_idx)
    
    # Convert edge_to_k4s to tuples for faster iteration
    edge_to_k4s = [tuple(k4s_list) for k4s_list in edge_to_k4s]

    def solve(P):
        """
        Try to find a valid coloring using at most P colors.
        Returns the list of colors if found, else None.
        """
        nonlocal deadline
        
        colors = [-1] * num_edges
        k4_color_counts = [{} for _ in range(num_k4s)]
        k4_assigned = [0] * num_k4s
        
        def backtrack(edge_idx):
            if time.time() > deadline:
                return False
            
            if edge_idx == num_edges:
                return True
            
            relevant_k4s = edge_to_k4s[edge_idx]
            
            # Heuristic: Try colors that are already used in relevant K4s first?
            # This might lead to fewer distinct colors.
            # But we must ensure we don't violate the "at least 5" constraint.
            # If a K4 is close to violation, we might need a NEW color.
            
            # Let's just try 0 to P-1.
            
            for c in range(P):
                colors[edge_idx] = c
                
                undo_list = []
                valid = True
                
                for k4_idx in relevant_k4s:
                    count = k4_color_counts[k4_idx].get(c, 0)
                    k4_color_counts[k4_idx][c] = count + 1
                    undo_list.append((k4_idx, c, count))
                    
                    k4_assigned[k4_idx] += 1
                    
                    d = len(k4_color_counts[k4_idx])
                    rem = 6 - k4_assigned[k4_idx]
                    if d + rem < 5:
                        valid = False
                        break
                
                if valid:
                    if backtrack(edge_idx + 1):
                        return True
                
                # Undo
                for k4_idx, c, old_count in undo_list:
                    if old_count == 0:
                        del k4_color_counts[k4_idx][c]
                    else:
                        k4_color_counts[k4_idx][c] = old_count
                    k4_assigned[k4_idx] -= 1
                
                colors[edge_idx] = -1
                
            return False
        
        if backtrack(0):
            return colors[:]
        return None

    # Try P from 13 to 16
    for P in range(13, 17):
        if time.time() > deadline:
            break
        res = solve(P)
        if res is not None:
            # Verify
            distinct = len(set(res))
            valid = True
            for edge_idxs in k4s:
                if len(set(res[ei] for ei in edge_idxs)) < 5:
                    valid = False
                    break
            if valid:
                return res
    
    # Fallback
    colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    return colors
