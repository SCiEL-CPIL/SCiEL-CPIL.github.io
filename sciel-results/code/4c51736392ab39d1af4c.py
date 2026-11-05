import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    m = 401
    random.seed(42)
    
    # Groups: L1 (0-9), L2 (10-19), R1 (20-29), R2 (30-39)
    # L = 0-19, R = 20-39
    # Internal edges are fixed cliques K10
    # Bipartite edges between L and R
    
    # Initialize degrees in bipartite graph
    deg_L = [0] * 20 # for vertices 0-19
    deg_R = [0] * 20 # for vertices 20-39 (index 0-19 in this array, actual vertex 20+i)
    
    # adj_B[u] is set of neighbors in R (indices 0-19)
    # adj_B_R[v] is set of neighbors in L (indices 0-19)
    adj_B_L = [set() for _ in range(20)]
    adj_B_R = [set() for _ in range(20)]
    
    # Initialize bipartite graph
    # We need 221 edges
    # Greedy initialization: add edges to minimize max(deg(u)+deg(v))
    # Sort potential edges? Or just random with balancing?
    # Let's try to make it quasi-regular.
    # 221 edges / 20 verts = 11.05 avg degree.
    # So degrees should be around 11.
    
    # Create a list of all possible bipartite edges
    all_bip_edges = [(u, v) for u in range(20) for v in range(20)]
    random.shuffle(all_bip_edges)
    
    # Add edges greedily? 
    # For now, just random, then optimize.
    # But a better start helps.
    # Let's try to add edges such that degrees are balanced.
    
    edge_set_bip = set()
    
    # Target degrees: 11 or 12
    # 221 = 20 * 11 + 1. So one vertex has 12, rest 11? Or mix.
    # Let's just add random edges and let SA fix it.
    # But we can do a simple balancing pass.
    
    # Simple init:
    for e in all_bip_edges:
        if len(edge_set_bip) == 221:
            break
        u, v = e
        if (u, v) not in edge_set_bip:
            edge_set_bip.add((u, v))
            adj_B_L[u].add(v)
            adj_B_R[v].add(u)
            deg_L[u] += 1
            deg_R[v] += 1
            
    # Precompute internal clique intersection counts
    # For L1 (0-9): inter_L1[u][v] for u,v in 0..9
    # For L2 (10-19): inter_L2[u-10][v-10]
    # For R1 (20-29): inter_R1[u-20][v-20]
    # For R2 (30-39): inter_R2[u-30][v-30]
    
    def init_intersections(adj_sets, size):
        # adj_sets[i] is set of neighbors in the other partition
        # We need intersections of neighbors in the SAME partition?
        # No, for edge (u,w) in L1, triangles are from common neighbors in R.
        # So we need |N_R(u) \cap N_R(w)|.
        # N_R(u) is adj_B_L[u].
        # So inter_L1[u][w] = len(adj_B_L[u] & adj_B_L[w])
        
        inter = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(i+1, size):
                inter[i][j] = len(adj_sets[i] & adj_sets[j])
                inter[j][i] = inter[i][j]
        return inter
    
    # Note: adj_B_L[u] contains indices 0-19 (representing R vertices 20-39)
    # So for L1 vertices 0-9, their neighbors in R are adj_B_L[0]..adj_B_L[9]
    inter_L1 = init_intersections(adj_B_L[:10], 10)
    inter_L2 = init_intersections(adj_B_L[10:20], 10)
    # For R1 vertices 20-29 (idx 0-9 in R), neighbors in L are adj_B_R[0]..adj_B_R[9]
    # Edge (u,w) in R1 corresponds to u=20+i, w=20+j.
    # Common neighbors in L are adj_B_R[i] & adj_B_R[j].
    inter_R1 = init_intersections(adj_B_R[:10], 10)
    inter_R2 = init_intersections(adj_B_R[10:20], 10)
    
    def calculate_score():
        max_book = 0
        
        # 1. Bipartite edges
        for u in range(20):
            for v in adj_B_L[u]:
                # Edge (u, 20+v)
                # Book = deg_L[u] + deg_R[v]
                b = deg_L[u] + deg_R[v]
                if b > max_book:
                    max_book = b
        
        # 2. Clique edges
        # L1: 8 + inter_L1[i][j]
        for i in range(10):
            for j in range(i+1, 10):
                b = 8 + inter_L1[i][j]
                if b > max_book:
                    max_book = b
        # L2
        for i in range(10):
            for j in range(i+1, 10):
                b = 8 + inter_L2[i][j]
                if b > max_book:
                    max_book = b
        # R1
        for i in range(10):
            for j in range(i+1, 10):
                b = 8 + inter_R1[i][j]
                if b > max_book:
                    max_book = b
        # R2
        for i in range(10):
            for j in range(i+1, 10):
                b = 8 + inter_R2[i][j]
                if b > max_book:
                    max_book = b
                    
        return max_book

    current_score = calculate_score()
    best_score = current_score
    best_state = {
        'adj_B_L': [set(s) for s in adj_B_L],
        'adj_B_R': [set(s) for s in adj_B_R],
        'deg_L': list(deg_L),
        'deg_R': list(deg_R),
        'inter_L1': [row[:] for row in inter_L1],
        'inter_L2': [row[:] for row in inter_L2],
        'inter_R1': [row[:] for row in inter_R1],
        'inter_R2': [row[:] for row in inter_R2]
    }
    
    T = 2.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Choose a move: Add/Remove/Flip bipartite edge
        # To keep edge count constant, we must Add one and Remove one, or Flip?
        # Flipping an edge (u,v) -> remove (u,v), add (u', v').
        # Or just remove (u,v) and add (u', v').
        
        # Remove random edge
        if not edge_set_bip:
             break
        u_rem, v_rem = random.choice(list(edge_set_bip))
        
        # Add random non-edge
        # Sample candidates
        # We want to find (u_add, v_add) not in edge_set_bip
        # Try 20 random candidates
        best_add = None
        best_add_score = float('inf')
        
        for _ in range(20):
            u_add = random.randint(0, 19)
            v_add = random.randint(0, 19)
            if (u_add, v_add) in edge_set_bip:
                continue
            
            # Calculate delta score for adding (u_add, v_add) and removing (u_rem, v_rem)
            # This is complex to calculate exactly without applying.
            # Let's just apply, calculate, revert if worse.
            # With n=40, calculation is fast.
            
            # Apply Add
            edge_set_bip.add((u_add, v_add))
            adj_B_L[u_add].add(v_add)
            adj_B_R[v_add].add(u_add)
            deg_L[u_add] += 1
            deg_R[v_add] += 1
            
            # Apply Remove
            edge_set_bip.remove((u_rem, v_rem))
            adj_B_L[u_rem].remove(v_rem)
            adj_B_R[v_rem].remove(u_rem)
            deg_L[u_rem] -= 1
            deg_R[v_rem] -= 1
            
            # Update Intersections
            # This is the slow part.
            # When deg_L[u] changes, inter_L1 or inter_L2 involving u changes.
            # Specifically, if u in L1 (0-9):
            # For all w in L1 (w!=u), inter_L1[u][w] changes.
            # Change in inter_L1[u][w] = 1 if v_add in adj_B_L[w] (and v_rem not) etc.
            # Actually, inter_L1[u][w] = |adj_B_L[u] & adj_B_L[w]|.
            # If we add v_add to adj_B_L[u]:
            #   If v_add in adj_B_L[w], inter increases by 1.
            # If we remove v_rem from adj_B_L[u]:
            #   If v_rem in adj_B_L[w], inter decreases by 1.
            
            # Let's just recompute intersections for the affected cliques.
            # Affected L vertices: u_add, u_rem.
            # Affected R vertices: v_add, v_rem.
            
            # Recompute inter_L1 if u_add or u_rem in 0..9
            if u_add < 10 or u_rem < 10:
                # Recompute rows/cols for u_add, u_rem in inter_L1
                for u in [u_add, u_rem]:
                    if u < 10:
                        for w in range(10):
                            if u == w: continue
                            inter_L1[u][w] = len(adj_B_L[u] & adj_B_L[w])
                            inter_L1[w][u] = inter_L1[u][w]
            # Recompute inter_L2 if u_add or u_rem in 10..19
            if u_add >= 10 or u_rem >= 10:
                for u in [u_add, u_rem]:
                    if u >= 10:
                        u_idx = u - 10
                        for w in range(10):
                            if u_idx == w: continue
                            w_actual = w + 10
                            inter_L2[u_idx][w] = len(adj_B_L[u] & adj_B_L[w_actual])
                            inter_L2[w][u_idx] = inter_L2[u_idx][w]
            
            # Recompute inter_R1 if v_add or v_rem in 0..9
            if v_add < 10 or v_rem < 10:
                for v in [v_add, v_rem]:
                    if v < 10:
                        for w in range(10):
                            if v == w: continue
                            inter_R1[v][w] = len(adj_B_R[v] & adj_B_R[w])
                            inter_R1[w][v] = inter_R1[v][w]
            # Recompute inter_R2 if v_add or v_rem in 10..19
            if v_add >= 10 or v_rem >= 10:
                for v in [v_add, v_rem]:
                    if v >= 10:
                        v_idx = v - 10
                        for w in range(10):
                            if v_idx == w: continue
                            w_actual = w + 10
                            inter_R2[v_idx][w] = len(adj_B_R[v] & adj_B_R[w_actual])
                            inter_R2[w][v_idx] = inter_R2[v_idx][w]
                            
            new_score = calculate_score()
            
            if new_score < current_score or (new_score == current_score and random.random() < 0.1):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_state = {
                        'adj_B_L': [set(s) for s in adj_B_L],
                        'adj_B_R': [set(s) for s in adj_B_R],
                        'deg_L': list(deg_L),
                        'deg_R': list(deg_R),
                        'inter_L1': [row[:] for row in inter_L1],
                        'inter_L2': [row[:] for row in inter_L2],
                        'inter_R1': [row[:] for row in inter_R1],
                        'inter_R2': [row[:] for row in inter_R2]
                    }
            else:
                # Revert
                # Re-add removed
                edge_set_bip.add((u_rem, v_rem))
                adj_B_L[u_rem].add(v_rem)
                adj_B_R[v_rem].add(u_rem)
                deg_L[u_rem] += 1
                deg_R[v_rem] += 1
                
                # Remove added
                edge_set_bip.remove((u_add, v_add))
                adj_B_L[u_add].remove(v_add)
                adj_B_R[v_add].remove(u_add)
                deg_L[u_add] -= 1
                deg_R[v_add] -= 1
                
                # Revert Intersections
                if u_add < 10 or u_rem < 10:
                    for u in [u_add, u_rem]:
                        if u < 10:
                            for w in range(10):
                                if u == w: continue
                                inter_L1[u][w] = len(adj_B_L[u] & adj_B_L[w])
                                inter_L1[w][u] = inter_L1[u][w]
                if u_add >= 10 or u_rem >= 10:
                    for u in [u_add, u_rem]:
                        if u >= 10:
                            u_idx = u - 10
                            for w in range(10):
                                if u_idx == w: continue
                                w_actual = w + 10
                                inter_L2[u_idx][w] = len(adj_B_L[u] & adj_B_L[w_actual])
                                inter_L2[w][u_idx] = inter_L2[u_idx][w]
                
                if v_add < 10 or v_rem < 10:
                    for v in [v_add, v_rem]:
                        if v < 10:
                            for w in range(10):
                                if v == w: continue
                                inter_R1[v][w] = len(adj_B_R[v] & adj_B_R[w])
                                inter_R1[w][v] = inter_R1[v][w]
                if v_add >= 10 or v_rem >= 10:
                    for v in [v_add, v_rem]:
                        if v >= 10:
                            v_idx = v - 10
                            for w in range(10):
                                if v_idx == w: continue
                                w_actual = w + 10
                                inter_R2[v_idx][w] = len(adj_B_R[v] & adj_B_R[w_actual])
                                inter_R2[w][v_idx] = inter_R2[v_idx][w]
                                
            T *= cooling
            if T < T_min:
                break

    # Reconstruct final edge list
    final_edges = []
    # Add clique edges
    groups = [
        list(range(0, 10)),
        list(range(10, 20)),
        list(range(20, 30)),
        list(range(30, 40))
    ]
    for g in groups:
        for i in range(len(g)):
            for j in range(i+1, len(g)):
                final_edges.append([g[i], g[j]])
                
    # Add bipartite edges from best_state
    for u in range(20):
        for v in best_state['adj_B_L'][u]:
            # u is 0-19, v is 0-19 (represents 20+v)
            final_edges.append([u, 20 + v])
            
    return final_edges
