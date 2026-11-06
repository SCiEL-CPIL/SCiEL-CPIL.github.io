import time
import random
import sys
from itertools import combinations

def construct():
    # Set a hard deadline with safety margin
    deadline = time.time() + 285.0
    
    # 1. Generate Paley Graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    n = q
    
    # Helper to get K4s and Alpha4s (K4 in complement)
    def get_k4s(adj, n):
        k4s = []
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n):
                    if not ((adj[a] >> c) & 1 and (adj[b] >> c) & 1):
                        continue
                    for d in range(c + 1, n):
                        if ((adj[a] >> d) & 1 and (adj[b] >> d) & 1 and (adj[c] >> d) & 1):
                            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            k4s.append(mask)
        return k4s

    def get_alpha4s(adj, n):
        alpha4s = []
        for a in range(n):
            for b in range(a + 1, n):
                if (adj[a] >> b) & 1: # If edge exists, not in independent set
                    continue
                for c in range(b + 1, n):
                    if ((adj[a] >> c) & 1 or (adj[b] >> c) & 1):
                        continue
                    for d in range(c + 1, n):
                        if not ((adj[a] >> d) & 1 or (adj[b] >> d) & 1 or (adj[c] >> d) & 1):
                            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            alpha4s.append(mask)
        return alpha4s

    # Initial lists
    k4_list = get_k4s(adj, n)
    alpha4_list = get_alpha4s(adj, n)
    
    # Helper to build maps for fast incremental updates
    def build_maps(k4_list, alpha4_list, n):
        k4_by_vertex = [[] for _ in range(n)]
        alpha4_by_vertex = [[] for _ in range(n)]
        for k4 in k4_list:
            m = k4
            while m:
                lsb = m & -m
                idx = (lsb.bit_length() - 1)
                k4_by_vertex[idx].append(k4)
                m ^= lsb
        for a4 in alpha4_list:
            m = a4
            while m:
                lsb = m & -m
                idx = (lsb.bit_length() - 1)
                alpha4_by_vertex[idx].append(a4)
                m ^= lsb
        return k4_by_vertex, alpha4_by_vertex

    k4_by_vertex, alpha4_by_vertex = build_maps(k4_list, alpha4_list, n)
    
    # Optimized search for a valid neighbor set for a new vertex
    # Uses incremental score updates based on precomputed maps
    def find_valid_N_opt(n, k4_list, alpha4_list, k4_by_vertex, alpha4_by_vertex, time_limit=5.0):
        start_time = time.time()
        best_score = float('inf')
        best_N = None
        full_mask = (1 << n) - 1
        
        # Try multiple random restarts with adaptive time allocation
        max_restarts = 50
        restart_count = 0
        while restart_count < max_restarts and time.time() - start_time < time_limit:
            # Allocate time per restart based on remaining time and expected restarts
            remaining_time = time_limit - (time.time() - start_time)
            expected_restarts_left = max(1, min(10, max_restarts - restart_count))
            time_per_restart = remaining_time / expected_restarts_left
            
            # Initialize N with bias towards existing edges (heuristic)
            N = 0
            for i in range(n):
                # Slight bias: if vertex i has many neighbors in current graph, more likely to connect
                deg_i = bin(adj[i]).count('1')
                prob = 0.4 + 0.2 * (deg_i / n) if n > 0 else 0.5
                if random.random() < prob:
                    N |= (1 << i)
            
            # Calculate initial score
            current_score = 0
            for k4 in k4_list:
                if (N & k4) == k4:
                    current_score += 1
            not_N = full_mask & ~N
            for a4 in alpha4_list:
                if (not_N & a4) == a4:
                    current_score += 1
            
            if current_score == 0:
                return N
            
            if current_score < best_score:
                best_score = current_score
                best_N = N
            
            # Simulated annealing with faster cooling
            T = 2.0
            T_min = 0.1
            cooling_rate = 0.97
            local_start = time.time()
            
            while time.time() - local_start < time_per_restart and current_score > 0:
                if T < T_min:
                    break
                
                # Pick a random bit to flip
                bit = random.randint(0, n - 1)
                delta_bit = 1 << bit
                new_N = N ^ delta_bit
                
                # Incremental update
                added = (new_N & delta_bit) != 0
                delta_score = 0
                
                if added:
                    # Check K4s containing bit that are now fully in N
                    for k4 in k4_by_vertex[bit]:
                        if (new_N & k4) == k4 and (N & k4) != k4:
                            delta_score += 1
                    # Check Alpha4s containing bit that are now fully in N (so broken in ~N)
                    for a4 in alpha4_by_vertex[bit]:
                        if (N & a4) == 0:
                            delta_score -= 1
                else:
                    # Removing bit
                    # K4s: If bit was in N and in k4, and all other bits of k4 were in N, then it was a violation. Now it's not.
                    for k4 in k4_by_vertex[bit]:
                        if (N & k4) == k4 and (new_N & k4) != k4:
                            delta_score -= 1
                    # Alpha4s: If we remove bit from N (add to ~N), we might complete an Alpha4.
                    for a4 in alpha4_by_vertex[bit]:
                        if (new_N & a4) == 0 and (N & a4) == delta_bit:
                            delta_score += 1
                
                new_score = current_score + delta_score
                
                if new_score < 0: new_score = 0 # Safety
                
                if new_score < current_score or random.random() < pow(2.718, -(new_score - current_score) / T):
                    N = new_N
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_N = N
                T *= cooling_rate
            
            if best_score == 0:
                return best_N
            restart_count += 1

        return best_N if best_score == 0 else None

    # Extension loop
    for target_n in range(n + 1, 47):
        if time.time() > deadline - 20.0:
            break
        
        # Adaptive time budget per vertex: more time for later vertices (harder)
        remaining_vertices = 47 - target_n
        time_per_vertex = 3.0 + 2.0 * (target_n - 37) / 9.0
        if target_n > 42:
            time_per_vertex = 6.0
        
        found_N = find_valid_N_opt(n, k4_list, alpha4_list, k4_by_vertex, alpha4_by_vertex, time_per_vertex)
        
        if found_N is None:
            # Try with increased time budget
            if time.time() < deadline - 25.0:
                found_N = find_valid_N_opt(n, k4_list, alpha4_list, k4_by_vertex, alpha4_by_vertex, time_per_vertex * 1.5)
            if found_N is None:
                break
        
        # Add the new vertex
        new_vertex = n
        adj.append(0)
        for i in range(n):
            if (found_N >> i) & 1:
                adj[new_vertex] |= (1 << i)
                adj[i] |= (1 << new_vertex)
        
        n += 1
        
        # Update k4_list and alpha4_list
        new_k4s = []
        neigh_mask = adj[new_vertex]
        # Iterate over triples in neigh_mask
        for a in range(n - 1):
            if not (neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (neigh_mask >> b) & 1:
                    continue
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n - 1):
                    if not (neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) and ((adj[b] >> c) & 1):
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                        new_k4s.append(mask)
        
        k4_list.extend(new_k4s)
        
        new_alpha4s = []
        non_neigh_mask = ((1 << (n - 1)) - 1) & ~neigh_mask
        for a in range(n - 1):
            if not (non_neigh_mask >> a) & 1:
                continue
            for b in range(a + 1, n - 1):
                if not (non_neigh_mask >> b) & 1:
                    continue
                if (adj[a] >> b) & 1: # Edge exists, so not independent
                    continue
                for c in range(b + 1, n - 1):
                    if not (non_neigh_mask >> c) & 1:
                        continue
                    if ((adj[a] >> c) & 1) or ((adj[b] >> c) & 1):
                        continue
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << new_vertex)
                    new_alpha4s.append(mask)
        
        alpha4_list.extend(new_alpha4s)
        
        # Update maps
        k4_by_vertex, alpha4_by_vertex = build_maps(k4_list, alpha4_list, n)

    # Convert adj to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append((i, j))
    
    return n, edges
