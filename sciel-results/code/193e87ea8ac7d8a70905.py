import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_n = 0
    best_edges = []

    # Try n=47 first, then 46, then 45...
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        if min_deg_needed < 0:
            min_deg_needed = 0
            
        found = False
        
        # Number of restarts depends on time available
        max_restarts = 500
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            random.seed(restart * 100 + target_n)
            
            # Bitset representation of neighbors
            N = [0] * target_n
            deg = [0] * target_n
            
            def can_add(u, v):
                # Check if u and v are already connected
                if (N[u] >> v) & 1:
                    return False
                
                nu = N[u]
                nv = N[v]
                
                # Check if there is any edge between N(u) and N(v)
                # Iterate over the smaller neighbor set
                if deg[u] > deg[v]:
                    temp = nv
                    while temp:
                        lsb = temp & -temp
                        b = lsb.bit_length() - 1
                        if N[b] & nu:
                            return False
                        temp ^= lsb
                else:
                    temp = nu
                    while temp:
                        lsb = temp & -temp
                        a = lsb.bit_length() - 1
                        if N[a] & nv:
                            return False
                        temp ^= lsb
                            
                return True

            def add_edge(u, v):
                N[u] |= (1 << v)
                N[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1

            def remove_edge(u, v):
                N[u] &= ~(1 << v)
                N[v] &= ~(1 << u)
                deg[u] -= 1
                deg[v] -= 1

            success = True
            removals = 0
            max_removals = 50
            
            for _ in range(5000): # Max steps
                if time.time() > deadline:
                    success = False
                    break
                    
                # Check if we are done
                if min(deg) >= min_deg_needed:
                    success = True
                    break

                low_vertices = [i for i in range(target_n) if deg[i] < min_deg_needed]
                if not low_vertices:
                    success = True
                    break
                    
                u = random.choice(low_vertices)
                
                # Find a valid v to connect to u
                candidates = []
                for v in range(target_n):
                    if v == u:
                        continue
                    if can_add(u, v):
                        candidates.append(v)
                        
                if not candidates:
                    # Stuck. Try to remove a random edge to unblock?
                    if removals < max_removals:
                        all_edges = []
                        for i in range(target_n):
                            if deg[i] > 0:
                                temp = N[i]
                                while temp:
                                    lsb = temp & -temp
                                    j = lsb.bit_length() - 1
                                    if i < j:
                                        all_edges.append((i, j))
                                    temp ^= lsb
                        if all_edges:
                            e_u, e_v = random.choice(all_edges)
                            remove_edge(e_u, e_v)
                            removals += 1
                            continue
                    else:
                        success = False
                        break
                            
                v = random.choice(candidates)
                add_edge(u, v)
                
            if success:
                # Final verification
                c4_free = True
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if (N[i] & N[j]).bit_count() > 1:
                            c4_free = False
                            break
                    if not c4_free:
                        break
                
                if c4_free and min(deg) >= min_deg_needed:
                    final_edges = []
                    for i in range(target_n):
                        temp = N[i]
                        while temp:
                            lsb = temp & -temp
                            j = lsb.bit_length() - 1
                            if i < j:
                                final_edges.append([i, j])
                            temp ^= lsb
                    
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = final_edges
                    found = True
                    break # Break restart loop
        
        if best_n == 47:
            break
        if found:
            break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback: n=39, empty graph is valid (min deg 0 >= 0)
        return {'n': 39, 'edges': []}
