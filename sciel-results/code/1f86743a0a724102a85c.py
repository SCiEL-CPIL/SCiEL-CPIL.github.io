import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin
    
    # Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [0] * n
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
    
    best_n = n
    best_edges = edges[:]
    
    def get_k4s_and_ik4s():
        k4s = []
        ik4s = []
        for quad in combinations(range(n), 4):
            is_k4 = True
            is_ik4 = True
            for u, v in combinations(quad, 2):
                if adj[u] & (1 << v):
                    is_ik4 = False
                else:
                    is_k4 = False
            if is_k4:
                k4s.append(quad)
            if is_ik4:
                ik4s.append(quad)
        return k4s, ik4s
    
    k4s, ik4s = get_k4s_and_ik4s()
    num_k4 = len(k4s)
    num_ik4 = len(ik4s)
    
    # Pre-compute incidence: for each vertex, which k4s and ik4s it's in
    vertex_k4s = [[] for _ in range(n)]
    vertex_ik4s = [[] for _ in range(n)]
    for idx, quad in enumerate(k4s):
        for u in quad:
            vertex_k4s[u].append(idx)
    for idx, quad in enumerate(ik4s):
        for u in quad:
            vertex_ik4s[u].append(idx)
    
    def try_add_vertex():
        """Find a valid connection vector for a new vertex using local search"""
        best_x = None
        best_viol = float('inf')
        
        # Try multiple random restarts
        for restart in range(100):
            if time.time() > deadline - 5:
                break
            
            # Initialize
            if restart == 0:
                x = [0] * n
            elif restart == 1:
                x = [1] * n
            elif restart < 10:
                x = [random.randint(0, 1) for _ in range(n)]
            else:
                # Biased random
                p = 0.3 + 0.4 * random.random()
                x = [1 if random.random() < p else 0 for _ in range(n)]
            
            # Compute initial constraint sums
            k4_sums = [0] * num_k4
            ik4_sums = [0] * num_ik4
            for idx, quad in enumerate(k4s):
                s = 0
                for u in quad:
                    s += x[u]
                k4_sums[idx] = s
            for idx, quad in enumerate(ik4s):
                s = 0
                for u in quad:
                    s += x[u]
                ik4_sums[idx] = s
            
            def cur_violation():
                v = 0
                for s in k4_sums:
                    if s >= 4:
                        v += s - 3
                for s in ik4_sums:
                    if s <= 0:
                        v += 1
                return v
            
            cur_viol = cur_violation()
            if cur_viol == 0:
                return x[:]
            
            # Local search
            improved = True
            for it in range(3000):
                if time.time() > deadline - 5:
                    break
                if not improved:
                    break
                improved = False
                best_flip = -1
                best_new_viol = cur_viol
                
                # Try flipping each bit, compute delta efficiently
                for u in range(n):
                    old_val = x[u]
                    new_val = 1 - old_val
                    delta = 0
                    
                    # Update k4 sums
                    for k4_idx in vertex_k4s[u]:
                        old_s = k4_sums[k4_idx]
                        new_s = old_s + (new_val - old_val)
                        old_v = max(0, old_s - 3)
                        new_v = max(0, new_s - 3)
                        delta += new_v - old_v
                    
                    # Update ik4 sums
                    for ik4_idx in vertex_ik4s[u]:
                        old_s = ik4_sums[ik4_idx]
                        new_s = old_s + (new_val - old_val)
                        old_v = 1 if old_s <= 0 else 0
                        new_v = 1 if new_s <= 0 else 0
                        delta += new_v - old_v
                    
                    new_viol = cur_viol + delta
                    if new_viol < best_new_viol:
                        best_new_viol = new_viol
                        best_flip = u
                
                if best_flip >= 0 and best_new_viol < cur_viol:
                    u = best_flip
                    old_val = x[u]
                    new_val = 1 - old_val
                    x[u] = new_val
                    
                    # Update sums
                    for k4_idx in vertex_k4s[u]:
                        k4_sums[k4_idx] += (new_val - old_val)
                    for ik4_idx in vertex_ik4s[u]:
                        ik4_sums[ik4_idx] += (new_val - old_val)
                    
                    cur_viol = best_new_viol
                    improved = True
                    if cur_viol == 0:
                        return x[:]
                else:
                    # Stuck: random perturbation
                    if random.random() < 0.5:
                        u = random.randint(0, n - 1)
                        old_val = x[u]
                        new_val = 1 - old_val
                        x[u] = new_val
                        for k4_idx in vertex_k4s[u]:
                            k4_sums[k4_idx] += (new_val - old_val)
                        for ik4_idx in vertex_ik4s[u]:
                            ik4_sums[ik4_idx] += (new_val - old_val)
                        cur_viol = cur_violation()
                        improved = True
            
            if cur_viol < best_viol:
                best_viol = cur_viol
                best_x = x[:]
        
        if best_viol == 0:
            return best_x
        return None
    
    # Main loop: try to add vertices
    attempts = 0
    while time.time() < deadline and n < 42:
        x = try_add_vertex()
        if x is not None:
            new_v = n
            adj.append(0)
            for u in range(n):
                if x[u]:
                    adj[new_v] |= (1 << u)
                    adj[u] |= (1 << new_v)
                    edges.append((u, new_v))
            n += 1
            if n > best_n:
                best_n = n
                best_edges = edges[:]
            k4s, ik4s = get_k4s_and_ik4s()
            num_k4 = len(k4s)
            num_ik4 = len(ik4s)
            vertex_k4s = [[] for _ in range(n)]
            vertex_ik4s = [[] for _ in range(n)]
            for idx, quad in enumerate(k4s):
                for u in quad:
                    vertex_k4s[u].append(idx)
            for idx, quad in enumerate(ik4s):
                for u in quad:
                    vertex_ik4s[u].append(idx)
            attempts = 0
        else:
            attempts += 1
            if attempts > 5:
                break
    
    return best_n, best_edges
