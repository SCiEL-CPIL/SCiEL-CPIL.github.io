import time
import random
import sys
from collections import defaultdict

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 50.0
    
    num_edges = n * (n - 1) // 2
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            edge_list.append((i, j))
            
    # Map edge to index
    edge_idx = {}
    for i, (u, v) in enumerate(edge_list):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
        
    def get_mask_edges(mask):
        # Not needed for score, just count
        return bin(mask).count("1")

    def evaluate(mask):
        # Compute degrees
        degs = [0] * n
        for i in range(num_edges):
            if mask & (1 << i):
                u, v = edge_list[i]
                degs[u] += 1
                degs[v] += 1
                
        # Check violations
        groups = defaultdict(list)
        for i in range(n):
            groups[degs[i]].append(i)
            
        violations = 0
        for deg_val, verts in groups.items():
            if len(verts) < 2: continue
            for i in range(len(verts)):
                s = verts[i]
                for j in range(i+1, len(verts)):
                    t = verts[j]
                    # Check path s-u-v-t
                    # We need to know neighbors.
                    # Reconstructing neighbors every time is slow.
                    # But N=15 is small.
                    # Let's precompute neighbors from mask?
                    pass
        return violations, degs

    # Faster check:
    # Precompute neighbors for current mask?
    
    def check_violations_fast(mask, degs):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for i in range(num_edges):
            if mask & (1 << i):
                u, v = edge_list[i]
                adj[u].add(v)
                adj[v].add(u)
                
        groups = defaultdict(list)
        for i in range(n):
            groups[degs[i]].append(i)
            
        violations = 0
        for deg_val, verts in groups.items():
            if len(verts) < 2: continue
            for i in range(len(verts)):
                s = verts[i]
                for j in range(i+1, len(verts)):
                    t = verts[j]
                    found = False
                    for u in adj[s]:
                        if u == t: continue
                        for v in adj[t]:
                            if v == s: continue
                            if u == v: continue
                            # Check if (u,v) is edge
                            if mask & (1 << edge_idx[(u, v) if u < v else (v, u)]):
                                found = True
                                break
                        if found: break
                    if found:
                        violations += 1
        return violations

    # SA Loop
    current_mask = 0
    for i in range(num_edges):
        if random.random() < 0.5:
            current_mask |= (1 << i)
            
    curr_degs = [0]*n
    for i in range(num_edges):
        if current_mask & (1 << i):
            u, v = edge_list[i]
            curr_degs[u] += 1
            curr_degs[v] += 1
            
    curr_violations = check_violations_fast(current_mask, curr_degs)
    curr_score = bin(current_mask).count("1") * 100 - curr_violations * 1000
    
    best_mask = current_mask
    best_score = curr_score
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Flip random bit
        i = random.randint(0, num_edges - 1)
        new_mask = current_mask ^ (1 << i)
        
        # Update degs
        u, v = edge_list[i]
        if current_mask & (1 << i):
            new_degs = curr_degs[:]
            new_degs[u] -= 1
            new_degs[v] -= 1
        else:
            new_degs = curr_degs[:]
            new_degs[u] += 1
            new_degs[v] += 1
            
        new_violations = check_violations_fast(new_mask, new_degs)
        new_score = bin(new_mask).count("1") * 100 - new_violations * 1000
        
        delta = new_score - curr_score
        if delta > 0 or random.random() < pow(2.718, delta / max(T, 0.1)):
            current_mask = new_mask
            curr_degs = new_degs
            curr_violations = new_violations
            curr_score = new_score
            
            if curr_score > best_score:
                best_score = curr_score
                best_mask = current_mask
                
        T *= cooling
        
    # Extract edges
    res = []
    for i in range(num_edges):
        if best_mask & (1 << i):
            res.append(list(edge_list[i]))
            
    return res
