import time
import random

def construct():
    deadline = time.time() + 115
    n = 40
    adj = [0] * n
    masks = [1 << i for i in range(n)]
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    non_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def popcount(x):
        return bin(x).count('1')
    
    def has_c6_with_edge(u, v):
        # Check if adding (u,v) creates a C6
        # Path: u - a - b - c - d - v
        # Try from the vertex with fewer neighbors first
        nu = popcount(adj[u])
        nv = popcount(adj[v])
        if nu > nv:
            u, v = v, u
        
        a_mask = adj[u] & ~masks[v]
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            b_mask = adj[a] & ~masks[u] & ~masks[v]
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                c_mask = adj[b] & ~masks[u] & ~masks[a] & ~masks[v]
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    # d must be in N(c) ∩ N(v), excluding u,a,b,c
                    d_mask = adj[c] & adj[v] & ~(masks[u] | masks[a] | masks[b] | masks[c])
                    if d_mask:
                        return True
        return False
    
    def add_edge(u, v):
        adj[u] |= masks[v]
        adj[v] |= masks[u]
        
    def remove_edge(u, v):
        adj[u] &= ~masks[v]
        adj[v] &= ~masks[u]

    def count_edges():
        return sum(popcount(adj[i]) for i in range(n)) // 2

    def greedy_fill():
        # Try to add all possible edges, prioritizing those that are "easy"
        candidates = []
        for u, v in all_edges:
            if (adj[u] & masks[v]) == 0:
                candidates.append((u, v))
        
        # Sort by a heuristic: prefer edges between vertices with lower degree
        # This tends to fill in "sparse" areas first
        candidates.sort(key=lambda e: popcount(adj[e[0]]) + popcount(adj[e[1]]))
        
        for u, v in candidates:
            if (adj[u] & masks[v]) == 0:  # Re-check in case earlier additions changed things
                if not has_c6_with_edge(u, v):
                    add_edge(u, v)

    def get_non_edges():
        return [(i, j) for i in range(n) for j in range(i + 1, n) if (adj[i] & masks[j]) == 0]
    
    def get_edges_list():
        return [(i, j) for i in range(n) for j in range(i + 1, n) if (adj[i] & masks[j])]
    
    def score_non_edge(u, v):
        """Score how 'desirable' it is to add edge (u,v). Lower score = more desirable.
        A non-edge is more desirable if adding it would create fewer C6s."""
        # Count potential C6 obstructions: number of paths of length 5 from u to v
        # This is expensive, so use a cheaper proxy: sum of degrees
        return popcount(adj[u]) + popcount(adj[v])
    
    def local_search():
        improved = True
        while improved and time.time() < deadline:
            improved = False
            non_edges = get_non_edges()
            if not non_edges:
                break
            
            # Try each non-edge, sorted by desirability
            non_edges.sort(key=lambda e: score_non_edge(e[0], e[1]))
            
            for u, v in non_edges[:50]:  # Limit to top 50 candidates
                if time.time() > deadline:
                    break
                if (adj[u] & masks[v]) != 0:
                    continue
                if not has_c6_with_edge(u, v):
                    add_edge(u, v)
                    improved = True
                    break  # Restart with new state
            
            if not improved:
                # Try 1-remove/1-add swaps
                edges = get_edges_list()
                random.shuffle(edges)
                for eu, ev in edges[:30]:
                    if time.time() > deadline:
                        break
                    remove_edge(eu, ev)
                    added = False
                    for nu, nv in non_edges[:30]:
                        if (adj[nu] & masks[nv]) == 0 and not has_c6_with_edge(nu, nv):
                            add_edge(nu, nv)
                            added = True
                            break
                    if not added:
                        add_edge(eu, ev)
                    else:
                        improved = True
                        break

    def perturb():
        """Remove a few edges to break local structure."""
        edges = get_edges_list()
        if not edges:
            return
        # Remove 2-4 random edges
        k = random.randint(2, 4)
        random.shuffle(edges)
        for eu, ev in edges[:k]:
            remove_edge(eu, ev)
        greedy_fill()

    # Try multiple initial structures
    best_edges = 0
    best_adj = [0] * n
    
    # Structure 1: 8 disjoint K5s
    for k in range(8):
        start = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                add_edge(start + i, start + j)
    
    greedy_fill()
    local_search()
    if count_edges() > best_edges:
        best_edges = count_edges()
        best_adj = adj[:]
    
    # Try more perturbations and restarts
    attempts = 0
    while time.time() < deadline - 5 and attempts < 50:
        attempts += 1
        perturb()
        local_search()
        if count_edges() > best_edges:
            best_edges = count_edges()
            best_adj = adj[:]
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] & masks[j]):
                edges.append([i, j])
    return edges
