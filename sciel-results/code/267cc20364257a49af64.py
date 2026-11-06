import time
import random

def construct():
    deadline = time.time() + 110
    n = 40
    adj = [0] * n
    masks = [1 << i for i in range(n)]
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def has_c4_in_subset(mask):
        """Check if induced subgraph on vertices in mask contains a C4."""
        vertices = []
        m = mask
        while m:
            bit = m & -m
            v = bit.bit_length() - 1
            vertices.append(v)
            m ^= bit
        
        k = len(vertices)
        if k < 4:
            return False
        
        # Check all 4-tuples for C4
        for i in range(k):
            for j in range(i + 1, k):
                for l in range(j + 1, k):
                    for r in range(l + 1, k):
                        a, b, c, d = vertices[i], vertices[j], vertices[l], vertices[r]
                        # Check if a-b-c-d-a is a cycle
                        ab = (adj[a] & masks[b]) != 0
                        bc = (adj[b] & masks[c]) != 0
                        cd = (adj[c] & masks[d]) != 0
                        da = (adj[d] & masks[a]) != 0
                        ac = (adj[a] & masks[c]) != 0
                        bd = (adj[b] & masks[d]) != 0
                        # C4: a-b-c-d-a with no diagonals
                        if ab and bc and cd and da and not ac and not bd:
                            return True
        return False

    def has_c6_with_edge(u, v):
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
                    
                    d_mask = adj[c] & adj[v]
                    forbidden = masks[u] | masks[a] | masks[b] | masks[c]
                    if d_mask & ~forbidden:
                        return True
        return False

    def add_edge(u, v):
        adj[u] |= masks[v]
        adj[v] |= masks[u]
        
    def remove_edge(u, v):
        adj[u] &= ~masks[v]
        adj[v] &= ~masks[u]

    def count_edges():
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def greedy_fill():
        random.shuffle(all_edges)
        for u, v in all_edges:
            if (adj[u] & masks[v]) == 0:
                if not has_c6_with_edge(u, v):
                    add_edge(u, v)

    def structural_prune():
        """Remove edges that cause C4 in common neighborhoods of non-adjacent pairs."""
        changed = True
        while changed:
            changed = False
            # For each non-adjacent pair, check if their common neighborhood has C4
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] & masks[j]) == 0:
                        common = adj[i] & adj[j] & ~masks[i] & ~masks[j]
                        if has_c4_in_subset(common):
                            # Find which edge to remove - try removing random edges from common neighborhood
                            # This is a heuristic; we'll just mark that we need to remove something
                            # For now, just return False to indicate pruning needed
                            pass
        return True

    # Init with random seed
    for k in range(8):
        start = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                add_edge(start + i, start + j)
                
    greedy_fill()
    best_edges = count_edges()
    best_adj = adj[:]
    
    while time.time() < deadline:
        improved = False
        
        # Try 1-remove/2-add swaps
        for _ in range(20):
            if time.time() > deadline:
                break
                
            # Pick random edge to remove
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (adj[u] & masks[v]) == 0:
                continue
                
            # Pick two random non-edges to add
            x1, y1 = u, v
            x2, y2 = u, v
            attempts = 0
            while attempts < 10:
                x1 = random.randint(0, n - 1)
                y1 = random.randint(0, n - 1)
                if x1 != y1 and (adj[x1] & masks[y1]) == 0:
                    break
                attempts += 1
            
            attempts = 0
            while attempts < 10:
                x2 = random.randint(0, n - 1)
                y2 = random.randint(0, n - 1)
                if x2 != y2 and (adj[x2] & masks[y2]) == 0 and not (x2 == x1 and y2 == y1):
                    break
                attempts += 1
            
            if attempts >= 10:
                continue
            
            # Remove edge (u,v)
            remove_edge(u, v)
            
            # Try adding both edges
            valid1 = not has_c6_with_edge(x1, y1)
            if valid1:
                add_edge(x1, y1)
                valid2 = not has_c6_with_edge(x2, y2)
                if valid2:
                    add_edge(x2, y2)
                    greedy_fill()
                    new_count = count_edges()
                    if new_count > best_edges:
                        best_edges = new_count
                        best_adj = adj[:]
                        improved = True
                        break
                    else:
                        remove_edge(x2, y2)
                        remove_edge(x1, y1)
                else:
                    remove_edge(x1, y1)
            # Revert removal
            add_edge(u, v)
            
            if improved:
                break
        
        if not improved:
            # Perturb: remove 2-3 edges and greedy fill
            for _ in range(3):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v and (adj[u] & masks[v]):
                    remove_edge(u, v)
            greedy_fill()
            if count_edges() > best_edges:
                best_edges = count_edges()
                best_adj = adj[:]

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] & masks[j]):
                edges.append([i, j])
    return edges
