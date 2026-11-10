import time
import random
import sys

def construct():
    N = 40
    deadline = time.time() + 115  # Safety margin
    
    # Bitset representation of adjacency
    # adj[u] is an integer where bit v is 1 if (u, v) is an edge
    adj = [0] * N
    
    def get_edges():
        edges = []
        for u in range(N):
            mask = adj[u]
            while mask:
                v_bit = mask & -mask
                v = v_bit.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask -= v_bit
        return edges

    def can_add(u, v):
        # Check if adding (u, v) creates a C6
        # Path: u - a - b - c - d - v
        # a in N(u)\{v}, b in N(a)\{u,v}, c in N(b)\{a,v,u}, d in N(c)\{b,v,u,a} and d in N(v)
        
        # S1: neighbors of u, excluding v
        S1 = adj[u] & ~(1 << v)
        
        # S2: nodes at distance 2 from u, avoiding v and u
        S2 = 0
        mask = S1
        while mask:
            a_bit = mask & -mask
            a = a_bit.bit_length() - 1
            # Neighbors of a, excluding u and v
            S2 |= adj[a] & ~(1 << u) & ~(1 << v)
            mask -= a_bit
            
        # S3: nodes at distance 3 from u, avoiding v, u, and S1
        S3 = 0
        mask = S2
        while mask:
            b_bit = mask & -mask
            b = b_bit.bit_length() - 1
            # Neighbors of b, excluding v, u, and S1
            S3 |= adj[b] & ~(1 << v) & ~(1 << u) & ~S1
            mask -= b_bit
            
        # S4: nodes at distance 4 from u, avoiding v, u, S1, S2
        S4 = 0
        mask = S3
        while mask:
            c_bit = mask & -mask
            c = c_bit.bit_length() - 1
            # Neighbors of c, excluding v, u, S1, S2
            S4 |= adj[c] & ~(1 << v) & ~(1 << u) & ~S1 & ~S2
            mask -= c_bit
            
        # Check if any node in S4 is a neighbor of v
        if S4 & adj[v]:
            return False
        return True

    # Initial Graph: 8 disjoint K5s
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    best_edges_count = sum(bin(adj[u]).count('1') for u in range(N)) // 2
    best_adj = adj[:]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < deadline:
        # Generate list of missing edges
        missing = []
        for u, v in all_pairs:
            if not (adj[u] & (1 << v)):
                missing.append((u, v))
                
        if not missing:
            break
            
        # Heuristic: Sort by degree sum (higher degree vertices first)
        # This tends to fill in dense areas
        missing.sort(key=lambda x: (bin(adj[x[0]]).count('1') + bin(adj[x[1]]).count('1')), reverse=True)
        
        added_in_pass = 0
        for u, v in missing:
            if time.time() > deadline:
                break
            if not (adj[u] & (1 << v)):
                if can_add(u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added_in_pass += 1
                    current_edges = sum(bin(adj[i]).count('1') for i in range(N)) // 2
                    if current_edges > best_edges_count:
                        best_edges_count = current_edges
                        best_adj = adj[:]
                        
        if added_in_pass == 0:
            # Stuck. Perform a kick: remove 1-2 random edges
            current_edges = get_edges()
            if current_edges:
                kick_size = random.randint(1, 2)
                for _ in range(kick_size):
                    if not current_edges:
                        break
                    idx = random.randint(0, len(current_edges) - 1)
                    u, v = current_edges[idx]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_edges.pop(idx)

    # Return the best graph found
    final_edges = []
    for u in range(N):
        mask = best_adj[u]
        while mask:
            v_bit = mask & -mask
            v = v_bit.bit_length() - 1
            if u < v:
                final_edges.append([u, v])
            mask -= v_bit
            
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
