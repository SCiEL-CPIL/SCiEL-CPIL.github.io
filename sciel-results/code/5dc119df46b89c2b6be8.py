import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 40
    best_edges = []
    best_count = 0
    
    # Bitmasks for adjacency and distance-2 neighbors
    # adj[i] is a bitmask of neighbors of i
    # mask2[i] is a bitmask of vertices at distance <= 2 from i (excluding i? usually just distance 2, but let's say neighbors of neighbors)
    # For the check: path u-a-b-c-d-v.
    # a in N(u), d in N(v).
    # Need path a-b-c-d of length 3.
    # This exists if N(b) intersects N(d) for some b in N(a).
    # Equivalent to: Union of N(b) for b in N(a) intersects N(d).
    # Let Mask[a] = Union of N(b) for b in N(a).
    # Then check if Mask[a] & N(d) is non-zero.
    
    def check_c6(u, v, adj, mask2):
        # Check if adding (u,v) creates a C6.
        # This happens if there is a path of length 5 between u and v.
        # Path: u - a - b - c - d - v
        # a in N(u), d in N(v)
        # b in N(a), c in N(d), (b,c) in E
        # This is equivalent to: exists a in N(u), d in N(v) such that Mask[a] & N(d) != 0
        # Note: Mask[a] contains neighbors of neighbors of a.
        # If x in Mask[a] & N(d), then x is neighbor of some b in N(a) and x is neighbor of d.
        # So a-b-x-d is a path of length 3.
        # Then u-a-b-x-d-v is a path of length 5.
        # Adding v-u closes the cycle.
        
        # Optimization: iterate over the smaller set
        nu = adj[u]
        nv = adj[v]
        
        # Get list of neighbors for iteration
        # Since n=40, we can just loop 0..39 and check bits
        # Or use bit tricks to iterate set bits.
        
        # For each a in N(u)
        a = 0
        while nu:
            if nu & 1:
                # a is a neighbor
                # Check if Mask[a] intersects N(v)
                if mask2[a] & nv:
                    return True
            nu >>= 1
            a += 1
            
        return False

    def add_edge(u, v, adj, mask2):
        # Add edge (u,v)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update mask2
        # mask2[x] should contain all vertices at distance 2 from x.
        # Adding (u,v):
        # 1. v is distance 2 from all neighbors of u.
        # 2. u is distance 2 from all neighbors of v.
        # 3. All neighbors of v are distance 2 from u.
        # 4. All neighbors of u are distance 2 from v.
        
        # Update mask2[u] and mask2[v]
        mask2[u] |= adj[v]
        mask2[v] |= adj[u]
        
        # Update mask2 for neighbors of u
        nu = adj[u] & ~(1 << v) # Exclude v? v is already handled? 
        # Actually, adj[u] now includes v.
        # Neighbors of u are all bits in adj[u].
        # For each z in N(u), v is now distance 2 from z.
        z = 0
        temp = adj[u]
        while temp:
            if temp & 1:
                mask2[z] |= (1 << v)
            temp >>= 1
            z += 1
            
        # Update mask2 for neighbors of v
        nv = adj[v]
        z = 0
        temp = nv
        while temp:
            if temp & 1:
                mask2[z] |= (1 << u)
            temp >>= 1
            z += 1

    def greedy_build(order):
        adj = [0] * n
        mask2 = [0] * n
        edges = []
        
        for u, v in order:
            if (adj[u] >> v) & 1:
                continue
            if check_c6(u, v, adj, mask2):
                continue
            add_edge(u, v, adj, mask2)
            edges.append((u, v))
            
        return edges

    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Multiple random restarts
    # Also try a "dense" start: K5s?
    # The prompt says "Keep this design... investigate improvements".
    # The seed was 8 K5s.
    # Let's try starting with a random sparse graph and greedily adding.
    # Or just greedy from empty with random order.
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        random.shuffle(all_edges)
        # To encourage density, we might prefer edges between low-degree vertices?
        # Or just random.
        # Let's try sorting by some heuristic?
        # Random shuffle is a good baseline.
        
        edges = greedy_build(all_edges)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
        # If we hit a high score, we can try local search or more restarts
        # But greedy is fast.
        
    # Final check: ensure best_edges is valid
    # (It should be, as we check before adding)
    
    # Convert to list of lists as required
    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Edges: {len(res)}")
