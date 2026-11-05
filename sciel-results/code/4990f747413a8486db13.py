import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    # Split vertices into two sets A and B
    A = list(range(10))
    B = list(range(10, 20))
    
    # We will try to construct a subgraph of K_{10,10}
    # Vertices are 0..19. A={0..9}, B={10..19}
    
    best_edges = []
    best_score = 0
    
    # Try several coloring strategies
    strategies = [
        lambda a, b: (a + b) % 20,
        lambda a, b: (a * 2 + b) % 20,
        lambda a, b: (a * 3 + b) % 20,
        lambda a, b: (a + 2 * b) % 20,
        lambda a, b: (a * b) % 20,
        lambda a, b: (a + b) % 10 + 10 * ((a * b) % 2),
    ]
    
    # Add random permutations to strategies
    for _ in range(5):
        perm_a = list(range(10))
        perm_b = list(range(10))
        random.shuffle(perm_a)
        random.shuffle(perm_b)
        strategies.append(lambda a, b, pa=perm_a, pb=perm_b: (pa[a] + pb[b-10]) % 20)
        
    for color_func in strategies:
        if time.time() > deadline:
            break
            
        # Initialize edges with colors
        edges = []
        # Map (u,v) to color
        edge_colors = {}
        
        for u in A:
            for v in B:
                c = color_func(u, v)
                edges.append([u, v, c])
                edge_colors[(u, v)] = c
                edge_colors[(v, u)] = c
                
        # Check all 4-cycles
        # A 4-cycle in K_{10,10} is u1-v1-u2-v2-u1 with u1,u2 in A, v1,v2 in B
        invalid_cycles = []
        
        for i, u1 in enumerate(A):
            for u2 in A[i+1:]:
                for j, v1 in enumerate(B):
                    for v2 in B[j+1:]:
                        # Cycle u1-v1-u2-v2-u1
                        c1 = edge_colors[(u1, v1)]
                        c2 = edge_colors[(u2, v1)]
                        c3 = edge_colors[(u2, v2)]
                        c4 = edge_colors[(u1, v2)]
                        
                        if len({c1, c2, c3, c4}) < 4:
                            # Non-rainbow cycle
                            # Edges involved: (u1,v1), (u2,v1), (u2,v2), (u1,v2)
                            invalid_cycles.append([(u1, v1), (u2, v1), (u2, v2), (u1, v2)])
                            
        # If no invalid cycles, we have a perfect solution
        if not invalid_cycles:
            return edges
            
        # Greedy removal of edges to break all invalid cycles
        # Count frequency of each edge in invalid cycles
        edge_freq = {}
        for cycle in invalid_cycles:
            for e in cycle:
                edge_freq[e] = edge_freq.get(e, 0) + 1
                
        # Sort edges by frequency (remove most frequent first)
        # We need to remove edges such that every invalid cycle is hit
        # This is a Set Cover problem (hitting set). Greedy is good enough.
        
        edges_to_remove = set()
        remaining_cycles = list(invalid_cycles)
        
        while remaining_cycles:
            # Pick the edge that hits the most remaining cycles
            best_edge = None
            max_hits = 0
            # We can just iterate through all edges in the remaining cycles
            # For efficiency, we can pick randomly if too slow, but 2025 cycles is small
            for cycle in remaining_cycles:
                for e in cycle:
                    if e in edges_to_remove:
                        continue
                    hits = sum(1 for c in remaining_cycles if e in c)
                    if hits > max_hits:
                        max_hits = hits
                        best_edge = e
                        
            if best_edge is None:
                break
                
            edges_to_remove.add(best_edge)
            # Remove all cycles hit by best_edge
            remaining_cycles = [c for c in remaining_cycles if best_edge not in c]
            
        # Construct final edge list
        final_edges = []
        for e in edges:
            u, v, c = e
            if (u, v) not in edges_to_remove and (v, u) not in edges_to_remove:
                final_edges.append(e)
                
        if len(final_edges) > best_score:
            best_score = len(final_edges)
            best_edges = final_edges[:]
            
    # If we didn't find a good solution in K_{10,10}, fallback to C4-free greedy
    # But K_{10,10} with removal should be > 47 edges.
    
    if not best_edges:
        # Fallback: simple greedy C4-free
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        neighbors = [0] * n
        current_edges = []
        for u, v in all_edges:
            nu = neighbors[u]
            nv = neighbors[v]
            creates_c4 = False
            candidates_w = nu & ~(1 << v)
            cw = candidates_w
            while cw:
                lsb = cw & -cw
                w = lsb.bit_length() - 1
                cw -= lsb
                candidates_x = nv & ~(1 << u) & ~(1 << w)
                if candidates_x & neighbors[w]:
                    creates_c4 = True
                    break
            if not creates_c4:
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
                current_edges.append([u, v, 0])
        best_edges = current_edges
        
    return best_edges
