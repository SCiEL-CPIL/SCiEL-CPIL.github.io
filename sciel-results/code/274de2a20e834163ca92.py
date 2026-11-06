import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 16
    # Generate all edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # Precompute all Q3 subgraphs in K16
    # Q3 vertices: 0..7
    # Edges of Q3:
    # 0-1, 0-2, 0-4
    # 1-3, 1-5, 1-4? No.
    # Standard Q3 edges:
    # (0,1), (0,2), (0,4)
    # (1,3), (1,5), (1,4) -- Wait, 1 is connected to 0, 3, 5. 4 is connected to 0, 2, 6.
    # Let's list properly.
    # Vertices 0..7 as binary 000..111.
    # Edge if Hamming distance 1.
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.append((i, j))
                
    q3_edges_set = set(q3_edges)
    
    # Find all mappings of Q3 to K16
    # This is expensive if done naively.
    # But we can just generate all subsets of 8 vertices and check if they induce a Q3?
    # No, Q3 is a subgraph, not necessarily induced.
    # So we need to check if the 8 vertices contain a subgraph isomorphic to Q3.
    # This is equivalent to checking if there is a bijection from Q3 vertices to the 8 vertices
    # such that all Q3 edges map to edges in the complete graph (which they always do).
    # Wait, if the host is K16, ANY 8 vertices contain MANY Q3 subgraphs.
    # Specifically, the number of Q3 subgraphs in K8 is fixed.
    # Let's count how many distinct edge sets correspond to Q3 in a fixed set of 8 vertices.
    # Actually, we just need to know IF a specific set of 12 edges (forming a Q3) is present.
    
    # Better approach:
    # Enumerate all possible Q3 subgraphs in K16.
    # A Q3 subgraph is defined by 8 distinct vertices and a specific set of 12 edges among them.
    # Since K16 is complete, any assignment of Q3 vertices to 8 distinct vertices in K16
    # yields a valid Q3 subgraph.
    # How many such assignments?
    # P(16, 8) * (number of automorphisms of Q3)?
    # Actually, we just need the set of edges.
    # Two different assignments might yield the same set of edges?
    # No, the edge set determines the vertices (endpoints).
    # So we can generate all edge sets.
    
    # Number of Q3 subgraphs in K16:
    # Choose 8 vertices: 12870.
    # For each set, how many Q3 subgraphs?
    # The number of copies of Q3 in K8.
    # Let's calculate this once.
    
    # To save time, we can just use the property:
    # G is Q3-free iff no Q3 subgraph is fully contained.
    
    # Let's generate the list of Q3 edge masks.
    # This might be large.
    # Number of Q3 in K8:
    # Q3 is 3-regular.
    # Number of labeled Q3 in K8?
    # It is known that K8 contains many Q3.
    
    # Given time constraints, let's use a heuristic search instead of precomputation if precomputation is slow.
    # But 12870 * (Q3s in K8) is manageable.
    
    # Let's find Q3s in K8.
    # We can do a DFS or just hardcode the count?
    # Actually, we can just generate all permutations of 8 vertices? No.
    
    # Alternative:
    # Start with K16.
    # While Q3 exists:
    #   Find one.
    #   Remove edge.
    
    # Finding a Q3:
    # Pick 8 random vertices.
    # Check if they contain a Q3.
    # How to check?
    # We can try all 8! mappings? Too slow.
    # But we can use the fact that Q3 is bipartite.
    
    # Let's stick to the seed + improvements.
    # Seed: 39 edges.
    # Improved Seed: K_3,13 + K_3(A) = 42 edges.
    
    # Let's try to add edges to B.
    # G = K_3,13 + K_3(A).
    # Try adding random edges in B.
    
    def is_q3_free(edges_set):
        # Check if any Q3 subgraph exists.
        # This is the bottleneck.
        # For now, assume we can check quickly.
        pass

    # Let's return the 42-edge graph first to ensure improvement.
    # A = [0, 1, 2]
    # B = [3..15]
    edges = []
    # All A-B edges
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
    # All A-A edges
    for i in range(3):
        for j in range(i+1, 3):
            edges.append([i, j])
            
    return edges
