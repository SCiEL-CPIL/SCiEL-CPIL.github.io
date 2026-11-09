import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 15
    
    # Build a threshold graph: vertex i (0-indexed) connects to all vertices j where j > i
    # This gives a "nested" structure where neighborhoods are strictly decreasing
    # Degree of vertex i: n - 1 - i (vertex 0 has degree 14, vertex 14 has degree 0)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append([i, j])
    
    # This is a complete graph K_15 minus... wait, that's all edges.
    # Actually for i in range(n), j in range(i+1, n) gives ALL pairs = K_15 = 105 edges.
    # But K_15 has all degrees equal to 14, so ANY path s-u-v-t has deg(s)=deg(t)=14. VIOLATION!
    
    # I need a proper threshold graph with DISTINCT or structured degrees.
    # Let me use: vertex i connects to vertices j where j > i AND j < i + (n-1-i)
    # Actually, let me think more carefully.
    
    # A valid threshold graph: split vertices into two sets A and B.
    # Within A: complete graph. Within B: no edges. Between A and B: nested.
    # 
    # Simpler approach: Use the "co-chain" threshold graph.
    # Vertices 0,1,...,n-1. Vertex i is adjacent to vertex j (i<j) iff i + j >= n-1.
    # This gives a Ferrers graph / threshold graph.
    
    # Let me reconsider. The key property needed:
    # For any path s-u-v-t (4 distinct vertices, edges su, uv, vt), we need deg(s) != deg(t).
    # 
    # In a threshold graph with degree sequence d_0 > d_1 > ... > d_{n-1} (strictly decreasing),
    # and nested neighborhoods N(0) ⊃ N(1) ⊃ ... ⊃ N(n-1):
    # If s-u is an edge, then s ∈ N(u) or u ∈ N(s). Since neighborhoods are nested,
    # if deg(s) > deg(u), then u ∈ N(s) means s is "higher". 
    # 
    # Let me just use a specific construction and verify.
    
    # Construction: vertex i connects to vertex j (i < j) iff j - i <= n - 1 - i
    # i.e., j <= n - 1. That's all j > i. So it's K_n again. Bad.
    
    # Better: Use a split graph. Partition into A = {0,...,a-1} and B = {a,...,n-1}.
    # A is a clique, B is independent. Each vertex in B connects to a prefix of A.
    # 
    # Let a = 7, so A = {0,...,6}, B = {7,...,14}.
    # A is a clique: C(7,2) = 21 edges.
    # B is independent: 0 edges within B.
    # Between A and B: vertex b in B connects to first (b - a + 1) vertices in A? 
    # Let's say vertex b (b from a to n-1) connects to A[0], A[1], ..., A[b-a].
    # So vertex 7 connects to {0}, vertex 8 to {0,1}, ..., vertex 14 to {0,...,7}... 
    # Wait, A only has 7 vertices (indices 0-6). Let me adjust.
    
    # Let a = 7. B vertices: 7,8,9,10,11,12,13,14 (8 vertices).
    # Vertex b in B connects to A[0..(b-a)] but capped at A[0..6].
    # Vertex 7: connects to A[0] (1 edge)
    # Vertex 8: connects to A[0,1] (2 edges)
    # Vertex 9: connects to A[0,1,2] (3 edges)
    # Vertex 10: connects to A[0,1,2,3] (4 edges)
    # Vertex 11: connects to A[0,1,2,3,4] (5 edges)
    # Vertex 12: connects to A[0,1,2,3,4,5] (6 edges)
    # Vertex 13: connects to A[0,1,2,3,4,5,6] (7 edges)
    # Vertex 14: connects to A[0,1,2,3,4,5,6] (7 edges)
    
    # Degrees:
    # A vertices: each in A has degree (a-1) + (number of B vertices that connect to it)
    # A[0]: connected to all 6 other A + all 8 B = 6+8 = 14
    # A[1]: 6 + 7 (B vertices 8-14) = 13
    # A[2]: 6 + 6 (B vertices 9-14) = 12
    # A[3]: 6 + 5 (B vertices 10-14) = 11
    # A[4]: 6 + 4 (B vertices 11-14) = 10
    # A[5]: 6 + 3 (B vertices 12-14) = 9
    # A[6]: 6 + 2 (B vertices 13-14) = 8
    # B[7]: 1
    # B[8]: 2
    # B[9]: 3
    # B[10]: 4
    # B[11]: 5
    # B[12]: 6
    # B[13]: 7
    # B[14]: 7
    
    # Degree sequence: 14,13,12,11,10,9,8,7,7,6,5,4,3,2,1
    # Note: B[13] and B[14] both have degree 7. That's OK as long as no path has them as endpoints.
    
    # Now check the constraint. In this graph, can we have a path s-u-v-t with deg(s)=deg(t)?
    # The degrees are mostly distinct except 7 appears twice (vertices 13 and 14).
    # So the only potential violation is a path with s=13, t=14 or s=14, t=13.
    # 
    # For s=13, t=14: We need path 13-u-v-14 where 13-u, u-v, v-14 are edges.
    # 13 is in B, connects to A[0..6]. 14 is in B, connects to A[0..6].
    # So u must be in A (since 13 only connects to A), and v must be in A (since 14 only connects to A).
    # And u-v must be an edge. Since A is a clique, any u,v in A with u≠v works.
    # So path: 13 - u - v - 14 where u,v ∈ A, u≠v.
    # This EXISTS! e.g., 13-0-1-14. deg(13)=7, deg(14)=7. VIOLATION!
    
    # So I need to fix this. Make B[13] and B[14] have different degrees.
    # Let vertex 14 connect to A[0..5] (6 edges) instead of A[0..6].
    # Then deg(14) = 6, deg(13) = 7. Now all degrees are distinct? 
    # Wait, B[12] also has degree 6. So deg(12)=6 and deg(14)=6. Same issue.
    
    # Let me make all B vertices have distinct degrees by adjusting.
    # Actually, let me use a different split. Let a=8, so A={0..7}, B={8..14} (7 vertices in B).
    # A is a clique: C(8,2)=28 edges.
    # B is independent.
    # Vertex b in B connects to A[0..(b-a-1)]? Let me define:
    # Vertex 8: connects to A[0] (1 edge)
    # Vertex 9: connects to A[0,1] (2 edges)
    # Vertex 10: connects to A[0,1,2] (3 edges)
    # Vertex 11: connects to A[0,1,2,3] (4 edges)
    # Vertex 12: connects to A[0,1,2,3,4] (5 edges)
    # Vertex 13: connects to A[0,1,2,3,4,5] (6 edges)
    # Vertex 14: connects to A[0,1,2,3,4,5,6] (7 edges)
    
    # Degrees:
    # A[0]: 7 (other A) + 7 (all B) = 14
    # A[1]: 7 + 6 (B[9..14]) = 13
    # A[2]: 7 + 5 (B[10..14]) = 12
    # A[3]: 7 + 4 (B[11..14]) = 11
    # A[4]: 7 + 3 (B[12..14]) = 10
    # A[5]: 7 + 2 (B[13..14]) = 9
    # A[6]: 7 + 1 (B[14]) = 8
    # A[7]: 7 + 0 = 7
    # B[8]: 1
    # B[9]: 2
    # B[10]: 3
    # B[11]: 4
    # B[12]: 5
    # B[13]: 6
    # B[14]: 7
    
    # Degree sequence: 14,13,12,11,10,9,8,7,7,6,5,4,3,2,1
    # Again A[7] and B[14] both have degree 7. Same problem.
    
    # The issue is that the max degree in B equals min degree in A when they're adjacent in the sequence.
    # Let me just verify if this actually causes a violation.
    # deg(7)=7 (A[7]) and deg(14)=7 (B[14]).
    # Path 7-u-v-14: 7 is in A, connects to all A and B[14]. 14 is in B, connects to A[0..6].
    # So u must be a neighbor of 7: u ∈ A\{7} or u=14. v must be neighbor of 14: v ∈ A[0..6].
    # And u-v must be edge.
    # If u=14, then v must be neighbor of 14 (v∈A[0..6]) and 14-v edge (yes). Path: 7-14-v-14? No, t=14, so v-14 is the last edge, but we already used 14 as u. s=7,u=14,v=?,t=14. But u≠t required? The problem says s-u-v-t with s≠t. It doesn't say all distinct, but "simple three-edge path" implies 4 distinct vertices.
    # 
    # Let me re-read: "every simple three-edge path has endpoints of equal degree". Simple path means all 4 vertices distinct.
    # 
    # So for s=7, t=14: need distinct u,v with 7-u, u-v, v-14 all edges.
    # 7's neighbors: A\{7} = {0,1,2,3,4,5,6} and {14}. So u ∈ {0,1,2,3,4,5,6,14}.
    # 14's neighbors: A[0..6] = {0,1,2,3,4,5,6}. So v ∈ {0,1,2,3,4,5,6}.
    # u-v must be edge. If u ∈ A and v ∈ A, they're connected (A is clique). 
    # So pick u=0, v=1: path 7-0-1-14. All distinct. deg(7)=7, deg(14)=7. VIOLATION!
    
    # I need all degrees to be distinct, OR ensure that equal-degree vertices can't be endpoints of any 3-path.
    # 
    # Easiest: make all 15 degrees distinct. Max possible with 15 vertices: degrees 0-14.
    # A graph with degree sequence 0,1,2,...,14 is the threshold graph where vertex i has degree i.
    # This is: vertex i connects to vertices i+1, i+2, ..., 14. (i.e., j > i implies edge if... no)
    # Actually, degree sequence 0,1,...,14: vertex 0 has degree 0 (isolated), vertex 1 has degree 1, etc.
    # Construction: vertex i (0-indexed) connects to vertices i+1, ..., i+(n-1-i)? No.
    # Standard construction: vertex i connects to all vertices j where j > i. Then deg(i) = n-1-i.
    # That gives degrees 14,13,...,0. All distinct!
    # 
    # Wait, that's the same as before: edges (i,j) for i<j. That's K_15! All degrees = 14. NO!
    # 
    # I'm confusing myself. Let me be careful.
    # If edge (i,j) exists for i<j, then vertex i is connected to i+1,...,n-1, so deg(i) = n-1-i.
    # Vertex 0: deg=14, vertex 1: deg=13, ..., vertex 14: deg=0.
    # But this IS the complete graph K_15! Every pair is connected. So deg(i) = 14 for all i.
    # 
    # No wait. In K_15, every vertex connects to every other, so deg=14 for all. 
    # The construction "edge (i,j) for i<j" gives ALL pairs, which is K_15. I was wrong to think degrees differ.
    # 
    # For a threshold graph with degree sequence d_0 > d_1 > ... > d_{n-1}:
    # The standard construction: vertex i is connected to vertices 0,1,...,d_i-1? No.
    # 
    # Let me use the proper threshold graph construction.
    # A threshold graph is built by adding vertices one at a time, each either isolated or dominating.
    # 
    # For n=15, let me use: add vertices 0,1,...,14. Vertex 0 is isolated. 
    # Vertex 1 dominates (connects to 0). Vertex 2 dominates (connects to 0,1). Etc.
    # But that gives vertex i connected to all 0,...,i-1, so deg(i)=i, and the last vertex has degree 14.
    # Degrees: 0,1,2,...,14. All distinct!
    # 
    # Edges: (i,j) for i<j. Wait, that's the same as before!
    # Vertex 0: no edges, deg=0. Vertex 1: edge to 0, deg=1. Vertex 2: edges to 0,1, deg=2.
    # Vertex 14: edges to 0,1,...,13, deg=14.
    # Total edges: sum_{i=1}^{14} i = 105. That's K_15 again!
    # 
    # I see my error. "Vertex i connects to all 0,...,i-1" means for vertex 14, it connects to 0-13, which is 14 edges. And vertex 13 connects to 0-12, 13 edges. But vertex 0 and 1: edge. Vertex 0 and 2: edge. Etc. So yes, every pair (i,j) with i<j is an edge. It's K_15.
    # 
    # The issue: if vertex i connects to ALL previous vertices, you get a complete graph. 
    # 
    # For a proper threshold graph with distinct degrees, I need a different construction.
    # 
    # Let me use the split graph approach but ensure distinct degrees.
    # 
    # Alternative: Use a "chain" threshold graph where not all cross edges are present.
    # 
    # Let me try: A = {0,...,6} (clique), B = {7,...,14} (independent).
    # Cross edges: vertex b in B connects to A[0], A[1], ..., A[b-7-1]? 
    # Let me define: vertex (7+k) for k=0..7 connects to A[0..k-1]. So:
    # 7: no A neighbors (k=0)
    # 8: A[0] (k=1)
    # 9: A[0,1] (k=2)
    # 10: A[0,1,2] (k=3)
    # 11: A[0,1,2,3] (k=4)
    # 12: A[0,1,2,3,4] (k=5)
    # 13: A[0,1,2,3,4,5] (k=6)
    # 14: A[0,1,2,3,4,5,6] (k=7)
    
    # Degrees:
    # A[i]: (6) + (number of B vertices that connect to A[i])
    # A[0]: 6 + 7 (B[8..14]) = 13
    # A[1]: 6 + 6 (B[9..14]) = 12
    # A[2]: 6 + 5 (B[10..14]) = 11
    # A[3]: 6 + 4 (B[11..14]) = 10
    # A[4]: 6 + 3 (B[12..14]) = 9
    # A[5]: 6 + 2 (B[13..14]) = 8
    # A[6]: 6 + 1 (B[14]) = 7
    # B[7]: 0
    # B[8]: 1
    # B[9]: 2
    # B[10]: 3
    # B[11]: 4
    # B[12]: 5
    # B[13]: 6
    # B[14]: 7
    
    # Degree sequence: 13,12,11,10,9,8,7,7,6,5,4,3,2,1,0
    # A[6] and B[14] both have degree 7. Potential issue.
    
    # Check: can we have path s-u-v-t with s=6, t=14 (both deg 7)?
    # 6 is in A, neighbors: A\{6} = {0,1,2,3,4,5} and B[14]. So N(6) = {0,1,2,3,4,5,14}.
    # 14 is in B, neighbors: A[0..6] = {0,1,2,3,4,5,6}. So N(14) = {0,1,2,3,4,5,6}.
    # Need u ∈ N(6), v ∈ N(14), u≠v, u≠6, u≠14, v≠6, v≠14, and u-v edge.
    # u can be in {0,1,2,3,4,5,14}, v in {0,1,2,3,4,5,6}.
    # If u=0, v=1: path 6-0-1-14. 6-0 edge? Yes (A clique). 0-1 edge? Yes. 1-14 edge? Yes. All distinct. VIOLATION!
    
    # I keep hitting this. Let me just make B[14] not connect to A[6].
    # Modify: B[14] connects to A[0..5] only. Then deg(14) = 6.
    # But B[13] also has deg 6. Now A[5] has deg 8, B[13] and B[14] both deg 6.
    # Check s=13, t=14 (both deg 6):
    # 13 in B, N(13) = A[0..5]. 14 in B, N(14) = A[0..5].
    # u ∈ A[0..5], v ∈ A[0..5], u≠v. u-v edge (A clique). Path 13-u-v-14. VIOLATION!
    
    # The fundamental issue: if two vertices in B have the same "prefix" of A neighbors, they can be endpoints of a path through A.
    
    # Solution: Make all B vertices have DIFFERENT neighbor sets in A, AND ensure no two vertices (anywhere) have the same degree, OR if they do, they can't be connected by a 2-edge path.
    
    # Easiest: make all degrees distinct. With 15 vertices, I need 15 distinct degree values.
    # 
    # Let me use a threshold graph with degree sequence 0,1,2,...,14 properly.
    # The correct construction: vertex i has degree i, and N(i) = {j : j < i and ...}
    # 
    # Actually, the standard threshold graph with degree sequence 0,1,...,n-1:
    # It's the graph where edge (i,j) exists iff i+j >= n-1? Let me check.
    # For n=15: edge (i,j) iff i+j >= 14.
    # deg(0): j such that 0+j>=14, j>0: j=14. deg=1.
    # deg(1): j such that 1+j>=14, j≠1: j=13,14. deg=2.
    # deg(2): j=12,13,14. deg=3.
    # ...
    # deg(7): j such that 7+j>=14, j≠7: j=7? No, j>7 or j<7. j=7: 14>=14 yes but j=i. j=8..14: 7 edges. j=6: 13<14 no. So deg(7)=7.
    # deg(8): j such that 8+j>=14: j=6,7,8(skip),9,...,14. j=6,7,9,10,11,12,13,14. deg=8.
    # deg(14): j such that 14+j>=14: all j≠14. deg=14.
    # 
    # Degrees: 1,2,3,4,5,6,7,7,8,9,10,11,12,13,14. 
    # deg(7) and deg(8) both... wait let me recount.
    # deg(0)=1, deg(1)=2, deg(2)=3, deg(3)=4, deg(4)=5, deg(5)=6, deg(6)=7, deg(7)=7, deg(8)=8, deg(9)=9, deg(10)=10, deg(11)=11, deg(12)=12, deg(13)=13, deg(14)=14.
    # deg(6)=7 and deg(7)=7. Again a collision!
    
    # This is frustrating. Let me just write a verifier and use a greedy approach.
    
    # Start with a known valid base and greedily add edges.
    
    def make_adj(edge_list, n):
        adj = [set() for _ in range(n)]
        for e in edge_list:
            u, v = e
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_valid(adj, n):
        deg = [len(adj[i]) for i in range(n)]
        for u in range(n):
            for v in adj[u]:
                if u > v:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    # Start with a star (valid, 14 edges) and greedily add edges.
    # Better: start with a valid dense base.
    
    # Let me start with the split graph A={0..6} clique, B={7..14} independent, 
    # with B[b] connecting to A[0..b-7-1], and verify/fix.
    
    # Actually, let me just use a greedy approach from a star, which we know is valid.
    
    adj = [set() for _ in range(n)]
    # Star: vertex 0 connected to all others
    for v in range(1, n):
        adj[0].add(v)
        adj[v].add(0)
    
    deg = [len(adj[i]) for i in range(n)]
    # deg = [14, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # Many vertices have degree 1. Can we have path s-u-v-t with deg(s)=deg(t)=1?
    # s and t have degree 1, so their only neighbor is 0. 
    # Path s-0-v-t: s=1, 0, v, t. But s-0 edge, 0-v edge, v-t edge. 
    # v must be neighbor of 0 (all of them) and t must be neighbor of v.
    # If t has degree 1, t's only neighbor is 0. So v-t edge means v=0. But v≠s=1, v≠t, and u=0, so v≠0. Contradiction.
    # So no such path. Star is valid.
    
    # Now greedily add edges. Try all missing edges, add if valid.
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Sort by some heuristic: prioritize edges that connect high-degree vertices
    # to make the graph denser.
    
    current_edges = [[0, v] for v in range(1, n)]
    
    # Greedy: repeatedly try to add the best edge
    improved = True
    while improved:
        improved = False
        best_edge = None
        best_score = -1
        for i in range(n):
            for j in range(i+1, n):
                if j in adj[i]:
                    continue
                # Try adding edge (i,j)
                adj[i].add(j)
                adj[j].add(i)
                deg_new = [len(adj[k]) for k in range(n)]
                # Check validity
                valid = True
                for u in range(n):
                    for v in adj[u]:
                        if u > v:
                            continue
                        for s in adj[u]:
                            if s == v:
                                continue
                            for t in adj[v]:
                                if t == u or t == s:
                                    continue
                                if deg_new[s] == deg_new[t]:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                    if not valid:
                        break
                if valid:
                    # Score: prefer edges that increase density
                    score = len(adj[i]) + len(adj[j])
                    if score > best_score:
                        best_score = score
                        best_edge = (i, j)
                else:
                    adj[i].discard(j)
                    adj[j].discard(i)
        
        if best_edge:
            i, j = best_edge
            adj[i].add(j)
            adj[j].add(i)
            current_edges.append([i, j])
            improved = True
            if time.time() > deadline - 10:
                break
    
    # Convert to proper format
    result = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                result.append([int(i), int(j)])
    
    return result
