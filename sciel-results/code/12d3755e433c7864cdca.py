import time
import random

def construct():
    n = 15
    deadline = time.time() + 80
    
    # Start with K_{7,8} bipartite graph
    A = list(range(7))   # vertices 0-6
    B = list(range(7, 15))  # vertices 7-14
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
    
    def get_adjacency(edges):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def get_degrees(adj):
        return {i: len(adj[i]) for i in range(n)}
    
    def is_valid(adj, deg):
        # Check all 3-edge paths s-u-v-t where s != t
        # A 3-edge path is s-u-v-t with edges (s,u), (u,v), (v,t)
        # s and t are endpoints, need deg(s) != deg(t)
        for u in range(n):
            for v in adj[u]:
                if v <= u:
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
    
    adj = get_adjacency(edges)
    deg = get_degrees(adj)
    
    # Try to add intra-A edges
    # For an intra-A edge (a1, a2):
    #   3-paths: a1-a2-b-t where b in B, t in A (t != a1)
    #   deg(a1) vs deg(t): a1 has degree 8+1=9 (after adding), t has degree 8
    #   So deg(a1) != deg(t) always. Good.
    #   Also: s-a1-a2-b where s in A (s != a2), b in B
    #   deg(s) vs deg(b): s has degree 8, b has degree 7. 8 != 7. Good.
    #   Also: a1-a2-b-s' where s' in A... wait, that's the same as above.
    #   What about a2-a1-b-t? Same as first case with a1 and a2 swapped.
    #   What about s-a2-a1-b? deg(s)=8, deg(b)=7. Good.
    #   
    #   But wait: after adding one intra-A edge, degrees change.
    #   Let's say we add edge (a1, a2). Then deg(a1)=9, deg(a2)=9, others in A have deg 8.
    #   Now consider adding another intra-A edge (a3, a4).
    #   3-path: a3-a4-b-t where t in A, t != a3.
    #   If t = a1: deg(a3)=9, deg(a1)=9. VIOLATION!
    #   So we can only add one intra-A edge, or if we add multiple, we need to be careful.
    #
    # Actually, if we add k intra-A edges, the vertices involved in those edges will have higher degrees.
    # Let's think more carefully. If we add a matching of intra-A edges, each involved vertex gets +1 degree.
    # So vertices in the matching have degree 9, others have degree 8.
    # A 3-path a3-a4-b-t: if t is in the matching, deg(t)=9=deg(a3). Violation!
    # So we can't add any intra-A edges if there are other intra-A edges that make degrees equal.
    #
    # Unless... we add intra-B edges too, to differentiate degrees.
    
    # Let's try: add one intra-A edge and see if we can add intra-B edges.
    # After adding (a1, a2): deg(a1)=9, deg(a2)=9, other A: 8, all B: 7.
    # Now add an intra-B edge (b1, b2): deg(b1)=8, deg(b2)=8, other B: 7.
    # Check 3-paths:
    #   s-a1-a2-b-t: deg(s) vs deg(t). s in A, t in A. If s is not a1,a2: deg(s)=8. If t is not a1,a2: deg(t)=8. VIOLATION if s != a1,a2 and t != a1,a2.
    #   So this doesn't work either.
    
    # Let me try a different approach: greedy search for maximum edges.
    
    random.seed(42)
    
    best_edges = list(edges)
    best_count = len(edges)
    
    def check_validity(edge_list):
        adj = get_adjacency(edge_list)
        deg = get_degrees(adj)
        return is_valid(adj, deg), adj, deg
    
    # Try adding edges one by one
    all_possible = []
    for i in range(n):
        for j in range(i+1, n):
            if [i, j] not in best_edges and [j, i] not in best_edges:
                all_possible.append([i, j])
    
    random.shuffle(all_possible)
    
    current_edges = list(best_edges)
    
    for e in all_possible:
        if time.time() > deadline:
            break
        test_edges = current_edges + [e]
        valid, adj, deg = check_validity(test_edges)
        if valid:
            current_edges = test_edges
    
    if len(current_edges) > best_count:
        best_edges = current_edges
        best_count = len(current_edges)
    
    # Try removing edges to allow adding more
    # This is a local search: try swapping one edge for another
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        current = list(best_edges)
        adj = get_adjacency(current)
        deg = get_degrees(adj)
        
        # Try removing one edge and adding two
        for i in range(len(current)):
            if time.time() > deadline:
                break
            removed = current[i]
            rest = current[:i] + current[i+1:]
            
            # Find pairs of edges to add
            for j in range(len(all_possible)):
                if time.time() > deadline:
                    break
                e1 = all_possible[j]
                if e1 == removed:
                    continue
                if e1 in rest:
                    continue
                
                test1 = rest + [e1]
                valid1, adj1, deg1 = check_validity(test1)
                
                if not valid1:
                    continue
                
                for k in range(j+1, len(all_possible)):
                    e2 = all_possible[k]
                    if e2 == removed or e2 == e1:
                        continue
                    if e2 in rest:
                        continue
                    if e2 in test1:
                        continue
                    
                    test2 = test1 + [e2]
                    valid2, adj2, deg2 = check_validity(test2)
                    if valid2:
                        best_edges = test2
                        best_count = len(best2) if False else len(test2)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    # Final validation
    valid, adj, deg = check_validity(best_edges)
    if not valid:
        # Fallback to K_{7,8}
        best_edges = []
        for u in A:
            for v in B:
                best_edges.append([u, v])
    
    # Ensure types are correct
    result = [[int(u), int(v)] for u, v in best_edges]
    return result
