import time
import random
import sys

N = 40

def has_c6_fast(adj, u, v):
    """
    Check if adding edge (u,v) creates a C6.
    adj is list of sets.
    We look for a path u-a-b-c-d-v where a in N(u), d in N(v), and a-b-c-d is a path of length 3.
    """
    nu = adj[u]
    nv = adj[v]
    
    # We need to find a, b, c, d such that:
    # a in nu, (a,b) in E, (b,c) in E, (c,d) in E, d in nv
    # All vertices u,a,b,c,d,v distinct.
    
    # Optimization: 
    # 1. Compute set of all 'b' reachable from nu in 1 step (excluding u).
    #    Let S_b = { b | exists a in nu, b in adj[a], b != u }
    # 2. Compute set of all 'c' reachable from nv in 1 step (excluding v).
    #    Let S_c = { c | exists d in nv, c in adj[d], c != v }
    # 3. Check if there is an edge (b,c) such that b in S_b and c in S_c.
    #    If so, we have a path a-b-c-d.
    #    We must verify distinctness:
    #    - a is chosen from nu such that (a,b) in E.
    #    - d is chosen from nv such that (c,d) in E.
    #    - We need to ensure there exists a valid a and d for the found (b,c).
    #    - Also b != c (guaranteed if we check edge (b,c) and simple graph, but b,c could be same? No, edge (b,c) implies b!=c).
    #    - We need a != c, a != d, a != v (a in nu so a!=v unless self-loop, but simple graph), a != u (a in nu so a!=u).
    #    - We need d != b, d != a, d != u.
    
    # Let's precompute S_b and S_c.
    # Actually, iterating over edges in S_b x S_c might be expensive if sets are large.
    # Better: iterate over b in S_b, then for each neighbor c of b, check if c in S_c.
    
    # Build S_b:
    S_b = set()
    for a in nu:
        for b in adj[a]:
            if b != u:
                S_b.add(b)
                
    # Build S_c:
    S_c = set()
    for d in nv:
        for c in adj[d]:
            if c != v:
                S_c.add(c)
                
    # Now check for edge (b,c) with b in S_b, c in S_c.
    for b in S_b:
        for c in adj[b]:
            if c in S_c:
                # We have a candidate path structure a-b-c-d.
                # We need to verify that there exist a in nu and d in nv
                # such that:
                # 1. (a,b) in E  (a in nu, a in adj[b])
                # 2. (c,d) in E  (d in nv, d in adj[c])
                # 3. Distinctness:
                #    u, a, b, c, d, v all distinct.
                #    We know:
                #    - u != a (a in nu)
                #    - u != b (b in S_b, b != u)
                #    - u != c (c in S_c, c != v, but c could be u? If c=u, then d in adj[u] and d in nv. 
                #             Path u-a-b-u-d-v. This is not a simple cycle. 
                #             So we must check c != u.)
                #    - u != d (d in nv, so d != u)
                #    - a != b (simple graph)
                #    - a != c (must check)
                #    - a != d (must check)
                #    - a != v (a in nu, so a != v)
                #    - b != c (edge (b,c), so b != c)
                #    - b != d (must check)
                #    - b != v (b in S_b, b != u, but b could be v? If b=v, then a in adj[v] and a in nu. 
                #             Path u-a-v-c-d-v. Not simple. 
                #             So we must check b != v.)
                #    - c != d (simple graph)
                #    - c != v (c in S_c, c != v)
                #    - d != v (d in nv, so d != v)
                
                # So we need to check:
                # - c != u
                # - b != v
                # - Find a in nu such that (a,b) in E and a != c and a != d
                # - Find d in nv such that (c,d) in E and d != b and d != a
                
                if c == u or b == v:
                    continue
                    
                # Find valid a and d
                # Candidates for a: a in nu, a in adj[b], a != c
                # Candidates for d: d in nv, d in adj[c], d != b
                
                valid_a = None
                for a in nu:
                    if a in adj[b] and a != c:
                        # Potential a
                        # Now find d
                        for d in nv:
                            if d in adj[c] and d != b and d != a:
                                # Found valid a, d
                                # Check a != d (already checked d != a)
                                # Check a != c (checked a != c)
                                # Check b != d (checked d != b)
                                # Check c != u (checked)
                                # Check b != v (checked)
                                return True
                        # If no d found for this a, try next a
                        # Wait, we need ANY valid pair (a,d).
                        # So if we find one, return True.
                        # If we don't find d for this a, we continue to next a.
                        # But we must be careful: maybe a different a allows a d.
                        # So the loop structure should be:
                        # for a in candidates_a:
                        #   for d in candidates_d:
                        #     if valid: return True
                        # return False (for this b,c)
                
                # Let's rewrite the inner check more efficiently
                # Precompute candidate_a and candidate_d sets?
                # Since this is inside a loop, let's just iterate.
                
                found = False
                for a in nu:
                    if a == c:
                        continue
                    if a not in adj[b]:
                        continue
                    # a is valid candidate
                    for d in nv:
                        if d == b or d == a:
                            continue
                        if d not in adj[c]:
                            continue
                        # d is valid candidate
                        found = True
                        break
                    if found:
                        break
                if found:
                    return True
                    
    return False


def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    best_edges = []
    best_count = 0
    
    random.seed(42)
    
    # Precompute all edges
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Run multiple randomized greedy constructions
    # The C6 check is the bottleneck. 
    # Let's see how fast it is.
    
    # Optimization: Use a bitset representation for adjacency?
    # Python sets are okay for N=40.
    
    for restart in range(500):
        if time.time() > deadline - 5:
            break
            
        adj = [set() for _ in range(N)]
        edges = []
        
        # Shuffle edges
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if u in adj[v]:
                continue
            if not has_c6_fast(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
        # Early stopping if we hit a very high count?
        # Best known is 105. If we get 100+, we might stop early?
        # No, let's keep searching for a bit longer.
        
    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
