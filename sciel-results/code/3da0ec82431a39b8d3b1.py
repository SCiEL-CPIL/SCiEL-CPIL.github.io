import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = []
    best_count = 0
    
    # Helper to check if graph contains Q3
    # adj[u] is a set of neighbors
    def has_q3(adj):
        # adj: list of sets
        for u in range(n):
            for v in adj[u]:
                if u >= v: continue
                # u-v is an edge
                # Find w1, w2 in N(u) \ {v}
                # Find z1, z2 in N(v) \ {u}
                nu = adj[u] - {v}
                nv = adj[v] - {u}
                
                if len(nu) < 2 or len(nv) < 2:
                    continue
                
                # Get lists for iteration
                nu_list = list(nu)
                nv_list = list(nv)
                
                # Pairs from nu
                for i in range(len(nu_list)):
                    w1 = nu_list[i]
                    for j in range(i+1, len(nu_list)):
                        w2 = nu_list[j]
                        
                        # Find x neighbor of w1, w2 not u
                        # x must be in N(w1) and N(w2)
                        # In Q3, x is unique for this pair?
                        # N(w1) and N(w2) intersection size in Q3 is 1 (plus u).
                        # So x is the only candidate.
                        inter_w = adj[w1] & adj[w2]
                        inter_w.discard(u)
                        if not inter_w:
                            continue
                        # If more than 1, we might have multiple candidates, but in Q3 it's 1.
                        # We can just check each x in inter_w
                        for x in inter_w:
                            # Pairs from nv
                            for k in range(len(nv_list)):
                                z1 = nv_list[k]
                                for l in range(k+1, len(nv_list)):
                                    z2 = nv_list[l]
                                    
                                    # Find y neighbor of z1, z2 not v
                                    inter_z = adj[z1] & adj[z2]
                                    inter_z.discard(v)
                                    if not inter_z:
                                        continue
                                    
                                    for y in inter_z:
                                        # Check distinctness
                                        if x == y: continue
                                        if x in (u, v, w1, w2, z1, z2): continue
                                        if y in (u, v, w1, w2, z1, z2): continue
                                        
                                        # Check edges
                                        # 1. x-y
                                        if y not in adj[x]: continue
                                        
                                        # 2. w1-z1 and w2-z2 OR w1-z2 and w2-z1
                                        if (z1 in adj[w1] and z2 in adj[w2]) or (z2 in adj[w1] and z1 in adj[w2]):
                                            return True
        return False

    def run_greedy(seed):
        random.seed(seed)
        edges = all_edges[:]
        random.shuffle(edges)
        
        adj = [set() for _ in range(n)]
        edge_set = set()
        
        count = 0
        for u, v in edges:
            # Check if adding (u,v) creates Q3
            # Temporarily add
            adj[u].add(v)
            adj[v].add(u)
            
            if has_q3(adj):
                # Remove
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                edge_set.add((u, v))
                count += 1
        
        # Local search: try to add more edges
        # Iterate through all non-edges, try to add if possible
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in all_edges:
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                # Try add
                adj[u].add(v)
                adj[v].add(u)
                if not has_q3(adj):
                    edge_set.add((u, v))
                    count += 1
                    improved = True
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
        
        return count, list(edge_set)

    # Run multiple restarts
    restart = 0
    while time.time() < deadline:
        count, edges_list = run_greedy(restart)
        if count > best_count:
            best_count = count
            best_edges = edges_list
        
        # Try a few more seeds
        restart += 1
        
        # If we found 55+, we can probably stop or try harder
        if best_count >= 55:
            # Try to polish
            pass

    # Return edges in required format
    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    print(construct())
