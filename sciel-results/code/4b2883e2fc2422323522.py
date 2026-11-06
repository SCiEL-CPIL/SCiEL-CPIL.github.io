import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # Safety margin
    N = 40
    n_pairs = N * (N - 1) // 2
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    pair_index = {p: k for k, p in enumerate(all_pairs)}

    def creates_c6(adj, u, v):
        """
        Check if adding edge (u, v) to graph 'adj' creates a C6.
        This happens if there is a path of length 4 from u to v in G \ {(u,v)}.
        Path: u - a - b - c - v.
        Conditions:
        - a in N(u)
        - c in N(v)
        - b in N(a) and b in N(c)
        - a, b, c are distinct from u, v and each other.
        """
        if u == v:
            return False
        if v in adj[u]:
            return False # Already an edge, no new C6 from adding it
        
        nu = adj[u]
        nv = adj[v]
        
        # We need to find a, c, b such that:
        # a in N(u), c in N(v)
        # b in N(a) ∩ N(c)
        # a != c, a != v, c != u, b != u, b != v
        # Also a != b, c != b (implied by simple path structure usually, but let's be explicit)
        
        for a in nu:
            if a == v: continue
            for c in nv:
                if c == u or c == a: continue
                # Check if there is a b in N(a) ∩ N(c) such that b not in {u, v, a, c}
                # Actually b just needs to be distinct from u, v, a, c to form a simple cycle u-a-b-c-v-u
                # If b == a or b == c, it's not a simple path of length 4.
                # If b == u or b == v, it's not a simple path.
                
                # Iterate over the smaller of N(a) and N(c)
                na = adj[a]
                nc = adj[c]
                if len(na) > len(nc):
                    na, nc = nc, na
                    # swap roles if we want, but logic is symmetric for intersection
                    # just iterate over smaller
                    # Note: we are checking if ANY b exists.
                
                for b in na:
                    if b == u or b == v or b == a or b == c:
                        continue
                    if b in nc:
                        # Found a valid path u-a-b-c-v
                        # Verify it's a simple cycle: u, a, b, c, v are distinct.
                        # u != a (a in N(u))
                        # u != c (c in N(v), if c==u then u-v edge exists? No, we checked v not in N(u). But c could be u? If c==u, then u in N(v), so (u,v) is edge. But we are checking if ADDING (u,v) creates C6. So (u,v) is NOT in adj yet. So u not in N(v). So c != u is guaranteed if we assume simple graph and (u,v) not present. Wait, c is in N(v). If c==u, then u in N(v), which means (u,v) is an edge. But we check `if v in adj[u]: return False` at start. So if we reach here, (u,v) is NOT an edge. Thus u not in N(v). So c cannot be u. Similarly a cannot be v.
                        # So we just need b distinct from u, v, a, c.
                        return True
        return False

    def count_edges(adj):
        return sum(len(s) for s in adj) // 2

    def greedy_build():
        adj = [set() for _ in range(N)]
        random.shuffle(all_pairs)
        for (i, j) in all_pairs:
            if not creates_c6(adj, i, j):
                adj[i].add(j)
                adj[j].add(i)
        return adj

    def local_search(adj, time_limit):
        best_adj = [set(s) for s in adj]
        best_edges = count_edges(best_adj)
        no_improve = 0
        max_no_improve = 50 # Restart local search if stuck for a while
        
        while time.time() < time_limit and no_improve < max_no_improve:
            # 1. Try to add a random non-edge
            non_edges = [(i, j) for (i, j) in all_pairs if j not in adj[i]]
            if not non_edges:
                break
            
            random.shuffle(non_edges)
            improved = False
            for (i, j) in non_edges:
                if time.time() >= time_limit:
                    break
                if not creates_c6(adj, i, j):
                    adj[i].add(j)
                    adj[j].add(i)
                    current_edges = count_edges(adj)
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [set(s) for s in adj]
                        no_improve = 0
                        improved = True
                        break
                    else:
                        # Edge count same? No, adding edge always increases count by 1.
                        # So if we add it, we improve.
                        pass
            
            if not improved:
                no_improve += 1
                # 2. Try 1-edge swap: Remove (a,b), add (c,d) and (e,f)
                # This allows us to change structure to allow more edges later.
                # Strategy: Pick a random edge (a,b) to remove.
                # Then try to add two random non-edges.
                # If we can add both, we net +1 edge.
                # If we can add one, net 0.
                # We only accept if net > 0 or if we are stuck and want to explore.
                # But for "maximize edges", we should only accept if net > 0.
                # If net == 0, it's a move to a different local optimum. We might accept it with some probability to escape.
                
                edges_list = [(i, j) for (i, j) in all_pairs if j in adj[i]]
                if edges_list:
                    a, b = random.choice(edges_list)
                    # Temporarily remove (a,b)
                    adj[a].remove(b)
                    adj[b].remove(a)
                    
                    # Get non-edges in this state
                    non_edges_state = [(i, j) for (i, j) in all_pairs if j not in adj[i]]
                    random.shuffle(non_edges_state)
                    
                    added_edges = []
                    # Try to add 2 edges
                    for (i, j) in non_edges_state:
                        if len(added_edges) == 2:
                            break
                        if not creates_c6(adj, i, j):
                            adj[i].add(j)
                            adj[j].add(i)
                            added_edges.append((i, j))
                    
                    current_edges = count_edges(adj)
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [set(s) for s in adj]
                        no_improve = 0
                    else:
                        # Revert
                        for (i, j) in added_edges:
                            adj[i].remove(j)
                            adj[j].remove(i)
                        adj[a].add(b)
                        adj[b].add(a)
                        
                        # With some probability, accept a neutral move (net 0) to escape local optimum
                        # But we must ensure we don't degrade.
                        # If current_edges == best_edges (which is true if we reverted), we are back to same state.
                        # To escape, we should accept a move that results in SAME number of edges but different structure.
                        # The swap removed 1 and added k. Net change = k - 1.
                        # If k=1, net 0.
                        # Let's try: if k==1, accept with prob 0.5?
                        # If we accept, we keep the new state.
                        if len(added_edges) == 1:
                            if random.random() < 0.5:
                                # Keep the state with 1 removal and 1 addition
                                # We already have it in adj (we didn't revert yet in this branch? 
                                # Wait, the code above reverts if not improved.
                                # Let's restructure.
                                pass
                    # Simplified: Just revert if not strictly better.
                    # The "escape" mechanism is weak. 
                    # Better escape: Random restart.
        
        return best_adj, best_edges

    # Main loop
    best_adj = None
    best_edges = 0
    
    while time.time() < deadline:
        # Greedy init
        adj = greedy_build()
        e = count_edges(adj)
        
        # Local search
        adj, e = local_search(adj, deadline)
        
        if e > best_edges:
            best_edges = e
            best_adj = [set(s) for s in adj]
            
        # If we hit a high score, we can try to polish or just continue
        # 105 is the target.
        if best_edges >= 105:
            break

    if best_adj is None:
        # Fallback
        best_adj = [set() for _ in range(N)]
        
    edges = []
    for i in range(N):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
    edges.sort()
    return edges
# EVOLVE-BLOCK-END
