import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    total_pairs = n * (n - 1) // 2
    
    def is_valid_addition(adj, u, v):
        if adj[u] & (1 << v):
            return False
        
        # Precompute neighbors of u and v
        nu = adj[u]
        nv = adj[v]
        
        # 1. Check triples {u, a, b} where a, b in N(v)
        # v becomes a common neighbor of {u, a, b}
        # We need |N(u) ∩ N(a) ∩ N(b) \ {v}| <= 1
        # Since v is not in adj[u] yet, and we are checking before adding,
        # adj[u] does not contain v.
        # adj[a] and adj[b] might contain v?
        # a, b are in N(v), so v is in N(a) and N(b).
        # So we need to exclude v from the intersection.
        # But v is not in nu.
        # Is v in adj[a]? Yes.
        # So (nu & adj[a] & adj[b]) includes v?
        # No, nu does not have v. So the intersection does not have v.
        # So we just check bit_count(nu & adj[a] & adj[b]) <= 1.
        
        # Iterate a in N(v)
        a = nv
        while a:
            la = a & -a
            a ^= la
            i = la.bit_length() - 1
            if i == u or i == v:
                continue
            # Iterate b in N(v) with b > i
            b = nv & ((1 << i) - 1) # Only check b < i to avoid duplicates? 
            # Actually, we need all pairs a, b in N(v).
            # Let's just iterate all a, b in N(v) with a < b.
            # To be efficient, we can do:
            # for a in N(v): for b in N(v) \ {a}:
            # But this is O(k^2).
            pass
            
        # Efficient way:
        # Get list of neighbors of v
        nbs_v = [i for i in range(n) if (nv >> i) & 1 and i != u]
        for i in range(len(nbs_v)):
            a = nbs_v[i]
            for j in range(i + 1, len(nbs_v)):
                b = nbs_v[j]
                if a == u or b == u:
                    continue
                # Check triple {u, a, b}
                # Common neighbors of u, a, b
                # v is a common neighbor (since a,b in N(v) and we are adding u-v so u in N(v))
                # Wait, u is in N(v) after addition.
                # So v is in N(u) (new), N(a), N(b).
                # So v is a common neighbor.
                # We need total common neighbors <= 2.
                # So existing common neighbors (excluding v) must be <= 1.
                # Existing common neighbors of u, a, b:
                # nu & adj[a] & adj[b]
                # Does this include v?
                # nu does not have v.
                # So it does not include v.
                # So we check bit_count(nu & adj[a] & adj[b]) <= 1.
                if (nu & adj[a] & adj[b]).bit_count() > 1:
                    return False
        
        # 2. Check triples {v, a, b} where a, b in N(u)
        nbs_u = [i for i in range(n) if (nu >> i) & 1 and i != v]
        for i in range(len(nbs_u)):
            a = nbs_u[i]
            for j in range(i + 1, len(nbs_u)):
                b = nbs_u[j]
                if a == v or b == v:
                    continue
                # Check triple {v, a, b}
                # u is a common neighbor (since a,b in N(u) and we are adding v-u so v in N(u))
                # So u is in N(v) (new), N(a), N(b).
                # Existing common neighbors of v, a, b:
                # nv & adj[a] & adj[b]
                # Does this include u?
                # nv does not have u.
                # So it does not include u.
                # Check bit_count(nv & adj[a] & adj[b]) <= 1.
                if (nv & adj[a] & adj[b]).bit_count() > 1:
                    return False
        
        # 3. Check triples {a, b, c} subset of N(u)
        # u becomes a common neighbor of {a, b, c}
        # If a, b, c in N(u) ∩ N(v), then v also becomes a common neighbor.
        # So for {a, b, c} in N(u) ∩ N(v):
        #   New common neighbors = Old + {u, v}
        #   We need Old + 2 <= 2 => Old = 0.
        #   Old common neighbors = adj[a] & adj[b] & adj[c]
        #   Does it include u or v?
        #   a, b, c in N(u) => u in N(a), N(b), N(c). So u is in Old?
        #   Wait, adj is the current state. u is NOT in N(a) yet?
        #   No, a is in N(u) means u is in N(a) in the current graph?
        #   Yes, graph is undirected.
        #   So u IS in adj[a].
        #   So u is in the intersection adj[a] & adj[b] & adj[c].
        #   Similarly, if a, b, c in N(v), v is in the intersection.
        
        # So:
        # Let I = adj[a] & adj[b] & adj[c].
        # If a, b, c in N(u) ∩ N(v):
        #   u in I, v in I.
        #   We add u-v.
        #   Does this change N(a)? No.
        #   Wait, adding u-v does not change N(a) unless a is u or v.
        #   a, b, c are distinct from u, v.
        #   So N(a), N(b), N(c) do NOT change!
        #   
        #   My previous reasoning was wrong.
        #   Adding edge (u, v) only changes N(u) and N(v).
        #   It does NOT change N(x) for x != u, v.
        #   
        #   So the common neighbors of {a, b, c} do NOT change!
        #   UNLESS u or v is in {a, b, c}.
        #   
        #   So we only need to check triples containing u or v.
        #   
        #   Let's re-verify.
        #   Condition: No 3 vertices have 3 common neighbors.
        #   If we add (u, v), the only sets of neighbors that change are N(u) and N(v).
        #   For any triple {x, y, z}, the common neighbors are N(x) ∩ N(y) ∩ N(z).
        #   This set changes only if u or v is in {x, y, z}.
        #   Because if u not in {x, y, z}, then N(u) is not involved in the intersection.
        #   So N(x), N(y), N(z) are unchanged.
        #   
        #   So the original code's logic was CORRECT in scope!
        #   It only needed to check triples containing u or v.
        
        # So why did I think it was wrong?
        # "u becomes a common neighbor of {a, b, c}"
        # This implies u is in N(a) ∩ N(b) ∩ N(c).
        # This is true if a, b, c in N(u).
        # But u is a vertex, not a neighbor.
        # The common neighbors are vertices.
        # If u is a common neighbor of {a, b, c}, it means u is connected to a, b, c.
        # This is a property of the graph.
        # Adding (u, v) does NOT change the connectivity of u to a, b, c.
        # So if u was a common neighbor before, it is still a common neighbor.
        # It does NOT become a common neighbor due to the addition of (u, v).
        #
        # So the original code was correct.
        
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return adj

    def local_search(adj, max_steps=200, use_kick=False):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        # Phase 1: Greedy Add
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if time.time() > deadline:
                    break
                if is_valid_addition(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges += 1
                    improved = True
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
            
        # Phase 2: Edge Swaps (1 in, 1 out) - Faster than 1 in 2 out?
        # Let's try 1-in 1-out first, then 2-in 1-out if needed.
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1
            
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            if not edges:
                break
            random.shuffle(edges)
            
            found_swap = False
            for u, v in edges:
                if time.time() > deadline:
                    break
                if found_swap:
                    break
                
                # Remove u-v
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                # Try to add one edge
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j)) and (i != u or j != v) and (i != v or j != u)]
                random.shuffle(missing)
                
                added = None
                for a, b in missing:
                    if time.time() > deadline:
                        break
                    if is_valid_addition(current_adj, a, b):
                        current_adj[a] |= (1 << b)
                        current_adj[b] |= (1 << a)
                        added = (a, b)
                        break
                
                if added:
                    # Edge count unchanged
                    # But structure changed, might allow future improvements
                    found_swap = True
                else:
                    # Revert
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
        return best_adj, best_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj = greedy_construct(i * 13 + 1)
        adj, edges = local_search(adj, max_steps=50, use_kick=False)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Final polish
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            adj, edges = local_search(best_adj, max_steps=100, use_kick=False)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj[:]
            else:
                break

    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
