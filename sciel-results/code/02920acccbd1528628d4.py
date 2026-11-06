import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118
    
    n = 20
    total_pairs = n * (n - 1) // 2
    
    # Precompute all triples
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
    num_triples = len(triples)
    
    # triple_common[i,j,k] = number of common neighbors of {i,j,k}
    # We need this to stay <= 2 (no K3,3 means no 3 common neighbors)
    
    def triple_index(i, j, k):
        """Map (i,j,k) with i<j<k to index in triples list."""
        # Precomputed lookup
        return triple_idx[i][j][k]
    
    # Build triple index lookup
    triple_idx = {}
    for idx, (i, j, k) in enumerate(triples):
        triple_idx[(i, j, k)] = idx
    
    def get_triple_count(adj, i, j, k):
        """Count common neighbors of vertices i, j, k."""
        return (adj[i] & adj[j] & adj[k]).bit_count()
    
    def is_valid_addition(adj, common, u, v):
        """Check if adding edge (u,v) keeps all triples with <= 2 common neighbors.
        common[t] = number of common neighbors of triple t."""
        if adj[u] & (1 << v):
            return False
        
        # For any triple (u, a, b) where a,b are neighbors of v:
        # Adding (u,v) increases common neighbors of (u,a,b) by 1 (v becomes common)
        # So we need current common[u,a,b] <= 1
        # Similarly for (v, a, b) where a,b are neighbors of u
        
        # Check triples containing u: (u, a, b) where a,b in N(v)
        nv = adj[v]
        na = adj[u]  # neighbors of u (before adding v)
        
        # For each pair (a, b) both in N(v), check if triple (u, a, b) has common > 1
        # Actually, we need: for triples (u, a, b) where v is in N(a) ∩ N(b),
        # adding edge (u,v) makes v a common neighbor of (u,a,b).
        # So common[u,a,b] increases by 1. Need old common[u,a,b] <= 1.
        
        # More precisely: v is added to N(u). For any triple (u, a, b) where a,b ∈ N(v):
        # v ∈ N(a) and v ∈ N(b), so v is a common neighbor of (u,a,b) after adding.
        # Before adding, common[u,a,b] = |N(u) ∩ N(a) ∩ N(b)|
        # After adding, it becomes common[u,a,b] + 1 (since v is now in N(u) ∩ N(a) ∩ N(b))
        # We need this to be <= 2, so old common[u,a,b] <= 1.
        
        # But we also need to check: does adding (u,v) affect other triples?
        # A triple (x, y, z) is affected if exactly one of x,y,z is u or v,
        # and the new edge creates a new common neighbor.
        
        # Case 1: Triple contains u but not v: (u, a, b)
        #   v is now in N(u). If v ∈ N(a) and v ∈ N(b), then v is a new common neighbor.
        #   Need: common[u,a,b] (old) <= 1
        #   Condition: v ∈ N(a) and v ∈ N(b), i.e., a,b ∈ N(v)
        
        # Case 2: Triple contains v but not u: (v, a, b)
        #   u is now in N(v). If u ∈ N(a) and u ∈ N(b), then u is a new common neighbor.
        #   Need: common[v,a,b] (old) <= 1
        #   Condition: u ∈ N(a) and u ∈ N(b), i.e., a,b ∈ N(u)
        
        # Case 3: Triple contains both u and v: (u, v, a)
        #   No new common neighbor is created because the edge (u,v) doesn't change
        #   N(u) ∩ N(v) ∩ N(a) in a way that adds a new vertex.
        #   Actually, N(u) gains v, N(v) gains u. 
        #   Common neighbors of (u,v,a): N(u) ∩ N(v) ∩ N(a)
        #   Before: N(u) ∩ N(v) ∩ N(a) (v not in N(u), u not in N(v))
        #   After: (N(u)∪{v}) ∩ (N(v)∪{u}) ∩ N(a)
        #   = [N(u)∩N(v)∩N(a)] ∪ [v∩N(v)∩N(a)] ∪ [N(u)∩u∩N(a)] ∪ [v∩u∩N(a)]
        #   v∩N(v) = ∅ (no self-loops), u∩N(u) = ∅, v∩u = ∅
        #   So common neighbors unchanged! No constraint from triples containing both.
        
        # So we only need to check cases 1 and 2.
        
        # Efficient check using bitmasks:
        # For case 1: find all pairs (a,b) in N(v) such that common[u,a,b] > 1
        # For case 2: find all pairs (a,b) in N(u) such that common[v,a,b] > 1
        
        # Since n=20, N(v) has at most 19 elements, so at most C(19,2)=171 pairs.
        # We can iterate over them.
        
        # Get list of neighbors of v (excluding u since edge doesn't exist yet)
        nv_list = []
        for a in range(n):
            if a != v and (nv & (1 << a)):
                nv_list.append(a)
        
        # Check case 1: triples (u, a, b) where a,b in nv_list
        for idx_a in range(len(nv_list)):
            a = nv_list[idx_a]
            if a == u:
                continue
            for idx_b in range(idx_a + 1, len(nv_list)):
                b = nv_list[idx_b]
                if b == u:
                    continue
                # Get common count for triple (u, a, b)
                i, j, k = sorted([u, a, b])
                t_idx = triple_idx[(i, j, k)]
                if common[t_idx] > 1:
                    return False
        
        # Get list of neighbors of u (excluding v)
        nu_list = []
        for a in range(n):
            if a != u and (na & (1 << a)):
                nu_list.append(a)
        
        # Check case 2: triples (v, a, b) where a,b in nu_list
        for idx_a in range(len(nu_list)):
            a = nu_list[idx_a]
            if a == v:
                continue
            for idx_b in range(idx_a + 1, len(nu_list)):
                b = nu_list[idx_b]
                if b == v:
                    continue
                i, j, k = sorted([v, a, b])
                t_idx = triple_idx[(i, j, k)]
                if common[t_idx] > 1:
                    return False
        
        return True
    
    def add_edge(adj, common, u, v):
        """Add edge (u,v) and update common neighbor counts for all affected triples."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update common counts for triples affected by adding (u,v)
        # Case 1: triples (u, a, b) where a,b ∈ N(v): v becomes new common neighbor
        nv = adj[v] & ~(1 << u)  # neighbors of v, excluding u
        for a in range(n):
            if a == u or a == v:
                continue
            if not (nv & (1 << a)):
                continue
            for b in range(a+1, n):
                if b == u or b == v:
                    continue
                if not (nv & (1 << b)):
                    continue
                i, j, k = sorted([u, a, b])
                t_idx = triple_idx[(i, j, k)]
                common[t_idx] += 1
        
        # Case 2: triples (v, a, b) where a,b ∈ N(u): u becomes new common neighbor
        nu = adj[u] & ~(1 << v)  # neighbors of u, excluding v
        for a in range(n):
            if a == u or a == v:
                continue
            if not (nu & (1 << a)):
                continue
            for b in range(a+1, n):
                if b == u or b == v:
                    continue
                if not (nu & (1 << b)):
                    continue
                i, j, k = sorted([v, a, b])
                t_idx = triple_idx[(i, j, k)]
                common[t_idx] += 1
    
    def remove_edge(adj, common, u, v):
        """Remove edge (u,v) and update common neighbor counts."""
        # First, revert the common count changes
        nv = adj[v] & ~(1 << u)  # neighbors of v, excluding u (after removal, same as before)
        # Actually, before removal, adj[v] has u. After removal, it won't.
        # The neighbors of v that are NOT u are the same before and after.
        # But we need to subtract the contributions that were added when (u,v) was added.
        
        # When (u,v) was added, for each (a,b) in N(v) (excluding u), 
        # common[u,a,b] was incremented.
        # Now we decrement.
        for a in range(n):
            if a == u or a == v:
                continue
            if not (nv & (1 << a)):
                continue
            for b in range(a+1, n):
                if b == u or b == v:
                    continue
                if not (nv & (1 << b)):
                    continue
                i, j, k = sorted([u, a, b])
                t_idx = triple_idx[(i, j, k)]
                common[t_idx] -= 1
        
        nu = adj[u] & ~(1 << v)
        for a in range(n):
            if a == u or a == v:
                continue
            if not (nu & (1 << a)):
                continue
            for b in range(a+1, n):
                if b == u or b == v:
                    continue
                if not (nu & (1 << b)):
                    continue
                i, j, k = sorted([v, a, b])
                t_idx = triple_idx[(i, j, k)]
                common[t_idx] -= 1
        
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2
    
    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = [0] * n
        common = [0] * num_triples
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if is_valid_addition(adj, common, u, v):
                add_edge(adj, common, u, v)
                
        return adj, common
    
    def local_search(adj, common, max_steps=100, use_kick=False, max_kicks=3):
        current_adj = adj[:]
        current_common = common[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_common = current_common[:]
        best_edges = current_edges
        
        # Phase 1: Greedy Add - try to add any valid missing edge
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            pairs = [(i, j) for i in range(n) for j in range(i+1, n) 
                    if not (current_adj[i] & (1 << j))]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if time.time() > deadline:
                    break
                if is_valid_addition(current_adj, current_common, u, v):
                    add_edge(current_adj, current_common, u, v)
                    current_edges += 1
                    improved = True
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                        best_common = current_common[:]
            
        # Phase 2: Edge Swaps (remove 1, add 2)
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1
            
            edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                    if current_adj[i] & (1 << j)]
            if not edges:
                break
            random.shuffle(edges)
            
            found_swap = False
            for u, v in edges:
                if time.time() > deadline:
                    break
                if found_swap:
                    break
                
                # Remove edge (u,v)
                remove_edge(current_adj, current_common, u, v)
                current_edges -= 1
                
                # Try to add 2 new edges
                missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                         if not (current_adj[i] & (1 << j))]
                random.shuffle(missing)
                
                added = 0
                added_edges = []
                for a, b in missing:
                    if time.time() > deadline:
                        break
                    if added == 2:
                        break
                    if is_valid_addition(current_adj, current_common, a, b):
                        add_edge(current_adj, current_common, a, b)
                        added += 1
                        added_edges.append((a, b))
                
                if added >= 1:
                    current_edges += added
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                        best_common = current_common[:]
                    found_swap = True
                else:
                    # Revert
                    add_edge(current_adj, current_common, u, v)
                    current_edges += 1
                    
        # Phase 3: Kick + Greedy Fill
        if use_kick:
            for _ in range(max_kicks):
                if time.time() > deadline:
                    break
                
                kick_adj = current_adj[:]
                kick_common = current_common[:]
                kick_edges = current_edges
                
                # Remove 2-4 random edges
                kick_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) 
                                  if kick_adj[i] & (1 << j)]
                random.shuffle(kick_edges_list)
                num_remove = random.randint(2, 4)
                for u, v in kick_edges_list[:num_remove]:
                    remove_edge(kick_adj, kick_common, u, v)
                    kick_edges -= 1
                
                # Greedy fill
                improved = True
                while improved:
                    if time.time() > deadline:
                        break
                    improved = False
                    pairs = [(i, j) for i in range(n) for j in range(i+1, n) 
                            if not (kick_adj[i] & (1 << j))]
                    random.shuffle(pairs)
                    
                    for u, v in pairs:
                        if time.time() > deadline:
                            break
                        if is_valid_addition(kick_adj, kick_common, u, v):
                            add_edge(kick_adj, kick_common, u, v)
                            kick_edges += 1
                            improved = True
                            if kick_edges > best_edges:
                                best_edges = kick_edges
                                best_adj = kick_adj[:]
                                best_common = kick_common[:]
                
                if kick_edges >= current_edges:
                    current_adj = kick_adj
                    current_common = kick_common
                    current_edges = kick_edges
        
        return best_adj, best_common, best_edges
    
    best_adj = [0] * n
    best_common = [0] * num_triples
    best_edges = 0
    
    # Run multiple restarts
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj, common = greedy_construct(i * 7 + 1)
        adj, common, edges = local_search(adj, common, max_steps=80, use_kick=True, max_kicks=2)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            best_common = common[:]
            
    # Final polish with more steps
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            adj, common, edges = local_search(best_adj, best_common, max_steps=200, use_kick=True, max_kicks=4)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj[:]
                best_common = common[:]
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
