"""
K2,3-free graph construction on 20 vertices.
Uses the polarity graph of the projective plane PG(2,4) (21 vertices, 40 edges per vertex),
deletes one vertex to get 20 vertices, then applies bitmask-based local search.
"""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute bit masks
    bit = [1 << i for i in range(n)]
    
    def init_adj():
        return [0] * n
    
    def count_edges(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2
    
    def common_count(adj, u, v):
        """Count common neighbors of u and v."""
        cn = adj[u] & adj[v]
        return bin(cn).count('1')
    
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) maintains K2,3-freeness.
        """
        if common_count(adj, u, v) > 2:
            return False
        
        nv = adj[v]
        w = 0
        while nv:
            low = nv & (-nv)
            w = low.bit_length() - 1
            if w != u:
                if common_count(adj, u, w) > 1:
                    return False
            nv ^= low
        
        nu = adj[u]
        w = 0
        while nu:
            low = nu & (-nu)
            w = low.bit_length() - 1
            if w != v:
                if common_count(adj, v, w) > 1:
                    return False
            nu ^= low
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= bit[v]
        adj[v] |= bit[u]
    
    def remove_edge(adj, u, v):
        adj[u] &= ~bit[v]
        adj[v] &= ~bit[u]
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return edges
    
    def greedy_fill(adj, order):
        """Add all valid edges from the given order."""
        for u, v in order:
            if not ((adj[u] >> v) & 1):
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
    
    def local_search(adj, max_iters=500, deadline=deadline):
        """
        Local search via edge swaps and removals.
        """
        best_adj = adj[:]
        best_count = count_edges(best_adj)
        current_adj = adj[:]
        current_count = best_count
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try a swap: remove one edge, then greedy fill
            cur_edges = get_edges(current_adj)
            
            if not cur_edges:
                break
            
            idx = random.randint(0, len(cur_edges) - 1)
            u_rem, v_rem = cur_edges[idx]
            
            backup = current_adj[:]
            remove_edge(current_adj, u_rem, v_rem)
            
            # Greedy fill
            missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                       if not ((current_adj[i] >> j) & 1)]
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(current_adj, u, v):
                    add_edge(current_adj, u, v)
            
            new_count = count_edges(current_adj)
            
            if new_count > best_count:
                best_adj = current_adj[:]
                best_count = new_count
                current_adj = best_adj[:]
                no_improve = 0
            else:
                current_adj = backup[:]
                no_improve += 1
                if no_improve > 40:
                    break
        
        return best_adj, best_count
    
    def edge_swap_local_search(adj, max_iters=300, deadline=deadline):
        """
        More aggressive local search: try swapping one edge for another.
        """
        best_adj = adj[:]
        best_count = count_edges(best_adj)
        current_adj = adj[:]
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            cur_edges = get_edges(current_adj)
            if not cur_edges:
                break
            
            # Pick an edge to remove
            u_rem, v_rem = cur_edges[random.randint(0, len(cur_edges) - 1)]
            
            # Pick a missing edge to add
            missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                       if not ((current_adj[i] >> j) & 1)]
            if not missing:
                break
            
            u_add, v_add = missing[random.randint(0, len(missing) - 1)]
            
            backup = current_adj[:]
            remove_edge(current_adj, u_rem, v_rem)
            
            if is_valid_add(current_adj, u_add, v_add):
                add_edge(current_adj, u_add, v_add)
                
                # Now try to add more edges greedily
                still_missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                                if not ((current_adj[i] >> j) & 1)]
                random.shuffle(still_missing)
                for u, v in still_missing:
                    if is_valid_add(current_adj, u, v):
                        add_edge(current_adj, u, v)
                
                new_count = count_edges(current_adj)
                if new_count > best_count:
                    best_adj = current_adj[:]
                    best_count = new_count
                    no_improve = 0
                else:
                    current_adj = backup[:]
                    no_improve += 1
            else:
                current_adj = backup[:]
                no_improve += 1
            
            if no_improve > 50:
                break
        
        return best_adj, best_count
    
    def build_polarity_graph():
        """
        Build the polarity graph of PG(2,4) on 21 vertices.
        Vertices are 1D subspaces of GF(4)^3, i.e., points of the projective plane.
        Two vertices are adjacent if their corresponding points are orthogonal under a fixed polarity.
        
        For GF(4) = {0, 1, α, α+1} where α^2 + α + 1 = 0 (so α^2 = α+1, α^3 = 1).
        
        We use the Hermitian polarity: x · y = x0*y0 + x1*y1 + x2*y2 where * is Frobenius automorphism.
        In GF(4), the Frobenius map sends x -> x^2.
        
        Actually, let's use a simpler approach: enumerate all 21 points of PG(2,4) and 
        define adjacency via a symmetric bilinear form.
        """
        # GF(4) elements: 0, 1, a, b where a^2 = a+1, b = a+1, a*b = 1
        # Multiplication table for GF(4):
        #   * | 0 1 a b
        #   ---+------
        #   0 | 0 0 0 0
        #   1 | 0 1 a b
        #   a | 0 a b 1
        #   b | 0 b 1 a
        
        gf4 = [0, 1, 2, 3]  # 0, 1, a, b
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        
        def gf4_add(x, y):
            return add_table[x][y]
        
        def gf4_mul(x, y):
            return mul_table[x][y]
        
        def gf4_frob(x):
            # Frobenius: x -> x^2
            # 0->0, 1->1, a->a^2=a+1=b, b->b^2=(a+1)^2=a^2+2a+1=a+1+2a+1=a (since 2a=0 in char 2)
            # Wait, in char 2: (a+1)^2 = a^2 + 1 = (a+1) + 1 = a
            # So: 0->0, 1->1, 2(a)->2(a), 3(b)->2(a)? Let me recompute.
            # a^2 = a + 1 = b, so frob(2) = 3
            # b = a+1, b^2 = (a+1)^2 = a^2 + 1 = (a+1) + 1 = a, so frob(3) = 2
            frob_map = {0: 0, 1: 1, 2: 3, 3: 2}
            return frob_map[x]
        
        # Enumerate all 1D subspaces (points) of GF(4)^3
        # A point is represented by a canonical representative (first nonzero coordinate is 1)
        points = []
        seen = set()
        
        for x0 in gf4:
            for x1 in gf4:
                for x2 in gf4:
                    if x0 == 0 and x1 == 0 and x2 == 0:
                        continue
                    # Canonical form: scale so first nonzero is 1
                    if x0 != 0:
                        rep = (1, gf4_mul(x1, gf4_mul(x0, 3)), gf4_mul(x2, gf4_mul(x0, 3)))
                        # x0^{-1} = x0^2 in GF(4) for nonzero x0
                        # Actually in GF(4), x^3 = 1 for x != 0, so x^{-1} = x^2
                        # frob(x0) = x0^2 = x0^{-1} for x0 != 0
                        inv_x0 = gf4_frob(x0)
                        rep = (1, gf4_mul(x1, inv_x0), gf4_mul(x2, inv_x0))
                    elif x1 != 0:
                        inv_x1 = gf4_frob(x1)
                        rep = (0, 1, gf4_mul(x2, inv_x1))
                    else:
                        rep = (0, 0, 1)
                    
                    if rep not in seen:
                        seen.add(rep)
                        points.append(rep)
        
        # Should have 21 points
        assert len(points) == 21, f"Expected 21 points, got {len(points)}"
        
        # Define the polarity: use the Hermitian form h(x,y) = x0*y0 + x1*y1 + x2*y2
        # where * is the Frobenius automorphism
        # Two points are adjacent if h(x, y) = 0 (orthogonal)
        
        def hermitian(x, y):
            # h(x,y) = sum_i x_i * y_i^{frob}
            result = 0
            for i in range(3):
                term = gf4_mul(x[i], gf4_frob(y[i]))
                result = gf4_add(result, term)
            return result
        
        # Build adjacency for 21 vertices
        adj21 = [0] * 21
        for i in range(21):
            for j in range(i + 1, 21):
                if hermitian(points[i], points[j]) == 0:
                    adj21[i] |= (1 << j)
                    adj21[j] |= (1 << i)
        
        return adj21, points
    
    def delete_vertex_and_reindex(adj21, delete_idx):
        """Delete vertex delete_idx from adj21 and reindex remaining 20 vertices."""
        n21 = 21
        adj20 = [0] * n  # 20 vertices
        
        # Mapping from old indices to new indices
        old_to_new = {}
        new_idx = 0
        for i in range(n21):
            if i != delete_idx:
                old_to_new[i] = new_idx
                new_idx += 1
        
        # Rebuild adjacency
        for i in range(n21):
            if i == delete_idx:
                continue
            ni = old_to_new[i]
            for j in range(n21):
                if j == i or j == delete_idx:
                    continue
                if (adj21[i] >> j) & 1:
                    nj = old_to_new[j]
                    adj20[ni] |= (1 << nj)
        
        return adj20
    
    # Main strategy:
    # 1. Build polarity graph of PG(2,4), delete one vertex
    # 2. Apply local search to maximize edges
    # 3. Also try random restarts as backup
    
    best_adj_overall = None
    best_count_overall = 0
    
    # Strategy 1: Polarity graph construction
    try:
        adj21, points = build_polarity_graph()
        
        # Try deleting each vertex and see which gives the best result after local search
        for del_idx in range(21):
            if time.time() > deadline - 20:
                break
            
            adj = delete_vertex_and_reindex(adj21, del_idx)
            cnt = count_edges(adj)
            
            # Local search to add more edges
            adj, cnt = local_search(adj, max_iters=200)
            
            if cnt > best_count_overall:
                best_count_overall = cnt
                best_adj_overall = adj[:]
            
            # Edge swap refinement
            adj2, cnt2 = edge_swap_local_search(adj, max_iters=100)
            if cnt2 > best_count_overall:
                best_count_overall = cnt2
                best_adj_overall = adj2[:]
            
            # Early exit if we hit a high count
            if best_count_overall >= 120:
                break
    except Exception as e:
        pass  # Fall through to random restarts
    
    # Strategy 2: Random restarts with varied greedy orderings (backup)
    if best_count_overall < 100:
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            adj = init_adj()
            
            # Different ordering strategies
            strategy = r % 5
            order = all_pairs[:]
            
            if strategy == 0:
                random.shuffle(order)
            elif strategy == 1:
                order.sort(key=lambda x: x[0] + x[1])
                random.shuffle(order)
            elif strategy == 2:
                order.sort(key=lambda x: abs(x[1] - x[0]))
                random.shuffle(order)
            elif strategy == 3:
                order.sort(key=lambda x: x[0])
                random.shuffle(order)
            else:
                random.shuffle(order)
            
            # Greedy construct
            greedy_fill(adj, order)
            
            # Local search
            adj, cnt = local_search(adj, max_iters=200)
            
            if cnt > best_count_overall:
                best_count_overall = cnt
                best_adj_overall = adj[:]
            
            # Edge swap refinement
            adj2, cnt2 = edge_swap_local_search(adj, max_iters=100)
            if cnt2 > best_count_overall:
                best_count_overall = cnt2
                best_adj_overall = adj2[:]
            
            if best_count_overall >= 120:
                break
    
    # Strategy 3: Final intensive local search on best result
    if best_adj_overall is not None and time.time() < deadline - 10:
        adj_final, cnt_final = local_search(best_adj_overall, max_iters=500)
        if cnt_final > best_count_overall:
            best_count_overall = cnt_final
            best_adj_overall = adj_final[:]
        
        adj_final2, cnt_final2 = edge_swap_local_search(best_adj_overall, max_iters=300)
        if cnt_final2 > best_count_overall:
            best_count_overall = cnt_final2
            best_adj_overall = adj_final2[:]
    
    # Convert best_adj to edge list
    if best_adj_overall is None:
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_overall[i] >> j) & 1:
                edges.append([i, j])
    
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
