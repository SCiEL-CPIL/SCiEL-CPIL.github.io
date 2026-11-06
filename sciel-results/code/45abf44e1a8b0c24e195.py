import time
import random
import numpy as np

def construct():
    """
    Construct a C4-free graph with alpha < 11, maximizing n.
    Strategy:
    1. Deterministically construct the polarity graph of PG(2, 5). This gives n=31.
    2. Verify it is C4-free and has alpha < 11.
    3. Use a structured local search / Max-SAT-like heuristic to extend the graph
       to n=39 by adding vertices and optimizing edges.
    4. Use a branch-and-bound to verify alpha < 11.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def time_left():
        return deadline - time.time()

    def popcount(x):
        return bin(x).count('1')

    def max_independent_set_size(n, adj):
        """
        Find the size of the maximum independent set using branch and bound.
        adj is a list of integers (bitmasks).
        Returns the size.
        """
        best = [0]
        
        def branch(candidates_mask, current_size):
            if time_left() < 5:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Heuristic: Pick vertex with highest degree in the candidate subgraph
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            
            if best_v == -1:
                return

            v_bit = 1 << best_v
            
            # Option 1: Include best_v
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Option 2: Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    def is_c4_free(n, adj):
        """Check if graph is C4-free."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    if popcount(common) > 1:
                        return False
        return True

    def construct_polarity_graph():
        """
        Construct the polarity graph of PG(2, 5).
        This gives a C4-free graph on 31 vertices.
        """
        # Points in PG(2, 5)
        points = []
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize
                    if x != 0:
                        inv_x = pow(x, -1, 5)
                        points.append((1, y * inv_x % 5, z * inv_x % 5))
                    elif y != 0:
                        inv_y = pow(y, -1, 5)
                        points.append((0, 1, z * inv_y % 5))
                    else:
                        points.append((0, 0, 1))
        points = list(set(points))
        n = len(points)
        
        # Bilinear form for polarity
        def bilinear(u, v):
            return (u[0]*v[2] + u[1]*v[1] + u[2]*v[0]) % 5
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if bilinear(points[i], points[j]) == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        return n, adj

    def can_add_edge(adj, n, i, j):
        """
        Check if adding edge (i, j) to a C4-free graph keeps it C4-free.
        """
        if i == j:
            return False
        if adj[i] & (1 << j):
            return True # Already exists
        
        # Check 1: |N(i) ∩ N(j)| <= 1
        common_ij = adj[i] & adj[j]
        if popcount(common_ij) > 1:
            return False
        
        # Check 2: For any k in N(i) \ N(j), N(k) ∩ N(j) must be empty?
        # Actually, the condition is: for any pair (a, b), |N(a) ∩ N(b)| <= 1.
        # Adding (i, j) creates new pairs involving i or j.
        # Pairs (i, k) for k in N(j):
        #   N(i) gains j. N(k) has j.
        #   Common neighbors of i and k in new graph:
        #   Old N(i) ∩ N(k) + {j if j in N(k)}
        #   We know k is in N(j), so j in N(k).
        #   So we need Old N(i) ∩ N(k) to be empty.
        #   i.e., for all k in N(j) \ N(i), N(i) ∩ N(k) must be empty.
        
        # Pairs (j, k) for k in N(i):
        #   Similarly, for all k in N(i) \ N(j), N(j) ∩ N(k) must be empty.
        
        # Pairs (k, l) where k in N(i), l in N(j)?
        #   N(k) and N(l) are unchanged.
        #   So no new C4s formed between existing vertices.
        
        # So the checks are:
        # 1. |N(i) ∩ N(j)| <= 1
        # 2. For all k in N(j) \ N(i): N(i) ∩ N(k) == 0
        # 3. For all k in N(i) \ N(j): N(j) ∩ N(k) == 0
        
        # Check 2
        nj = adj[j] & ~(1 << i)
        ni = adj[i] & ~(1 << j)
        
        # k in N(j) but not in N(i)
        candidates_k = nj & ~ni
        while candidates_k:
            lsb = candidates_k & -candidates_k
            k = lsb.bit_length() - 1
            if adj[i] & adj[k]:
                return False
            candidates_k -= lsb
            
        # Check 3
        # k in N(i) but not in N(j)
        candidates_k = ni & ~nj
        while candidates_k:
            lsb = candidates_k & -candidates_k
            k = lsb.bit_length() - 1
            if adj[j] & adj[k]:
                return False
            candidates_k -= lsb

        return True

    def maximize_edges(n, adj):
        """
        Add edges that do not create C4s to reduce alpha.
        """
        adj = adj[:]
        improved = True
        while improved and time_left() > 10:
            improved = False
            # Shuffle order for variety
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices:
                if time_left() < 10:
                    break
                for j in indices:
                    if i >= j:
                        continue
                    if not (adj[i] & (1 << j)):
                        if can_add_edge(adj, n, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            improved = True
        return adj

    def try_add_vertex(adj, n):
        """
        Try to add a new vertex to the graph.
        Returns (new_adj, new_n) if successful, else (adj, n).
        """
        new_n = n + 1
        new_adj = adj[:] + [0]
        
        # The new vertex v = n
        # We need to choose a subset of existing vertices to connect to.
        # Let S be the set of neighbors of v.
        # Constraints:
        # 1. For any i, j in S: |N(i) ∩ N(j) ∪ {v}| <= 1.
        #    Since v is new, N(i) ∩ N(j) is unchanged.
        #    But wait, v is in N(i) for all i in S.
        #    So for any i, j in S, v is a common neighbor.
        #    Thus we need |N(i) ∩ N(j)| <= 0 for all i, j in S.
        #    i.e., S must be an independent set in the "common neighbor conflict graph".
        #    Actually, if i and j are in S, they share neighbor v.
        #    If they share another neighbor k, then i-v-j-k-i is a C4?
        #    Edges: (i,v), (v,j), (j,k), (k,i). Yes.
        #    So S must be a set of vertices with pairwise empty intersection of neighbors.
        #    i.e., S is an independent set in the graph where edges connect vertices with common neighbors.
        #    Let's call this the "conflict graph" H.
        #    H has vertices 0..n-1. Edge (i, j) in H if |N(i) ∩ N(j)| >= 1.
        #    Wait, if |N(i) ∩ N(j)| >= 1, say {k}, then i-k-j is a path.
        #    If i, j both connect to v, then i-v-j-k-i is a C4.
        #    So yes, S must be an independent set in H.
        
        # 2. For any i in S, j not in S:
        #    v is in N(i), not in N(j).
        #    Pairs (i, j): N(i) gains v.
        #    Common neighbors: N(i) ∩ N(j).
        #    v is not in N(j). So common neighbors are unchanged.
        #    So no new C4s from this.
        
        # 3. For any i, j not in S:
        #    Unchanged.
        
        # So the only constraint is that S is an independent set in H.
        # Also, we want S to be "large" or "well-chosen" to keep alpha small.
        # Actually, adding a vertex increases n by 1.
        # To keep alpha < 11, the new vertex should not be in a large independent set.
        # If we connect v to many vertices, it's less likely to be in an independent set.
        # So we want S to be large?
        # But S must be an independent set in H.
        # H is dense?
        # In a C4-free graph, if degrees are high, H is likely dense.
        # In PG(2,5) polarity graph, degrees are around 6.
        # Common neighbors: |N(i) ∩ N(j)| <= 1.
        # So H has edges where |N(i) ∩ N(j)| = 1.
        
        # Let's build H.
        h_adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & adj[j]: # Common neighbor exists
                    h_adj[i] |= (1 << j)
                    h_adj[j] |= (1 << i)
        
        # Find a maximal independent set in H.
        # Use greedy: pick vertex with highest degree in H? Or lowest?
        # We want S to be large to maximize connections for v.
        # But we also need to ensure alpha doesn't blow up.
        # Let's try to find a maximal independent set in H.
        
        s_mask = 0
        candidates = (1 << n) - 1
        while candidates:
            # Pick vertex with highest degree in H within candidates
            best_v = -1
            best_deg = -1
            m = candidates
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(h_adj[v] & candidates)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            
            if best_v == -1:
                break
            
            # Add best_v to S
            s_mask |= (1 << best_v)
            # Remove best_v and its neighbors from candidates
            candidates &= ~(1 << best_v)
            candidates &= ~h_adj[best_v]
            
            if time_left() < 10:
                break
        
        # Connect v to all in S
        new_adj[n] = s_mask
        for i in range(n):
            if s_mask & (1 << i):
                new_adj[i] |= (1 << n)
        
        return new_adj, new_n

    # Main construction
    n, adj = construct_polarity_graph()
    
    # Verify initial graph
    if not is_c4_free(n, adj):
        return {'n': 1, 'edges': []}
    
    alpha = max_independent_set_size(n, adj)
    if alpha >= 11:
        return {'n': 1, 'edges': []}
    
    best_n = n
    best_adj = adj[:]
    
    # Try to extend to 39
    for target_n in range(n + 1, 40):
        if time_left() < 20:
            break
        
        # Try multiple random restarts for vertex addition
        success = False
        for seed in range(50):
            if time_left() < 10:
                break
            
            random.seed(seed * 1000 + target_n)
            
            # Start from best_adj
            cur_adj = best_adj[:]
            cur_n = best_n
            
            # Add vertices one by one
            for _ in range(target_n - cur_n):
                new_adj, new_n = try_add_vertex(cur_adj, cur_n)
                if new_n > cur_n:
                    cur_adj = new_adj
                    cur_n = new_n
                else:
                    break
            
            if cur_n == target_n:
                # Verify
                if is_c4_free(cur_n, cur_adj):
                    alpha = max_independent_set_size(cur_n, cur_adj)
                    if alpha < 11:
                        best_n = cur_n
                        best_adj = cur_adj[:]
                        success = True
                        break
                    else:
                        # Try to fix alpha by maximizing edges
                        fixed_adj = maximize_edges(cur_n, cur_adj)
                        if is_c4_free(cur_n, fixed_adj):
                            alpha2 = max_independent_set_size(cur_n, fixed_adj)
                            if alpha2 < 11:
                                best_n = cur_n
                                best_adj = fixed_adj[:]
                                success = True
                                break
        
        if not success:
            # If we failed to reach target_n, try to reach best_n + 1?
            # Or just stop?
            # If we can't add even one vertex, we are stuck.
            # But maybe we can add one vertex with a different seed?
            # The loop above tries 50 seeds. If none work, we stop extending.
            break
    
    # Final edge maximization to ensure alpha is minimized
    if time_left() > 10:
        final_adj = maximize_edges(best_n, best_adj)
        if is_c4_free(best_n, final_adj):
            alpha_final = max_independent_set_size(best_n, final_adj)
            if alpha_final < 11:
                best_adj = final_adj
            else:
                # Revert if it got worse
                pass
    
    # Convert to edge list
    edge_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])

    return {'n': best_n, 'edges': edge_list}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
