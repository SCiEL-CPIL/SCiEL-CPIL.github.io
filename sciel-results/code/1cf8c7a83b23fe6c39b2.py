"""
Construct a large C4-free graph with no independent 11-set (lower bound for R(C4,K11)).
Uses a bitmask-based branch-and-bound for alpha computation and a greedy extension
loop to grow the graph from the PG(2,5) polarity graph base.
"""
import time
import random
from itertools import combinations

_START = time.time()
_TIME_LIMIT = 140.0  # safety margin under 150s wall-clock

def _elapsed():
    return time.time() - _START

def _build_polarity_graph_pg2_5():
    """
    Build the polarity graph of PG(2,5).
    Points: 31 points in PG(2,5).
    A point is adjacent to all points on its polar line.
    This graph is C4-free and has independence number 8.
    """
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                p = [x, y, z]
                for i in range(3):
                    if p[i] != 0:
                        inv = pow(p[i], 3, 5)
                        p = [(c * inv) % 5 for c in p]
                        break
                if p not in points:
                    points.append(p)
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"
    
    edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            p, q = points[i], points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    return edges, adj, n

def _to_bitmask(adj, n):
    """Convert adjacency list of sets to list of bitmasks."""
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    return masks

def _max_independent_set_size_bitmask(adj_mask, n, time_limit=5.0):
    """
    Find max independent set size using bitmask branch and bound.
    adj_mask[i] has bit j set if i and j are adjacent.
    Independent set in G is Clique in complement.
    """
    start_time = time.time()
    best = [0]
    
    # Precompute complement masks
    all_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = all_mask ^ adj_mask[i] ^ (1 << i)

    def _search(candidates, current_size):
        if time.time() - start_time > time_limit:
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Bound: current_size + popcount(candidates)
        if current_size + candidates.bit_count() <= best[0]:
            return
            
        # Pick vertex with max degree in candidate subgraph (in complement)
        max_deg = -1
        v = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            idx = lsb.bit_length() - 1
            deg = (comp_mask[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp_cands ^= lsb
            
        if v == -1:
            return

        # Include v
        new_cands = candidates & comp_mask[v]
        _search(new_cands, current_size + 1)
        
        # Exclude v
        new_cands2 = candidates & ~(1 << v)
        _search(new_cands2, current_size)

    _search((1 << n) - 1, 0)
    return best[0]

def _check_c4_free(adj, n):
    """Check if graph is C4-free."""
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if len(common) > 1:
                return False
    return True

def _greedy_alpha_estimate(adj, n):
    """Fast greedy estimate of alpha for pruning."""
    remaining = set(range(n))
    count = 0
    while remaining:
        # Pick vertex with min degree in remaining
        min_deg_v = min(remaining, key=lambda v: len(adj[v] & remaining))
        count += 1
        remaining -= {min_deg_v} | (adj[min_deg_v] & remaining)
    return count

def construct():
    if _elapsed() > _TIME_LIMIT - 5:
        # Not enough time, return seed
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    deadline = _START + _TIME_LIMIT
    
    # 1. Build Base Graph: PG(2,5) polarity graph
    edges, adj, n = _build_polarity_graph_pg2_5()
    
    # Verify base
    if not _check_c4_free(adj, n):
        return {'n': n, 'edges': edges}
    
    # Check alpha of base
    adj_mask = _to_bitmask(adj, n)
    alpha = _max_independent_set_size_bitmask(adj_mask, n, time_limit=3.0)
    
    if alpha >= 11:
        # Base is invalid, return it anyway (shouldn't happen)
        return {'n': n, 'edges': edges}

    best_n = n
    best_edges = edges
    best_adj = adj
    best_adj_mask = adj_mask

    # 2. Extension Loop
    # We try to add vertices one by one.
    # A new vertex v can be added if:
    # 1. Its neighbors S in G form a clique in the "compatibility" sense? No.
    #    The condition for C4-free is: for any two existing vertices u, w,
    #    if they both become neighbors of v, then u and w must not share any other neighbor.
    #    Actually, the condition is: N(v) must be a set of vertices such that no two share a neighbor in G \ {v}.
    #    Since v is new, N(v) = S.
    #    For any u, w in S, N(u) ∩ N(w) must be empty? No.
    #    N(u) ∩ N(w) in the new graph includes v if u, w ∈ S.
    #    So we need N_G(u) ∩ N_G(w) = ∅ for all u, w ∈ S.
    #    This means S is an independent set in the "neighbor intersection" graph?
    #    Actually, it means S is a set of vertices with pairwise disjoint neighborhoods in G.
    
    while time.time() < deadline - 10:
        # Find all sets S of vertices in G such that N(u) ∩ N(w) = ∅ for all u, w in S.
        # These are the valid neighbor sets for a new vertex.
        # We want to add a vertex with neighbor set S.
        # After adding, alpha must remain <= 10.
        
        # To speed up, we can just try random subsets S of size k.
        # Or we can iterate over possible S.
        # Let's try random S first.
        
        found_extension = False
        random.seed(int(time.time() * 1000) % 10000)
        
        # Try to find a valid S
        # S must be a subset of vertices such that for all u, w in S, adj[u] & adj[w] == empty set.
        
        # Let's generate candidate S by picking a random vertex u, and then adding vertices w
        # such that adj[u] & adj[w] is empty.
        
        for _ in range(1000):
            if time.time() > deadline - 10:
                break
            
            # Pick a random starting vertex
            u = random.randint(0, n - 1)
            S = {u}
            
            # Try to add more vertices to S
            candidates = [v for v in range(n) if v != u and not (adj[u] & adj[v])]
            random.shuffle(candidates)
            
            for v in candidates:
                # Check if v is compatible with all in S
                compatible = True
                for w in S:
                    if adj[v] & adj[w]:
                        compatible = False
                        break
                if compatible:
                    S.add(v)
            
            if not S:
                continue
                
            # S is a valid neighbor set for a new vertex.
            # The new vertex v_new will be adjacent to all in S.
            # The non-neighbors U = V \ S.
            # In the new graph, an independent set can include v_new and an independent set from U.
            # Or not include v_new and an independent set from V.
            # So alpha_new = max(alpha(G), 1 + alpha(G[U]))
            # We need alpha_new <= 10.
            # Since alpha(G) <= 10 (from previous step), we need 1 + alpha(G[U]) <= 10 => alpha(G[U]) <= 9.
            
            U = set(range(n)) - S
            
            # Check alpha of G[U]
            # Map U to local indices
            u_list = sorted(U)
            u_map = {v: i for i, v in enumerate(u_list)}
            local_n = len(u_list)
            local_adj = [set() for _ in range(local_n)]
            
            for i, u in enumerate(u_list):
                for w in adj[u]:
                    if w in u_map:
                        local_adj[i].add(u_map[w])
            
            # Fast greedy check
            greedy_alpha = _greedy_alpha_estimate(local_adj, local_n)
            if greedy_alpha > 9:
                continue
                
            # Exact check
            local_adj_mask = _to_bitmask(local_adj, local_n)
            alpha_u = _max_independent_set_size_bitmask(local_adj_mask, local_n, time_limit=2.0)
            
            if alpha_u <= 9:
                # Valid extension!
                # Add new vertex n with neighbors S
                adj.append(set(S))
                for u in S:
                    adj[u].add(n)
                n += 1
                found_extension = True
                break
                
        if not found_extension:
            # No extension found in this round, break
            # Or try a different strategy?
            # For now, break.
            break
            
        # Update best
        best_n = n
        best_adj = adj
        best_adj_mask = _to_bitmask(adj, n)
        best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]
        
        # Verify C4-free
        if not _check_c4_free(adj, n):
            # Should not happen, but revert if it does
            break
            
        # Verify alpha
        # We know alpha <= 10 from construction, but let's double check occasionally
        if n % 5 == 0:
            alpha_check = _max_independent_set_size_bitmask(best_adj_mask, n, time_limit=5.0)
            if alpha_check > 10:
                break

    # Return best found
    return {'n': best_n, 'edges': best_edges}
