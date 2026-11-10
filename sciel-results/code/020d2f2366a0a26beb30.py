"""Combined program: Fast bitmask C6 check + Greedy + 2-Swap Local Search."""
# EVOLVE-BLOCK-START
import time
import random
import sys

DEADLINE = 145.0  # Safety margin before 150s kill

def _check_c6_through_edge(adj, u, v, n=40):
    """
    Check if adding edge (u,v) creates a C6.
    A C6 containing edge (u,v) has the form u-v-a-b-c-d-u.
    This is equivalent to finding a path of length 4 from u to v in the graph 
    G - {u, v} (where u and v are removed, but we keep their neighbors for the path endpoints).
    Actually, simpler: 
    Path u - x1 - x2 - x3 - x4 - v.
    x1 in N(v) \ {u}
    x4 in N(u) \ {v, x1}
    x2 in N(x1) \ {v, x4, u}
    x3 in N(x4) \ {u, x2, v, x1}
    Edge (x2, x3) must exist.
    """
    # x1 in N(v) \ {u}
    x1_mask = adj[v] & ~(1 << u)
    while x1_mask:
        x1 = (x1_mask & (-x1_mask)).bit_length() - 1
        x1_mask &= x1_mask - 1
        
        # x4 in N(u) \ {v, x1}
        x4_mask = adj[u] & ~(1 << v) & ~(1 << x1)
        while x4_mask:
            x4 = (x4_mask & (-x4_mask)).bit_length() - 1
            x4_mask &= x4_mask - 1
            
            # x2 in N(x1) \ {v, x4, u}
            # Note: x2 cannot be u, v, x1, x4.
            # adj[x1] contains u? No, x1 is neighbor of v, not necessarily u. 
            # But if x1 is neighbor of u, then x2 could be u? 
            # Path is u-x1-x2... so x2 is neighbor of x1.
            # x2 cannot be u (would be u-x1-u... cycle of length 2? No, simple graph).
            # x2 cannot be v (would be u-x1-v... but v is endpoint).
            # x2 cannot be x4 (would be u-x1-x4-x4... no).
            x2_mask = adj[x1] & ~(1 << v) & ~(1 << x4) & ~(1 << u)
            while x2_mask:
                x2 = (x2_mask & (-x2_mask)).bit_length() - 1
                x2_mask &= x2_mask - 1
                
                # x3 in N(x4) \ {u, x2, v, x1}
                x3_mask = adj[x4] & ~(1 << u) & ~(1 << x2) & ~(1 << v) & ~(1 << x1)
                while x3_mask:
                    x3 = (x3_mask & (-x3_mask)).bit_length() - 1
                    x3_mask &= x3_mask - 1
                    
                    # Check if x2-x3 is an edge
                    if adj[x2] & (1 << x3):
                        return False  # C6 found
    return True  # No C6 through this edge

def _is_c6_free(adj_bits, n=40):
    """Full check: verify no C6 exists in the graph."""
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                if not _check_c6_through_edge(adj_bits, i, j, n):
                    return False
    return True

def _count_edges(adj_bits, n=40):
    count = 0
    for i in range(n):
        count += bin(adj_bits[i]).count('1')
    return count // 2

def _get_edges(adj_bits, n=40):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                edges.append([i, j])
    return edges

def _random_greedy(n=40, seed=None, max_time=2.0):
    """Randomized greedy: try edges in random order, add if C6-free."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    adj = [0] * n
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() - start_time > max_time:
            break
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not _check_c6_through_edge(adj, u, v, n):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    return adj, _count_edges(adj, n)

def _spectral_gap_score(adj, u, v, n=40):
    """
    Compute a heuristic score for adding edge (u,v) based on spectral gap.
    Higher spectral gap (lambda_1 - lambda_2) correlates with better mixing properties
    and potentially fewer long cycles. We use a cheap approximation:
    - Look at the degrees of u and v
    - Look at common neighbors (adding edge increases connectivity)
    - Penalize if u and v are already well-connected through short paths
    """
    deg_u = bin(adj[u]).count('1')
    deg_v = bin(adj[v]).count('1')
    
    # Common neighbors
    common = adj[u] & adj[v]
    common_count = bin(common).count('1')
    
    # Score: prefer edges between vertices with moderate degree and few common neighbors
    # This avoids creating triangles and dense local structures
    score = (deg_u + deg_v) * 0.1 - common_count * 0.5
    
    # Bonus if the edge connects different "clusters" (few common neighbors)
    if common_count <= 1:
        score += 1.0
    
    return score

def _local_search_2swap(adj, n=40, max_time=10.0):
    """
    Local search using 1-add and 2-swap moves.
    1-add: Add a non-edge if it doesn't create a C6.
    2-swap: Remove 2 edges, add 2 non-edges, if the result is C6-free.
    Aggressive perturbation: Remove 2-4 edges when stuck to escape local optima.
    Uses spectral gap heuristic to prioritize candidate edges.
    """
    start_time = time.time()
    current = _count_edges(adj, n)
    best_adj = adj[:]
    best_count = current
    
    stuck_counter = 0
    
    while time.time() - start_time < max_time:
        improved = False
        
        # 1. Try to add edges (1-add) with spectral gap prioritization
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        # Score and sort non-edges by spectral gap heuristic
        if len(non_edges) > 0:
            scored_non_edges = [(e, _spectral_gap_score(adj, e[0], e[1], n)) for e in non_edges]
            scored_non_edges.sort(key=lambda x: -x[1])  # Sort by score descending
            non_edges = [e for e, s in scored_non_edges]
        
        # Try top candidates first, then randomize the rest
        top_k = min(20, len(non_edges))
        candidates = non_edges[:top_k] + non_edges[top_k:]
        random.shuffle(candidates[top_k:])
        
        for u, v in candidates:
            if time.time() - start_time > max_time:
                return best_adj, best_count
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _check_c6_through_edge(adj, u, v, n):
                current += 1
                improved = True
                if current > best_count:
                    best_count = current
                    best_adj = adj[:]
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if improved:
            stuck_counter = 0
            # If we added edges, loop back to try more adds
            continue
            
        # No edges could be added - we're stuck
        stuck_counter += 1
        
        # 2. Try aggressive perturbation: remove 2-4 edges to escape local optimum
        cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if len(cur_edges) < 2 or len(non_edges) < 2:
            break
        
        # Determine perturbation strength based on how long we've been stuck
        if stuck_counter >= 3:
            num_remove = random.choice([3, 4])
        elif stuck_counter >= 2:
            num_remove = random.choice([2, 3])
        else:
            num_remove = 2
        
        # Try a few random perturbations
        perturbations_tried = 0
        max_perturbations = 500
        
        while perturbations_tried < max_perturbations and time.time() - start_time < max_time:
            perturbations_tried += 1
            
            # Pick num_remove distinct edges to remove
            if len(cur_edges) < num_remove:
                break
            remove_edges = random.sample(cur_edges, num_remove)
            
            # Pick num_remove distinct non-edges to add
            if len(non_edges) < num_remove:
                break
            add_edges = random.sample(non_edges, num_remove)
            
            # Apply removals
            saved_state = adj[:]
            for e in remove_edges:
                adj[e[0]] &= ~(1 << e[1])
                adj[e[1]] &= ~(1 << e[0])
            
            # Apply additions
            valid = True
            for e in add_edges:
                adj[e[0]] |= (1 << e[1])
                adj[e[1]] |= (1 << e[0])
                if not _check_c6_through_edge(adj, e[0], e[1], n):
                    valid = False
                    break
            
            if valid:
                # Accept the perturbation (edge count remains same, structure changes)
                improved = True
                # Update cur_edges and non_edges for next iteration
                cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
                non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                break
            else:
                # Revert to saved state
                adj = saved_state[:]
        
        if not improved:
            # No valid perturbation found, try again or break
            if stuck_counter >= 5:
                break
            pass

    return best_adj, best_count

def _known_good(n=40):
    """8 disjoint K5s = 100 edges, C6-free."""
    adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a+1, 5):
                adj[base+a] |= (1 << (base+b))
                adj[base+b] |= (1 << (base+a))
    return adj

def construct():
    start_time = time.time()
    n = 40
    best_adj = None
    best_edges = 0
    
    # Phase 1: Start from 8xK5
    adj = _known_good(n)
    best_adj = adj[:]
    best_edges = _count_edges(adj, n)
    
    # Phase 2: Local search with 2-swaps from 8xK5
    if time.time() - start_time < DEADLINE - 10:
        adj_ls, count_ls = _local_search_2swap(adj, n, max_time=min(10.0, DEADLINE - (time.time() - start_time) - 5))
        if count_ls > best_edges:
            best_edges = count_ls
            best_adj = adj_ls[:]
    
    # Phase 3: Multiple random greedy restarts + 2-swap
    for attempt in range(20):
        remaining = DEADLINE - (time.time() - start_time) - 5
        if remaining < 2:
            break
        
        # Short greedy
        mt_greedy = min(1.0, remaining / 4)
        adj_g, count_g = _random_greedy(n, seed=500 + attempt, max_time=mt_greedy)
        
        if count_g > best_edges:
            best_edges = count_g
            best_adj = adj_g[:]
            
        # Apply 2-swap local search on the greedy result
        # Allocate remaining time split between this attempt and future ones
        # Simple heuristic: give it 50% of remaining time if it's promising
        if count_g >= best_edges - 5:
            mt_ls = min(5.0, (remaining / 2))
            adj_l, count_l = _local_search_2swap(adj_g, n, max_time=mt_ls)
            if count_l > best_edges:
                best_edges = count_l
                best_adj = adj_l[:]
    
    # Final verification
    if best_adj is None:
        best_adj = _known_good(n)
    
    if not _is_c6_free(best_adj, n):
        best_adj = _known_good(n)
    
    return _get_edges(best_adj, n)
# EVOLVE-BLOCK-END
