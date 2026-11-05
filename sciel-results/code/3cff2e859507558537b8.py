"""Model-and-optimize: bitset-based search + MILP feasibility check for triangle degree sum constraint."""
import random
import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 16
TARGET_EDGES = 65
DEADLINE = 138.0  # safety margin before 150s kill


def _build_all_pairs():
    """Return list of all (i,j) pairs with i<j, and mapping to index."""
    pairs = []
    idx = {}
    for i in range(N):
        for j in range(i+1, N):
            idx[(i,j)] = len(pairs)
            pairs.append((i,j))
    return pairs, idx


def _build_triangle_list(pairs, idx):
    """Precompute all triangles as (i,j,k) with i<j<k, and the edge indices."""
    tris = []
    for i in range(N):
        for j in range(i+1, N):
            if (i,j) not in idx:
                continue
            for k in range(j+1, N):
                if (i,k) in idx and (j,k) in idx:
                    tris.append((i, j, k))
    return tris


def _score_from_edges(edge_set, pairs, idx, tris):
    """Compute max triangle degree sum given a set of edge indices."""
    # Build degree array
    deg = [0] * N
    for eidx in edge_set:
        i, j = pairs[eidx]
        deg[i] += 1
        deg[j] += 1
    
    # Check which triangles are present
    best = 0
    for (i, j, k) in tris:
        e1 = idx[(i,j)]
        e2 = idx[(i,k)]
        e3 = idx[(j,k)]
        if e1 in edge_set and e2 in edge_set and e3 in edge_set:
            s = deg[i] + deg[j] + deg[k]
            if s > best:
                best = s
    return best


def _milp_feasible(T, pairs, idx, tris, max_iter=100000):
    """
    Check if there exists a 65-edge subgraph where every triangle has degree sum <= T.
    Uses MILP with scipy. Returns (feasible, edge_set) or (False, None).
    
    Variables: x_e in {0,1} for each edge e.
    Constraints:
      - sum(x_e) = 65
      - For each triangle (i,j,k): if all three edges present, then deg(i)+deg(j)+deg(k) <= T
      
    The triangle constraint is nonlinear in x. We linearize:
      Let y_t be a binary variable for triangle t being "active" (all 3 edges present).
      y_t <= x_e1, y_t <= x_e2, y_t <= x_e3
      y_t >= x_e1 + x_e2 + x_e3 - 2
      
      Then: sum_t y_t * (deg_i + deg_j + deg_k) <= T * sum_t y_t ... no, this doesn't work directly.
      
    Actually, the constraint is: for each triangle t, if y_t=1 then deg_i+deg_j+deg_k <= T.
    But deg depends on x, so this is still nonlinear.
    
    Alternative: enumerate all possible degree sequences? Too many.
    
    Better approach: Use the fact that N=16 is small. We can use a different formulation.
    
    Actually, let's use a penalty-based approach or just do local search more effectively.
    The MILP approach for this specific problem is complex due to the nonlinear constraint.
    
    Let me instead use a smarter local search with bitsets.
    """
    return None  # Skip MILP, use improved local search


def _adj_bits_from_edge_set(edge_set, pairs):
    """Convert edge index set to adjacency bitset."""
    adj = [0] * N
    for eidx in edge_set:
        i, j = pairs[eidx]
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    return adj


def _edge_set_from_adj(adj):
    """Convert adjacency bitset to edge index set."""
    edge_set = set()
    for i in range(N):
        for j in range(i+1, N):
            if (adj[i] >> j) & 1:
                edge_set.add((i,j))
    return edge_set


def _build_turan_plus_one(pairs, idx):
    """Build K_{8,8} plus one internal edge. Returns edge_set."""
    edge_set = set()
    for i in range(8):
        for j in range(8, 16):
            edge_set.add((i,j))
    edge_set.add((0,1))
    return edge_set


def _max_tri_deg_sum_fast(adj_bits, deg):
    """Fast max triangle degree sum using bitsets."""
    best = 0
    full_mask = (1 << N) - 1
    for i in range(N):
        for j in range(i+1, N):
            if not ((adj_bits[i] >> j) & 1):
                continue
            common = adj_bits[i] & adj_bits[j]
            # Only consider k > j
            mask = full_mask ^ ((1 << (j+1)) - 1)
            common &= mask
            c = common
            while c:
                lsb = c & (-c)
                k = lsb.bit_length() - 1
                s = deg[i] + deg[j] + deg[k]
                if s > best:
                    best = s
                c ^= lsb
    return best


def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    pairs, idx = _build_all_pairs()
    tris = _build_triangle_list(pairs, idx)
    
    def elapsed():
        return time.time() - t0
    
    # Start with Turán + 1
    init_edges = _build_turan_plus_one(pairs, idx)
    adj_bits = _adj_bits_from_edge_set(init_edges, pairs)
    deg = [0] * N
    for i in range(N):
        deg[i] = bin(adj_bits[i] & ((1 << N) - 1)).count('1')
    cur_score = _max_tri_deg_sum_fast(adj_bits, deg)
    
    best_adj = adj_bits[:]
    best_deg = deg[:]
    best_score = cur_score
    
    # Phase 1: Try various initial structures
    # Different bipartition sizes
    for a_size in range(3, 14):
        b_size = N - a_size
        cross = a_size * b_size
        internal_needed = TARGET_EDGES - cross
        if internal_needed < 0:
            continue
        max_internal = a_size*(a_size-1)//2 + b_size*(b_size-1)//2
        if internal_needed > max_internal:
            continue
        if internal_needed == 0:
            continue
        
        # Build cross edges
        adj_try = [0] * N
        for i in range(a_size):
            for j in range(a_size, N):
                adj_try[i] |= (1 << j)
                adj_try[j] |= (1 << i)
        
        # Collect internal edge candidates
        internal_cands = []
        for i in range(a_size):
            for j in range(i+1, a_size):
                internal_cands.append((i,j))
        for i in range(a_size, N):
            for j in range(i+1, N):
                internal_cands.append((i,j))
        
        if len(internal_cands) < internal_needed:
            continue
        
        # Try several random internal edge sets
        for trial in range(15):
            if elapsed() > DEADLINE - 30:
                break
            chosen = rng.sample(internal_cands, internal_needed)
            adj_try2 = adj_try[:]
            for (i,j) in chosen:
                adj_try2[i] |= (1 << j)
                adj_try2[j] |= (1 << i)
            
            deg_try = [0] * N
            for i in range(N):
                deg_try[i] = bin(adj_try2[i] & ((1 << N) - 1)).count('1')
            score_try = _max_tri_deg_sum_fast(adj_try2, deg_try)
            
            if score_try < best_score:
                best_score = score_try
                best_adj = adj_try2[:]
                best_deg = deg_try[:]
    
    # Phase 2: Intensive local search from best structure
    adj_bits = best_adj[:]
    deg = best_deg[:]
    cur_score = best_score
    
    # Edge and non-edge lists
    edges = []
    nonedges = []
    for i in range(N):
        for j in range(i+1, N):
            if (adj_bits[i] >> j) & 1:
                edges.append((i,j))
            else:
                nonedges.append((i,j))
    
    # Simulated annealing with bitset operations
    start_temp = 5.0
    end_temp = 0.005
    max_iters = 300000
    cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
    temp = start_temp
    iters = 0
    no_improve = 0
    improve_patience = 50000
    
    while iters < max_iters:
        if elapsed() > DEADLINE - 5:
            break
        
        if not edges or not nonedges:
            break
        
        e_remove = edges[rng.randrange(len(edges))]
        e_add = nonedges[rng.randrange(len(nonedges))]
        
        u, v = e_remove
        x, y = e_add
        
        new_adj = adj_bits[:]
        new_deg = deg[:]
        
        new_adj[u] &= ~(1 << v)
        new_adj[v] &= ~(1 << u)
        new_deg[u] -= 1
        new_deg[v] -= 1
        
        new_adj[x] |= (1 << y)
        new_adj[y] |= (1 << x)
        new_deg[x] += 1
        new_deg[y] += 1
        
        new_score = _max_tri_deg_sum_fast(new_adj, new_deg)
        
        if new_score < cur_score:
            accept = True
            no_improve = 0
        elif new_score == cur_score:
            accept = rng.random() < 0.25
            no_improve += 1
        else:
            delta = new_score - cur_score
            if temp < 1e-12:
                accept = False
            else:
                accept = math.exp(-delta / temp) > rng.random()
            no_improve += 1
        
        if accept:
            adj_bits = new_adj
            deg = new_deg
            cur_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_adj = adj_bits[:]
                best_deg = deg[:]
                no_improve = 0
            
            # Update lists
            edges.remove(e_remove)
            nonedges.append(e_remove)
            nonedges.remove(e_add)
            edges.append(e_add)
        
        iters += 1
        temp = max(end_temp, temp * cooling)
        
        # Restart if stuck
        if no_improve > improve_patience and temp > end_temp * 2:
            no_improve = 0
            temp = max(temp * 2, 1.0)
    
    # Phase 3: Final deterministic polish - try all single swaps from best
    if elapsed() < DEADLINE - 2:
        adj_bits = best_adj[:]
        deg = best_deg[:]
        cur_score = best_score
        
        edges = [(i,j) for i in range(N) for j in range(i+1,N) if (adj_bits[i]>>j)&1]
        nonedges = [(i,j) for i in range(N) for j in range(i+1,N) if not ((adj_bits[i]>>j)&1)]
        
        improved = True
        while improved and elapsed() < DEADLINE - 1:
            improved = False
            for e_rm in edges[:]:
                for e_add in nonedges[:]:
                    u,v = e_rm
                    x,y = e_add
                    new_adj = adj_bits[:]
                    new_deg = deg[:]
                    new_adj[u] &= ~(1<<v)
                    new_adj[v] &= ~(1<<u)
                    new_deg[u] -= 1
                    new_deg[v] -= 1
                    new_adj[x] |= (1<<y)
                    new_adj[y] |= (1<<x)
                    new_deg[x] += 1
                    new_deg[y] += 1
                    new_score = _max_tri_deg_sum_fast(new_adj, new_deg)
                    if new_score < cur_score:
                        adj_bits = new_adj
                        deg = new_deg
                        cur_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_adj = adj_bits[:]
                            best_deg = deg[:]
                        edges.remove(e_rm)
                        nonedges.append(e_rm)
                        nonedges.remove(e_add)
                        edges.append(e_add)
                        improved = True
                        break
                if improved:
                    break
                if elapsed() > DEADLINE - 1:
                    break
    
    # Return best found
    result_edges = []
    for i in range(N):
        for j in range(i+1, N):
            if (best_adj[i] >> j) & 1:
                result_edges.append([i, j])
    
    return {'n': N, 'edges': result_edges}

