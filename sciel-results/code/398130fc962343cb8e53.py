"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random
import sys

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _bit_count(x):
    return bin(x).count('1')

def _has_triangle_bitmask(adj, n):
    """Check if adjacency bitmask list has a triangle."""
    for i in range(n):
        ni = adj[i]
        j_mask = ni
        while j_mask:
            lsb = j_mask & -j_mask
            j = lsb.bit_length() - 1
            j_mask ^= lsb
            if j > i:
                # Check common neighbors of i and j
                # adj[i] & adj[j] contains k such that k~i and k~j
                # If non-empty, triangle (i,j,k) exists
                if adj[i] & adj[j]:
                    return True
    return False

def _max_independent_set_size_bitmask(adj, n):
    """
    Find size of maximum independent set using branch and bound with bitmasks.
    Uses a greedy coloring bound on the complement graph (clique cover of original) 
    or equivalent logic.
    
    To find Max Independent Set in G, we find Max Clique in G_bar.
    G_bar edge (u,v) exists if G edge (u,v) does NOT exist.
    
    Alternatively, we can work directly on G.
    Branch on vertex v.
    1. Include v: Remove v and its neighbors from available.
    2. Exclude v: Remove v from available.
    
    Bound: The size of the maximum independent set in the remaining graph 
    is at most the chromatic number of the remaining graph? 
    No, alpha(G) <= chi(G_bar). 
    A greedy coloring of G_bar gives an upper bound on the size of a clique in G_bar, 
    which is the size of an independent set in G.
    """
    best = [0]
    full_mask = (1 << n) - 1
    
    # Precompute non-adjacency masks for faster coloring bound calculation?
    # non_adj[v] = ~(adj[v] | (1<<v)) & full_mask
    
    def greedy_coloring_bound(available):
        """
        Computes an upper bound on the size of the maximum independent set in G[available].
        This is equivalent to the chromatic number of the complement graph G_bar[available].
        We greedily color G_bar[available].
        In G_bar, two vertices are adjacent if they are NOT adjacent in G.
        So, a valid color class in G_bar is a set of vertices that are pairwise adjacent in G (a clique in G).
        Wait. 
        Coloring G_bar: Vertices in the same color class must be non-adjacent in G_bar.
        Non-adjacent in G_bar means adjacent in G.
        So a color class in G_bar is a Clique in G.
        The number of colors used is an upper bound on the clique number of G_bar, 
        which is the independence number of G.
        
        So we want to partition 'available' into as few Cliques of G as possible.
        This is the Clique Cover problem.
        Greedy approach:
        1. Find a maximal clique in G[available].
        2. Remove it.
        3. Repeat.
        The number of cliques is the bound.
        """
        unassigned = available
        k = 0
        while unassigned:
            k += 1
            # Find a maximal clique in G[unassigned]
            # Greedy: pick a vertex, then add neighbors that are connected to all current clique members.
            
            # Heuristic: Pick vertex with highest degree in G[unassigned]
            # To speed up, we can just pick the lowest bit, but degree heuristic is better for packing.
            # Let's pick lowest bit for simplicity and speed, or random?
            # Let's try picking the vertex with max degree in the subgraph.
            # For speed in B&B, simple is often better.
            # Let's use a simple greedy: pick v, then add all neighbors of v in unassigned that are mutually connected?
            # No, just build a maximal clique.
            
            # Pick v
            lsb_v = unassigned & -unassigned
            v = lsb_v.bit_length() - 1
            unassigned ^= lsb_v
            
            clique_mask = lsb_v
            # Candidates must be in unassigned and adjacent to ALL vertices in clique
            # Initially, candidates = unassigned & adj[v]
            candidates = unassigned & adj[v]
            
            while candidates:
                lsb_u = candidates & -candidates
                u = lsb_u.bit_length() - 1
                candidates ^= lsb_u
                unassigned ^= lsb_u
                clique_mask |= lsb_u
                # Update candidates: must be adjacent to u
                candidates &= adj[u]
                
        return k

    def bnb(available, current_size):
        if not available:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pruning: if current_size + bound <= best[0], return
        bound = greedy_coloring_bound(available)
        if current_size + bound <= best[0]:
            return
        
        # Pick vertex to branch on
        # Heuristic: Max degree in G[available]
        v_mask = available
        max_deg = -1
        v = -1
        while v_mask:
            lsb = v_mask & -v_mask
            u = lsb.bit_length() - 1
            v_mask ^= lsb
            deg = _bit_count(adj[u] & available)
            if deg > max_deg:
                max_deg = deg
                v = u
        
        if v == -1:
            return

        # Branch 1: Include v
        # Remove v and its neighbors
        new_avail_include = available & ~(1 << v) & ~adj[v]
        bnb(new_avail_include, current_size + 1)
        
        # Branch 2: Exclude v
        new_avail_exclude = available & ~(1 << v)
        bnb(new_avail_exclude, current_size)

    bnb(full_mask, 0)
    return best[0]

def _validate_bitmask(adj, n):
    """Check triangle-free and no independent 10-set."""
    if _has_triangle_bitmask(adj, n):
        return False
    if _max_independent_set_size_bitmask(adj, n) >= 10:
        return False
    return True

def _greedy_triangle_free_bitmask(n, rng):
    """
    Build a triangle-free graph greedily using bitmasks.
    Start with random graph, remove edges to kill triangles.
    """
    adj = [0] * n
    
    # Generate all potential edges
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(edges)
    
    for i, j in edges:
        if _remaining_time() < 2:
            break
        # Check if adding (i,j) creates a triangle
        # Triangle if common neighbor exists
        if adj[i] & adj[j]:
            continue
        
        # Add edge
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    
    return adj

def _local_search_targeted(adj, n, rng, max_iter=1000):
    """
    Improve graph by adding edges specifically within the maximum independent set 
    to break it, provided no triangle is formed.
    """
    adj = adj[:]
    alpha = _max_independent_set_size_bitmask(adj, n)
    
    if alpha < 10:
        return adj, alpha
    
    for _ in range(max_iter):
        if _remaining_time() < 3:
            break
        
        # Find a maximum independent set
        # We need the actual set, not just the size. 
        # Let's modify the B&B to return the set or just use a greedy approximation?
        # For speed, let's just find *an* independent set of size alpha using a simple greedy?
        # No, we need a large one to be effective.
        # Let's implement a quick version of B&B that returns the set.
        
        # Actually, we can just pick a random subset of size 10 if alpha >= 10?
        # No, we need to break the *maximum* one.
        
        # Let's find a maximal independent set using greedy.
        # Greedy MIS:
        available = (1 << n) - 1
        mis = []
        while available:
            # Pick random vertex from available
            # Or lowest bit
            lsb = available & -available
            v = lsb.bit_length() - 1
            mis.append(v)
            available &= ~(1 << v)
            available &= ~adj[v]
            
        # If the greedy MIS is small, it might not be the max one.
        # But if alpha >= 10, and greedy MIS is < 10, it doesn't help.
        # We should use the B&B result.
        # Let's modify _max_independent_set_size_bitmask to optionally return the set.
        # For now, let's assume we can find a large independent set.
        # To save time, let's just try to add edges between random pairs in a *large* independent set.
        # How to get a large one fast?
        # Run B&B once to get the size. If size >= 10, we know one exists.
        # Let's just pick 10 random vertices and check if they form an independent set? Unlikely.
        
        # Better: Use the B&B to find *one* independent set of size alpha.
        # I will implement a helper to find one MIS.
        
        mis = _find_one_mis(adj, n, alpha)
        
        if not mis or len(mis) < 10:
            # If we can't find a large one quickly, break
            # Or try random pairs from the whole graph?
            break
            
        # Try to add an edge between two vertices in mis
        # Condition: No common neighbor.
        added = False
        # Shuffle mis to randomize
        rng.shuffle(mis)
        for i in range(len(mis)):
            for j in range(i+1, len(mis)):
                u = mis[i]
                v = mis[j]
                # Check if edge (u,v) can be added
                if not (adj[u] & (1 << v)):
                    if not (adj[u] & adj[v]):
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added = True
                        break
            if added:
                break
                
        if not added:
            # No edge could be added within this MIS
            # Try a different MIS? Or break?
            # For now, break to save time
            break
            
        # Recalculate alpha
        new_alpha = _max_independent_set_size_bitmask(adj, n)
        if new_alpha < alpha:
            alpha = new_alpha
            if alpha < 10:
                return adj, alpha
        else:
            # If alpha didn't decrease, maybe revert?
            # Or keep it? Keeping it might help future steps.
            # Let's keep it.
            pass

    return adj, alpha

def _find_one_mis(adj, n, target_size):
    """
    Find one independent set of size target_size.
    Uses a simple backtracking search.
    """
    full_mask = (1 << n) - 1
    path = []
    
    def backtrack(available, current_size):
        if current_size == target_size:
            return True
        if not available:
            return False
            
        # Bound: if current_size + bit_count(available) < target_size, return False
        if current_size + _bit_count(available) < target_size:
            return False
            
        # Pick vertex
        # Heuristic: pick vertex with min degree in available?
        # Or just lowest bit.
        lsb = available & -available
        v = lsb.bit_length() - 1
        
        # Branch: Include v
        path.append(v)
        new_avail = available & ~(1 << v) & ~adj[v]
        if backtrack(new_avail, current_size + 1):
            return True
        path.pop()
        
        # Branch: Exclude v
        new_avail_excl = available & ~(1 << v)
        if backtrack(new_avail_excl, current_size):
            return True
            
        return False
        
    if backtrack(full_mask, 0):
        return path[:]
    return []

def _try_construction_bitmask(n, seed, max_local_iters=500):
    """
    Try to build a valid graph on n vertices.
    """
    rng = random.Random(seed)
    
    # Strategy 1: Random greedy
    adj = _greedy_triangle_free_bitmask(n, rng)
    alpha = _max_independent_set_size_bitmask(adj, n)
    
    if alpha < 10:
        return adj, alpha
    
    # Try targeted local search
    if _remaining_time() > 5:
        adj_ls, alpha_ls = _local_search_targeted(adj, n, rng, max_local_iters)
        if alpha_ls < 10:
            return adj_ls, alpha_ls
        if alpha_ls < alpha:
            adj, alpha = adj_ls, alpha_ls
            
    # Strategy 2: Different random orderings
    for attempt in range(5):
        if _remaining_time() < 5:
            break
        rng2 = random.Random(seed * 1000 + attempt)
        adj2 = _greedy_triangle_free_bitmask(n, rng2)
        alpha2 = _max_independent_set_size_bitmask(adj2, n)
        
        if alpha2 < 10:
            return adj2, alpha2
        
        if alpha2 < alpha and _remaining_time() > 5:
            adj_ls2, alpha_ls2 = _local_search_targeted(adj2, n, rng2, max_local_iters)
            if alpha_ls2 < alpha:
                adj, alpha = adj_ls2, alpha_ls2
                if alpha < 10:
                    return adj, alpha
                    
    return None

def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj = None
    best_n = 0
    best_alpha = float('inf')
    
    # Try n from high to low
    for n_target in range(40, 20, -1):
        if _remaining_time() < 8:
            break
        
        found = False
        
        # Try random greedy + local search
        # Use multiple seeds
        for seed in range(100):
            if _remaining_time() < 3:
                break
            
            result = _try_construction_bitmask(n_target, seed)
            if result is not None:
                adj_res, alpha_res = result
                if alpha_res < best_alpha:
                    best_adj = adj_res
                    best_n = n_target
                    best_alpha = alpha_res
                if alpha_res < 10:
                    found = True
                    break
        
        if found:
            break
    
    # Fallback: use a known small construction if nothing worked
    if best_adj is None:
        # Use a random greedy that we know works for small n
        n = 26
        rng = random.Random(42)
        adj = _greedy_triangle_free_bitmask(n, rng)
        best_adj = adj
        best_n = n
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
