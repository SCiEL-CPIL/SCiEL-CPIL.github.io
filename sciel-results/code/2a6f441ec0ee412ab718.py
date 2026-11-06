import random
import time
import itertools
import numpy as np

def construct():
    """
    Constructs a graph with no K5 and no independent set of size 5.
    Uses Simulated Annealing with a fast cost function to find the largest possible graph.
    """
    start_time = time.time()
    # Safety margin: Evaluator kills at 300s. We should aim to finish by 290s to be safe.
    deadline = start_time + 285.0
    
    # We know the Paley graph on 37 vertices is a valid solution.
    # We will try to find a larger one (42, 41, ...) using SA.
    # If we find one, we return it. Otherwise, we return the Paley graph.
    
    # Order of attempts: 42, 41, 40, 39, 38
    # 42 is the target (Exoo).
    # If 42 fails, try 41? No, if we can't find 42, 41 is likely harder or similar difficulty but less reward.
    # Actually, finding 42 is the main goal. 
    # Let's allocate time to 42. If it fails, we can try 41 with remaining time?
    # Or just stick to 42? 
    # The score is n. 37 is the baseline. 38, 39, 40, 41, 42 are all improvements.
    # It is better to secure a higher n than to gamble all time on 42 and fail.
    # Strategy:
    # 1. Try 42 for a large chunk of time (e.g., 200s).
    # 2. If success, return.
    # 3. If fail, try 41 for remaining time (e.g., 50s).
    # 4. If success, return.
    # 5. If fail, try 40 for remaining time.
    # ...
    # 6. Fallback to 37.
    
    # Let's refine the time budget.
    # Total budget ~285s.
    # Attempts: 42, 41, 40, 39, 38.
    # Let's give 42 150s. 41 60s. 40 30s. 39 20s. 38 15s. Total 275s.
    
    attempts = [42, 41, 40, 39, 38]
    time_budgets = [150.0, 60.0, 30.0, 20.0, 15.0]
    
    for n, t_limit in zip(attempts, time_budgets):
        if time.time() >= deadline:
            break
        
        # Set a local deadline for this attempt
        local_deadline = time.time() + min(t_limit, deadline - time.time())
        
        result = find_graph(n, local_deadline)
        if result is not None:
            return result
            
    # Fallback to the known valid Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges

def find_graph(n, deadline):
    """
    Attempts to find a (5,5)-free graph on n vertices using Simulated Annealing.
    Returns (n, edges) if successful, else None.
    """
    # We use a NumPy array for the adjacency matrix for fast vectorized operations
    # But for bitmasks, standard lists or integers might be better.
    # Let's use a list of integers (bitmasks) for the adjacency list.
    # adj[i] is an integer where bit j is set if (i,j) is an edge.
    
    adj = [0] * n
    # Initialize random graph
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    def get_cost(adj):
        """
        Calculate the number of K5 and independent sets of size 5.
        """
        count_k5 = 0
        count_ind = 0
        # Iterate all 5-combinations
        # This is slow if done naively. 
        # However, we only call this for initial check and final verification.
        # For the SA loop, we use delta.
        for combo in itertools.combinations(range(n), 5):
            is_clique = True
            is_ind = True
            for i in range(5):
                for j in range(i+1, 5):
                    u = combo[i]
                    v = combo[j]
                    if adj[u] & (1 << v):
                        is_ind = False
                    else:
                        is_clique = False
            if is_clique:
                count_k5 += 1
            if is_ind:
                count_ind += 1
        return count_k5 + count_ind

    current_cost = get_cost(adj)
    if current_cost == 0:
        return extract_edges(n, adj)
    
    # Simulated Annealing Parameters
    # T_start should be high enough to allow moves.
    # Cost values are integers. Delta can be small.
    T = 1.0
    cooling = 0.9995 # Slow cooling
    min_T = 0.01
    
    # Precompute list of all edges (i, j) to pick from
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline - 1.0:
        if current_cost == 0:
            return extract_edges(n, adj)
            
        # Pick random edge
        u, v = random.choice(all_edges)
        
        # Calculate delta cost for flipping (u, v)
        # Delta = (New K5s + New Inds) - (Old K5s + Old Inds)
        # Only 5-sets containing u and v are affected.
        # Set S = {u, v, a, b, c}
        
        # We need to check the status of S before and after.
        # To make this fast, we can iterate over all a, b, c.
        # But we can optimize the check.
        
        # Let's iterate a, b, c from the remaining vertices.
        # n=42 -> 40 remaining vertices. C(40,3) = 9880.
        # 9880 iterations per step.
        # Inside: check 9 edges (u-a, u-b, u-c, v-a, v-b, v-c, a-b, a-c, b-c).
        # This is ~100k ops per step.
        # 300s / 0.1ms per step = 300,000 steps.
        # 300,000 * 9880 * 10 ops = 3 * 10^10 ops. Too slow in Python.
        
        # We need a faster delta calculation.
        # Key insight:
        # A 5-set S={u,v,a,b,c} becomes a K5 if:
        #   - It was "almost" a K5, differing only by the status of (u,v).
        #   - Specifically, if we ADD (u,v) (val 0->1):
        #     It becomes a K5 if all other edges in S were 1.
        #     i.e., a,b,c form a triangle in G, and u,v are connected to all a,b,c.
        #   - If we REMOVE (u,v) (val 1->0):
        #     It becomes an Independent set if all other edges in S were 0.
        #     i.e., a,b,c form a triangle in G_bar (independent in G), and u,v are not connected to any of a,b,c.
        
        # So, Delta = 
        #   If adding edge (u,v):
        #     + (# of {a,b,c} such that {a,b,c} is triangle in G, and u,v connected to all) 
        #     - (# of {a,b,c} such that {a,b,c} is triangle in G_bar, and u,v not connected to any)
        #     Wait, does adding an edge destroy independent sets?
        #     Yes, if S was an independent set, adding (u,v) destroys it.
        #     S is independent if all edges in S are 0.
        #     If we add (u,v), S is no longer independent.
        #     So we lose the independent sets that contained u,v and had all other edges 0.
        #     Condition: a,b,c independent in G (triangle in G_bar) AND u,v not connected to a,b,c.
        
        #   If removing edge (u,v):
        #     + (# of {a,b,c} such that {a,b,c} is triangle in G_bar, and u,v not connected to any)
        #     - (# of {a,b,c} such that {a,b,c} is triangle in G, and u,v connected to all)
        
        # So we need to count:
        # C_G(u, v) = number of triangles {a,b,c} in G such that u,v are connected to all a,b,c.
        # C_Gbar(u, v) = number of triangles {a,b,c} in G_bar such that u,v are NOT connected to all a,b,c (i.e., independent in G).
        
        # Let's calculate these counts efficiently.
        # Candidates for a,b,c must be in the common neighborhood (for G) or common non-neighborhood (for G_bar) of u and v.
        
        val = (adj[u] >> v) & 1
        
        if val == 0:
            # Adding edge
            # Count C_G(u,v): a,b,c must be neighbors of u AND neighbors of v.
            # And a,b,c must form a triangle in G.
            # Common neighbors: N(u) & N(v)
            common_neigh = adj[u] & adj[v] & ~((1 << u) | (1 << v))
            
            # Count triangles in the induced subgraph on common_neigh
            c_g = count_triangles_in_bitmask(common_neigh, adj)
            
            # Count C_Gbar(u,v): a,b,c must be non-neighbors of u AND non-neighbors of v.
            # Non-neighbors: ~(N(u) | {u}) & ~(N(v) | {v}) & mask_all
            # Let's define non_neigh(u) = ~(adj[u] | (1<<u)) & mask_all
            mask_all = (1 << n) - 1
            non_neigh_u = (~adj[u]) & mask_all
            non_neigh_v = (~adj[v]) & mask_all
            common_non_neigh = non_neigh_u & non_neigh_v & ~((1 << u) | (1 << v))
            
            # Count triangles in G_bar induced by common_non_neigh.
            # Triangle in G_bar means triangle in G with edges removed?
            # No, triangle in G_bar is an independent set of size 3 in G.
            # So we count triangles in the complement graph.
            c_gbar = count_triangles_in_bitmask_complement(common_non_neigh, adj)
            
            delta = c_g - c_gbar
        else:
            # Removing edge
            # We are removing (u,v).
            # This creates independent sets (if others are non-neighbors) and destroys cliques.
            # Wait, if we REMOVE (u,v):
            # - It might create an independent set if the other 3 were independent and u,v were non-connected to them.
            #   This is the same as C_Gbar(u,v).
            # - It destroys a clique if the other 3 were a clique and u,v were connected to them.
            #   This is the same as C_G(u,v).
            
            common_neigh = adj[u] & adj[v] & ~((1 << u) | (1 << v))
            c_g = count_triangles_in_bitmask(common_neigh, adj)
            
            non_neigh_u = (~adj[u]) & mask_all
            non_neigh_v = (~adj[v]) & mask_all
            common_non_neigh = non_neigh_u & non_neigh_v & ~((1 << u) | (1 << v))
            c_gbar = count_triangles_in_bitmask_complement(common_non_neigh, adj)
            
            delta = c_gbar - c_g

        # Acceptance
        if delta < 0:
            # Accept
            if val == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            current_cost += delta
        else:
            # Accept with probability exp(-delta / T)
            if random.random() < np.exp(-delta / T):
                if val == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                current_cost += delta
        
        T *= cooling
        if T < min_T:
            T = min_T

    # Final check
    if get_cost(adj) == 0:
        return extract_edges(n, adj)
        
    return None

def count_triangles_in_bitmask(mask, adj):
    """
    Count the number of triangles in the subgraph induced by the vertices in 'mask'.
    mask is an integer where bit i is set if vertex i is in the set.
    adj is the list of adjacency bitmasks for the full graph.
    """
    # Extract vertices
    vertices = []
    m = mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        vertices.append(idx)
        m ^= lsb
    
    k = len(vertices)
    if k < 3:
        return 0
        
    # Map vertex index to position in local list?
    # Or just use the global indices and check adj directly.
    # Triangles (i, j, l) in vertices.
    # Condition: (i,j) in adj, (j,l) in adj, (i,l) in adj.
    
    count = 0
    # To avoid double counting, we can iterate i < j < l.
    # Or use the standard O(k^3) loop.
    for i in range(k):
        u = vertices[i]
        for j in range(i+1, k):
            v = vertices[j]
            # Check if u and v are connected
            if (adj[u] >> v) & 1:
                # Check common neighbors of u and v within the set
                # Common neighbors of u and v are (adj[u] & adj[v] & mask)
                # We need to count how many of these are in vertices.
                # Since we are iterating i<j, we only care about l > j?
                # No, just count all common neighbors in the set that are distinct from u,v.
                common = adj[u] & adj[v] & mask
                # We need to ensure we only count vertices in 'vertices'.
                # 'mask' already represents 'vertices'.
                # So common contains all vertices in 'vertices' that are neighbors to both u and v.
                # We need to remove u and v themselves if they are in common?
                # u is not a neighbor of u. v is not a neighbor of v.
                # But u might be a neighbor of v? Yes.
                # adj[u] has bit v set. adj[v] has bit u set.
                # So common has bits u and v set?
                # adj[u] & adj[v]:
                # Bit u in adj[u] is 0. So bit u in common is 0.
                # Bit v in adj[v] is 0. So bit v in common is 0.
                # So common only contains other vertices.
                
                # We just need the count of bits in common.
                # But wait, we are counting triangles (u,v,l).
                # For a fixed pair (u,v) that is an edge, the number of triangles containing this edge
                # is the number of common neighbors l.
                # So we can just sum the bit counts.
                
                # However, we must ensure l is in 'vertices'.
                # 'mask' is the indicator for 'vertices'.
                # So common = adj[u] & adj[v] & mask gives exactly the l's.
                
                count += common.bit_count()
                
    # This counts each triangle 3 times (once for each edge).
    # Triangle (a,b,c) has edges (a,b), (b,c), (a,c).
    # When i=a, j=b, we count c.
    # When i=b, j=c, we count a.
    # When i=a, j=c, we count b.
    # So divide by 3.
    
    return count // 3

def count_triangles_in_bitmask_complement(mask, adj):
    """
    Count the number of triangles in the COMPLEMENT subgraph induced by 'mask'.
    i.e., Independent sets of size 3 in the original graph induced by 'mask'.
    """
    vertices = []
    m = mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        vertices.append(idx)
        m ^= lsb
        
    k = len(vertices)
    if k < 3:
        return 0
        
    count = 0
    for i in range(k):
        u = vertices[i]
        for j in range(i+1, k):
            v = vertices[j]
            # In complement, u and v are connected if they are NOT connected in G.
            if not ((adj[u] >> v) & 1):
                # Common non-neighbors of u and v within the set.
                # Non-neighbors of u: ~adj[u]
                # Non-neighbors of v: ~adj[v]
                # We need vertices in 'mask' that are non-neighbors to both.
                # common_non = (~adj[u]) & (~adj[v]) & mask
                # But ~adj[u] includes u itself? No, adj[u] doesn't have u.
                # However, ~adj[u] in Python is infinite bits negative.
                # We must mask it.
                # Let's use the fact that we only care about bits in 'mask'.
                
                # Bits in mask that are NOT in adj[u] and NOT in adj[v].
                # mask & ~adj[u] & ~adj[v]
                
                common_non = mask & (~adj[u]) & (~adj[v])
                
                # As before, u and v are not in common_non because:
                # u is in mask. Is u in ~adj[u]? Yes (adj[u] doesn't have u).
                # Is u in ~adj[v]? If v is connected to u, then u is in adj[v], so u NOT in ~adj[v].
                # Since (u,v) is NOT an edge in G, u is NOT in adj[v]. So u IS in ~adj[v].
                # So u is in common_non?
                # Let's check:
                # We want l such that (u,l) not edge, (v,l) not edge.
                # If l=u: (u,u) is not an edge? Self loops are 0. So (u,u) is not an edge.
                # But a triangle cannot have repeated vertices.
                # The standard triangle counting logic assumed distinct vertices.
                # The bitmask intersection might include u or v if they satisfy the "non-neighbor" condition.
                
                # For G: adj[u] does not have bit u. So u is not in adj[u] & adj[v].
                # For G_bar: ~adj[u] HAS bit u.
                # So we need to explicitly remove u and v from common_non.
                
                common_non &= ~(1 << u)
                common_non &= ~(1 << v)
                
                count += common_non.bit_count()
                
    return count // 3

def extract_edges(n, adj):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                edges.append((i, j))
    return n, edges

# Test locally?
# if __name__ == "__main__":
#     res = construct()
#     print(res[0], len(res[1]))
