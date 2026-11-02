"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def _generate_candidates(weights):
    """Generate all integer vectors with the specified weights."""
    candidates = []
    for w in weights:
        if w > D:
            continue
        # Choose w indices out of D
        for indices in itertools.combinations(range(D), w):
            # Choose signs for these indices
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def _tomita_mis(adj, n, time_limit, deadline):
    """
    Tomita's branch-and-bound algorithm for Maximum Independent Set.
    
    Parameters:
    adj: list of lists, adj[i] is the set of neighbors (conflicts) of i.
         We assume i is not in adj[i].
    n: number of vertices.
    time_limit: absolute time deadline.
    deadline: same as time_limit, kept for clarity.
    
    Returns:
    list of vertices in the MIS.
    """
    # Precompute neighbors as sets for fast intersection? 
    # Actually, Tomita usually works with bitsets or sorted lists.
    # Given n ~ 1000-2000, sets are okay but bitsets are faster.
    # Let's use Python integers as bitsets for speed.
    
    # Convert adj to bitsets
    adj_bits = [0] * n
    for i in range(n):
        bits = 0
        for j in adj[i]:
            bits |= (1 << j)
        adj_bits[i] = bits

    # Colors for the greedy coloring step
    # We need a function to color a candidate set P (represented as a bitset)
    # such that vertices with the same color form an independent set.
    # We want to maximize the number of colors to get a tight bound.
    # Standard greedy coloring: 
    # While P is not empty:
    #   Find a maximal independent set in P (greedy)
    #   Assign current color to these vertices
    #   Remove them from P
    #   Increment color count
    
    # To make it fast, we process vertices in a specific order (e.g., by degree).
    # However, the order of vertices in the bitset matters for the greedy step.
    # Let's stick to a simple greedy coloring on the subset P.
    
    def color_set(P):
        """
        Greedily color the vertices in P (bitset).
        Returns a list of colors for each vertex in P (indexed by original vertex id? 
        No, we need to map back to the set of vertices in P).
        Actually, Tomita's algorithm typically modifies the set P and assigns colors 
        to vertices in P. We need to return the maximum color assigned to any vertex in P,
        and potentially the specific color of each vertex to guide the branching order.
        
        Let's return:
        max_color: int
        colors: dict or array where colors[v] is the color of v for v in P.
        """
        if P == 0:
            return 0, {}
        
        colors = {}
        c = 0
        remaining = P
        
        # We need to pick vertices from 'remaining' to form independent sets.
        # To get a good coloring, we should pick vertices with high degree in the induced subgraph?
        # Or just iterate?
        # Standard greedy coloring for MIS bound:
        # We want to partition P into as few independent sets as possible? 
        # No, we want to assign colors such that adjacent vertices have different colors.
        # The number of colors is an upper bound on the size of the MIS in P.
        # So we want to MINIMIZE the number of colors? 
        # Wait. The bound is that any independent set can have at most one vertex of each color.
        # So if we use K colors, the MIS size <= K.
        # To get a TIGHT bound (small K), we should use a coloring that uses FEW colors.
        # Greedy coloring that minimizes colors: 
        # While P not empty:
        #   Find a maximal independent set S in P.
        #   Assign color c to all v in S.
        #   P = P \ S.
        #   c++.
        
        # How to find a maximal independent set in P quickly?
        # Greedy: pick a vertex v in P, add to S, remove v and its neighbors from P.
        # To make S large (and thus reduce colors), we should pick vertices with low degree in P?
        # Or high degree? 
        # If we pick a vertex with low degree, we remove few neighbors, leaving more for future sets?
        # Actually, to minimize colors, we want each color class (independent set) to be as large as possible.
        # So we want to find a large independent set in P.
        # Greedy heuristic for Max Independent Set: pick vertex with lowest degree in current graph.
        
        while remaining:
            c += 1
            # Find a vertex in remaining with minimum degree within remaining
            # This is O(n^2) if done naively. 
            # For n~2000, this might be slow if done repeatedly.
            # Alternative: Just iterate through vertices in remaining and pick the first one? 
            # That gives a valid coloring, but maybe loose bound.
            # Let's try picking the vertex with min degree in remaining.
            
            min_deg = float('inf')
            min_v = -1
            
            # Iterate through bits in remaining
            # Extracting bits is slow.
            # Let's just pick the first available vertex? It's faster.
            # If time permits, we can optimize.
            # For now, simple greedy: pick first vertex.
            
            # Find first set bit
            # (remaining & -remaining).bit_length() - 1
            if remaining == 0:
                break
                
            # Get least significant set bit
            lsb = remaining & -remaining
            v = lsb.bit_length() - 1
            
            # Assign color c to v
            colors[v] = c
            
            # Remove v and its neighbors from remaining
            # Neighbors of v in remaining: remaining & adj_bits[v]
            # Also remove v itself
            remaining = remaining & ~(adj_bits[v] | (1 << v))
            
        return c, colors

    best_set = []
    
    def expand(P, X):
        """
        Recursive branch and bound.
        P: bitset of candidate vertices.
        X: bitset of vertices already selected in the current path.
        """
        nonlocal best_set
        
        if time.time() > deadline:
            return
        
        if P == 0:
            if X.bit_count() > len(best_set):
                # Extract indices from X
                curr_set = []
                temp = X
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    curr_set.append(v)
                    temp = temp & (temp - 1)
                best_set = curr_set
            return
        
        # Get upper bound from coloring
        # We need to color P.
        # To speed up, we can cache colorings? Hard to do with bitsets.
        # Let's just compute it.
        
        # Optimization: If the current best size + |P| <= len(best_set), prune.
        if X.bit_count() + P.bit_count() <= len(best_set):
            return
            
        # Coloring
        # The coloring function above is O(n^2) worst case per call.
        # This might be too slow.
        # Let's use a simpler bound: P.bit_count()
        # But that's weak.
        
        # Let's try to implement a faster coloring or just use the count if coloring is too slow.
        # Actually, for n=2000, if the branching factor is low, it might be okay.
        
        # Let's refine the coloring to be faster.
        # We'll just use the number of vertices as a bound initially to ensure correctness,
        # then switch to coloring if we have time?
        # No, the directive says "Tomita (BB-MCS) coloring bound".
        
        # Let's try to make coloring efficient.
        # We can precompute the order of vertices?
        
        # Let's stick to the simple greedy coloring but optimize the inner loop.
        # Instead of finding min degree, we just pick vertices in a fixed order (e.g., index order) 
        # and remove neighbors. This is a valid coloring.
        
        def fast_color(P):
            if P == 0:
                return 0
            c = 0
            rem = P
            while rem:
                c += 1
                # Pick a vertex. Let's pick the one with smallest index for determinism/speed.
                lsb = rem & -rem
                v = lsb.bit_length() - 1
                # Remove v and neighbors
                rem = rem & ~(adj_bits[v] | (1 << v))
            return c
            
        bound = fast_color(P)
        
        if X.bit_count() + bound <= len(best_set):
            return
            
        # Branching:
        # We should branch on vertices in the order of decreasing color? 
        # Or just iterate through vertices in P?
        # Tomita suggests ordering vertices by color.
        # Since we didn't store the specific colors, let's just iterate through P.
        # To improve, we can iterate in reverse order of index or something.
        
        # Let's iterate through vertices in P.
        # We can extract them.
        temp = P
        # To allow backtracking and pruning, we process one vertex at a time.
        # We pick a vertex v from P.
        # Branch 1: Include v. New P = P \ (N(v) U {v}). New X = X U {v}.
        # Branch 2: Exclude v. New P = P \ {v}. New X = X.
        
        # To prune effectively, we should pick v such that it has high degree?
        # Let's pick the vertex with the highest degree in P.
        # Finding highest degree in P is expensive.
        # Let's just pick the first one (lowest index).
        
        lsb = P & -P
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        # New P must not contain neighbors of v or v itself.
        new_P = P & ~(adj_bits[v] | (1 << v))
        new_X = X | (1 << v)
        
        # Check bound before recursing? 
        # The recursive call will do it.
        
        expand(new_P, new_X)
        
        # Branch 2: Exclude v
        # Remove v from P
        new_P2 = P & ~(1 << v)
        expand(new_P2, X)

    # Initial call
    # P is all vertices
    initial_P = (1 << n) - 1
    expand(initial_P, 0)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Generate candidates with weight 2 and 3?
    # Weight 2: Norm^2 = 2. Pairs of weight 2: dist^2 = 4 - 2*dot.
    # If disjoint, dot=0, dist^2=4 >= 2. OK.
    # If share 1 index, dot=+-1. dist^2 = 4 - 2(+-1) = 2 or 6.
    # If dot=1, dist^2=2 >= max(2,2)=2. OK.
    # If dot=-1, dist^2=6 >= 2. OK.
    # So all weight 2 vectors are compatible with each other?
    # Wait. If v=(1,1,0..), w=(1,-1,0..). dot=0. dist^2=4.
    # If v=(1,1,0..), w=(1,1,0..). Same vector? We assume distinct.
    # If v=(1,1,0..), w=(-1,1,0..). dot=0.
    # If v=(1,1,0..), w=(1,0,1..). dot=1. dist^2 = 2+2-2=2. OK.
    # It seems all weight 2 vectors are mutually compatible.
    # Number of weight 2 vectors: C(11,2) * 4 = 55 * 4 = 220.
    
    # Weight 3: Norm^2 = 3.
    # Weight 2 vs Weight 3:
    # v (w2), w (w3).
    # dist^2 = 2 + 3 - 2*dot = 5 - 2*dot.
    # Need dist^2 >= 3.
    # 5 - 2*dot >= 3 => 2*dot <= 2 => dot <= 1.
    # Max dot between w2 and w3 is 2 (if w2 is subset of w3 with same signs).
    # If dot=2, dist^2 = 5-4=1 < 3. CONFLICT.
    # So weight 2 and weight 3 vectors conflict if the weight 2 vector is a sub-vector of the weight 3 vector with matching signs.
    
    # Weight 3 vs Weight 3:
    # dist^2 = 3 + 3 - 2*dot = 6 - 2*dot.
    # Need dist^2 >= 3.
    # 6 - 2*dot >= 3 => 2*dot <= 3 => dot <= 1.5 => dot <= 1.
    # Max dot is 3 (same vector).
    # If dot=3 (same), dist=0. Conflict.
    # If dot=2 (share 2 indices, same signs), dist^2 = 6-4=2 < 3. CONFLICT.
    # If dot=1 (share 1 index same, or 2 indices opposite? No, max overlap 2 gives dot 2 or 0 or -2. Overlap 1 gives 1 or -1. Overlap 0 gives 0).
    # So conflict if dot >= 2.
    
    # So we have a conflict graph.
    # Let's generate candidates of weight 2 and 3.
    candidates = _generate_candidates([2, 3])
    n_cand = len(candidates)
    
    # Build conflict graph
    # adj[i] is a set of j such that i and j conflict.
    # Conflict condition: dist^2 < max(norm_i^2, norm_j^2)
    # norms_sq = np.sum(candidates * candidates, axis=1)
    # We can precompute dot products.
    
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Compute Gram matrix
    # G = candidates @ candidates.T
    # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2*G[i,j]
    # Conflict if dist_sq[i,j] < max(norms_sq[i], norms_sq[j])
    
    # n_cand is ~ 220 + C(11,3)*8 = 220 + 165*8 = 220 + 1320 = 1540.
    # Matrix is 1540x1540. 2.4M elements. Fits in memory.
    
    G = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * G
    max_norms = np.maximum(norms_sq[:, None], norms_sq[None, :])
    
    # Conflict mask
    # Note: dist_sq[i,i] = 0. max_norms[i,i] = norm_i. 0 < norm_i. So self-conflict?
    # We don't include self in adjacency.
    conflict_mask = dist_sq < max_norms
    
    # Build adjacency list (sets)
    adj = [set() for _ in range(n_cand)]
    # Iterate over upper triangle
    # To speed up, use numpy to find indices
    i_idx, j_idx = np.where(conflict_mask)
    for i, j in zip(i_idx, j_idx):
        if i != j:
            adj[i].add(j)
            adj[j].add(i)
            
    # Convert to list of lists for Tomita? 
    # Tomita function expects adj[i] to be iterable. Sets are fine.
    
    # Run MIS
    # Time limit: 140s.
    # If n_cand is large, this might take a while.
    # 1540 nodes.
    
    # Let's add a timeout check inside the Tomita function if it runs too long.
    # The function `expand` checks time.
    
    try:
        mis_indices = _tomita_mis(adj, n_cand, deadline, deadline)
    except Exception:
        # Fallback to greedy if MIS fails or times out badly
        # Simple greedy
        selected = []
        remaining = set(range(n_cand))
        while remaining and time.time() < deadline:
            # Pick vertex with max degree? Or just first?
            v = next(iter(remaining))
            selected.append(v)
            # Remove v and its neighbors
            to_remove = {v} | adj[v]
            remaining -= to_remove
        mis_indices = selected
        
    if not mis_indices:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    result = candidates[mis_indices]
    return result
# EVOLVE-BLOCK-END
