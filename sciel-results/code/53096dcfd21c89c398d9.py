"""Rewritten program: exact MIS via branch-and-bound with spectral bounds for kissing number in D=11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to adjacency for MIS: edges in conflict graph = cannot both be in independent set
    # So MIS on conflict graph = maximum set with no two conflicting = our answer
    # adj[i,j] = True means i and j conflict (cannot both be selected)
    adj = conflicts
    
    # Precompute neighbor sets as bitsets for fast operations
    # N4 is about C(11,4)*2^4 = 330*16 = 5280
    # Use Python integers as bitsets for fast AND/OR
    neighbor_bits = []
    for i in range(N4):
        bits = 0
        nbrs = np.where(adj[i])[0]
        for j in nbrs:
            bits |= (1 << j)
        neighbor_bits.append(bits)
    
    # Spectral bound: compute Laplacian eigenvalues for lower bound on independence number
    # L = D - A where A is adjacency (conflict) matrix
    # The independence number alpha(G) >= n * lambda_n / (lambda_1 + lambda_n) for connected graphs
    # where lambda_1 is smallest eigenvalue of L, lambda_n is largest
    # Actually for Laplacian: alpha >= n * lambda_min / (lambda_max - lambda_min + lambda_min)
    # Let's use: alpha >= n * lambda_0 / (lambda_0 + lambda_max) where lambda_0=0 is trivial
    # Better bound: use the second smallest eigenvalue (algebraic connectivity)
    # For now, compute a simple spectral bound
    
    def compute_spectral_bound(subset_indices):
        """Compute a lower bound on independence number of induced subgraph."""
        n_sub = len(subset_indices)
        if n_sub <= 1:
            return n_sub
        if n_sub > 200:  # Too expensive for large subsets
            # Use greedy lower bound instead
            return greedy_lower_bound(subset_indices)
        
        # Build sub-adjacency
        sub_adj = adj[np.ix_(subset_indices, subset_indices)]
        degrees = sub_adj.sum(axis=1).astype(np.float64)
        L = np.diag(degrees) - sub_adj.astype(np.float64)
        
        try:
            eigvals = np.linalg.eigvalsh(L)
            lambda_min = max(eigvals[0], 0)  # Should be ~0
            lambda_max = eigvals[-1]
            if lambda_max > 0:
                bound = n_sub * lambda_min / (lambda_min + lambda_max)
            else:
                bound = n_sub
            return max(1, int(np.floor(bound)))
        except:
            return 1
    
    def greedy_lower_bound(indices):
        """Greedy independent set size as lower bound."""
        in_set = set()
        remaining = set(indices)
        while remaining:
            # Pick vertex with min degree in remaining
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = len(neighbor_bits[v] & _bitmask(remaining))
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            in_set.add(best_v)
            remaining.discard(best_v)
            # Remove neighbors of best_v
            nbr_mask = neighbor_bits[best_v]
            to_remove = []
            for u in remaining:
                if nbr_mask & (1 << u):
                    to_remove.append(u)
            for u in to_remove:
                remaining.discard(u)
        return len(in_set)
    
    def _bitmask(indices):
        """Create bitmask from list of indices."""
        m = 0
        for i in indices:
            m |= (1 << i)
        return m
    
    # Branch and bound for MIS
    def branch_and_bound(candidates, current_set, best_size, best_set, deadline):
        """
        candidates: list of indices that can still be added
        current_set: set of indices currently in the independent set
        best_size: size of best solution found so far
        best_set: the best solution found so far
        """
        if time.time() > deadline:
            return
        
        if len(candidates) == 0:
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
            return best_size, best_set
        
        # Upper bound: |current_set| + |candidates|
        if len(current_set) + len(candidates) <= best_size:
            return best_size, best_set
        
        # Try to get a better upper bound using spectral or greedy
        # Simple bound: use the fact that in the induced subgraph on candidates,
        # the independence number is at most n - min_degree (each vertex with degree d
        # means at most n-1-d can be chosen with it, but this is weak)
        
        # Better: use a greedy coloring upper bound on the conflict graph restricted to candidates
        # In the conflict graph, an independent set can have at most one vertex from each color class
        # So if we color the conflict graph on candidates with c colors, |IS| <= c
        
        def greedy_color_upper_bound(cand_list):
            """Color the conflict subgraph on cand_list, return number of colors used."""
            if len(cand_list) <= 1:
                return len(cand_list)
            
            # Order by degree (descending) for better coloring
            cand_set = set(cand_list)
            degrees = {}
            for v in cand_list:
                nbrs_in_cand = neighbor_bits[v] & _bitmask(cand_list)
                # Count bits
                d = 0
                temp = nbrs_in_cand
                while temp:
                    d += 1
                    temp &= temp - 1
                degrees[v] = d
            
            ordered = sorted(cand_list, key=lambda v: -degrees[v])
            
            # Greedy coloring
            color_of = {}
            next_color = 0
            for v in ordered:
                # Find smallest color not used by neighbors
                used_colors = set()
                nbrs_in_cand = neighbor_bits[v] & _bitmask(cand_list)
                temp = nbrs_in_cand
                while temp:
                    lowbit = temp & (-temp)
                    j = lowbit.bit_length() - 1
                    if j in color_of:
                        used_colors.add(color_of[j])
                    temp ^= lowbit
                
                c = 0
                while c in used_colors:
                    c += 1
                color_of[v] = c
                next_color = max(next_color, c + 1)
            
            return next_color
        
        # Use coloring bound
        color_bound = greedy_color_upper_bound(candidates)
        if len(current_set) + color_bound <= best_size:
            return best_size, best_set
        
        # Select vertex to branch on: pick one with high degree in candidate subgraph
        # to maximize pruning
        cand_set = set(candidates)
        best_v = None
        best_deg = -1
        for v in candidates:
            nbr_mask = neighbor_bits[v] & _bitmask(candidates)
            d = 0
            temp = nbr_mask
            while temp:
                d += 1
                temp &= temp - 1
            if d > best_deg:
                best_deg = d
                best_v = v
        
        # Branch 1: include best_v
        new_current = set(current_set)
        new_current.add(best_v)
        # Candidates that can still be added: those not conflicting with best_v
        nbr_mask = neighbor_bits[best_v]
        new_candidates = [u for u in candidates if u != best_v and not (nbr_mask & (1 << u))]
        
        best_size, best_set = branch_and_bound(
            new_candidates, new_current, best_size, best_set, deadline
        )
        
        # Branch 2: exclude best_v
        new_candidates2 = [u for u in candidates if u != best_v]
        best_size, best_set = branch_and_bound(
            new_candidates2, current_set, best_size, best_set, deadline
        )
        
        return best_size, best_set
    
    # First, get a good initial solution using greedy
    def greedy_independent_set():
        """Greedy min-degree independent set."""
        in_set = set()
        remaining = set(range(N4))
        while remaining:
            # Find vertex with minimum degree in remaining
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                nbr_mask = neighbor_bits[v] & _bitmask(remaining)
                d = 0
                temp = nbr_mask
                while temp:
                    d += 1
                    temp &= temp - 1
                if d < best_deg:
                    best_deg = d
                    best_v = v
            in_set.add(best_v)
            remaining.discard(best_v)
            # Remove neighbors
            nbr_mask = neighbor_bits[best_v]
            to_remove = []
            for u in remaining:
                if nbr_mask & (1 << u):
                    to_remove.append(u)
            for u in to_remove:
                remaining.discard(u)
        return in_set
    
    # Get initial solution
    initial_set = greedy_independent_set()
    best_size = len(initial_set)
    best_set = set(initial_set)
    
    # Now try branch and bound to improve
    # Start with all vertices as candidates
    all_candidates = list(range(N4))
    
    # We'll do a limited B&B with time limit
    bb_deadline = deadline - 5
    best_size, best_set = branch_and_bound(
        all_candidates, set(), best_size, best_set, bb_deadline
    )
    
    # If B&B didn't finish, try some local search improvements
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 2:
        # Try to add vertices greedily
        current = set(best_set)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for v in range(N4):
                if v in current:
                    continue
                # Check if v conflicts with any in current
                nbr_mask = neighbor_bits[v]
                conflict = False
                for u in current:
                    if nbr_mask & (1 << u):
                        conflict = True
                        break
                if not conflict:
                    current.add(v)
                    changed = True
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
    
    # Build final result
    final_v4_indices = np.array(sorted(best_set), dtype=np.int64)
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
