"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 512
    n = N
    
    def time_check():
        return time.time() >= deadline
    
    # Build adjacency using bitsets (Python ints) for fast intersection
    # adj[i] is a bitmask where bit j is set if node i and node j are compatible
    adj = [0] * n
    
    for i in range(n):
        vi = i + 1
        mask = 0
        for j in range(n):
            if i == j:
                continue
            vj = j + 1
            g = gcd(vi, vj)
            lcm_val = vi * vj // g
            if lcm_val <= N:
                mask |= (1 << j)
        adj[i] = mask
    
    # degrees[i] = number of compatible neighbors
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    
    # Bitset-based greedy: find a large clique
    def greedy_clique(order):
        """Build a clique greedily using bitset intersection."""
        clique = []
        # candidates is a bitmask of nodes still compatible with all in clique
        candidates = (1 << n) - 1
        for v in order:
            if not (candidates & (1 << v)):
                continue
            clique.append(v)
            candidates &= adj[v]
        return clique
    
    # Strategy 1: Multiple greedy passes with different orderings
    import random
    rng = random.Random(12345)
    
    best_clique = []
    best_size = 0
    
    # Greedy by degree descending
    order_by_degree = sorted(range(n), key=lambda x: -degrees[x])
    clique = greedy_clique(order_by_degree)
    best_size = len(clique)
    best_clique = clique
    
    # Greedy by value ascending (smaller numbers tend to have smaller lcm)
    order_by_value = list(range(n))
    clique = greedy_clique(order_by_value)
    if len(clique) > best_size:
        best_size = len(clique)
        best_clique = clique
    
    # Random orderings
    for trial in range(500):
        if time_check():
            break
        order = list(range(n))
        rng.shuffle(order)
        clique = greedy_clique(order)
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = clique
    
    if time_check():
        return [i + 1 for i in best_clique]
    
    # Strategy 2: Local search - try to add nodes to the clique
    def try_extend(clique):
        """Try to add any node compatible with all in clique."""
        clique_set = set(clique)
        # Find candidates compatible with all in clique
        candidates_mask = (1 << n) - 1
        for u in clique:
            candidates_mask &= adj[u]
        # Remove nodes already in clique
        for u in clique:
            candidates_mask &= ~(1 << u)
        
        if candidates_mask == 0:
            return None, None
        
        # Try to add one node
        # Pick the one with highest degree among candidates
        best_v = -1
        best_deg = -1
        temp = candidates_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            if degrees[v] > best_deg:
                best_deg = degrees[v]
                best_v = v
            temp ^= lsb
        
        if best_v >= 0:
            return clique + [best_v], best_v
        return None, None
    
    # Local search with swaps
    def local_search(clique):
        """Improve clique by trying to add or swap nodes."""
        improved = True
        while improved:
            improved = False
            # Try adding
            new_clique, added = try_extend(clique)
            if new_clique is not None:
                clique = new_clique
                improved = True
                continue
            
            # Try swap: remove one node, try to add two
            for idx, u in enumerate(clique):
                if time_check():
                    return clique
                remaining = [clique[j] for j in range(len(clique)) if j != idx]
                # Find candidates compatible with all in remaining
                candidates_mask = (1 << n) - 1
                for w in remaining:
                    candidates_mask &= adj[w]
                # Exclude nodes already in remaining and u
                for w in remaining:
                    candidates_mask &= ~(1 << w)
                candidates_mask &= ~(1 << u)
                
                if candidates_mask == 0:
                    continue
                
                # Get list of candidates
                cand_list = []
                temp = candidates_mask
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    cand_list.append(v)
                    temp ^= lsb
                
                # Sort by degree descending
                cand_list.sort(key=lambda v: -degrees[v])
                
                # Try to find two compatible candidates
                for ia in range(min(len(cand_list), 100)):
                    if time_check():
                        return clique
                    va = cand_list[ia]
                    for ib in range(ia + 1, min(len(cand_list), ia + 100)):
                        vb = cand_list[ib]
                        if (adj[va] & (1 << vb)) != 0:
                            # Both compatible with each other and all in remaining
                            new_clique = remaining + [va, vb]
                            if len(new_clique) > len(clique):
                                clique = new_clique
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return clique
    
    best_clique = local_search(best_clique)
    best_size = len(best_clique)
    
    if time_check():
        return [i + 1 for i in best_clique]
    
    # Strategy 3: Maximum clique via branch and bound with bitsets
    # Use a proper coloring-based B&B
    
    def max_clique_bnb(initial_clique):
        """Find maximum clique using branch and bound with bitset operations."""
        best = list(initial_clique)
        best_size = len(best)
        
        # Color the candidate set for upper bounds
        # We use a greedy coloring where we process nodes in order and assign
        # the smallest available color such that no two adjacent nodes share a color
        # Actually for clique upper bound, we need to color the complement (independent set coloring)
        # Standard approach: order candidates, color greedily, the max color gives upper bound
        
        def color_candidates(cand_mask):
            """Color candidates to get upper bound on clique size in this set."""
            # Get list of candidates
            cand_list = []
            temp = cand_mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                cand_list.append(v)
                temp ^= lsb
            
            if not cand_list:
                return 0, []
            
            # Sort by degree descending for better coloring
            cand_list.sort(key=lambda v: -degrees[v])
            
            # Greedy coloring: assign each vertex the smallest color not used by its neighbors
            # that have already been colored
            color_of = {}
            next_color = 1
            # Process in reverse degree order (high degree first)
            for v in cand_list:
                # Find colors used by colored neighbors
                neighbor_colors = set()
                for u in cand_list:
                    if u == v:
                        continue
                    if u in color_of and (adj[v] & (1 << u)) != 0:
                        neighbor_colors.add(color_of[u])
                # Assign smallest available color
                c = next_color
                while c in neighbor_colors:
                    c += 1
                color_of[v] = c
                if c > next_color:
                    next_color = c + 1
            
            max_color = max(color_of.values()) if color_of else 0
            # Order by color descending for B&B
            ordered = sorted(cand_list, key=lambda v: -color_of[v])
            return max_color, ordered
        
        def bnb(cand_mask, current_clique):
            nonlocal best, best_size
            
            if time_check():
                return
            
            if cand_mask == 0:
                if len(current_clique) > best_size:
                    best = list(current_clique)
                    best_size = len(best)
                return
            
            # Get upper bound via coloring
            max_color, ordered = color_candidates(cand_mask)
            if len(current_clique) + max_color <= best_size:
                return
            
            # Process candidates in order
            for v in ordered:
                if time_check():
                    return
                
                # Upper bound: if even adding v and all remaining can't beat best, stop
                # We need to compute the bound more carefully
                # For now, just proceed
                if len(current_clique) + 1 <= best_size and v == ordered[-1]:
                    break
                
                # Check if v is still in cand_mask
                if not (cand_mask & (1 << v)):
                    continue
                
                # Add v to clique
                current_clique.append(v)
                # New candidates: those in cand_mask that are compatible with v and come after v in order
                new_cand_mask = cand_mask & adj[v]
                # Remove v itself
                new_cand_mask &= ~(1 << v)
                # Only keep candidates that come after v in the ordering
                # For simplicity, keep all (less pruning but correct)
                bnb(new_cand_mask, current_clique)
                current_clique.pop()
                
                # Remove v from cand_mask for subsequent iterations
                cand_mask &= ~(1 << v)
        
        # Initial candidates: all nodes compatible with all in initial_clique
        init_mask = (1 << n) - 1
        for u in initial_clique:
            init_mask &= adj[u]
            init_mask &= ~(1 << u)
        
        bnb(init_mask, list(initial_clique))
        return best
    
    best_clique = max_clique_bnb(best_clique)
    best_size = len(best_clique)
    
    if time_check():
        return [i + 1 for i in best_clique]
    
    # Strategy 4: Try to find even larger clique with more aggressive search
    # Multiple restarts of B&B from different starting points
    for restart in range(20):
        if time_check():
            break
        
        # Create a new starting clique with some randomness
        order = list(range(n))
        rng.shuffle(order)
        # Greedy to get a decent starting point
        start_clique = greedy_clique(order)
        
        # Extend with local search
        start_clique = local_search(start_clique)
        
        # B&B from this starting point
        result = max_clique_bnb(start_clique)
        if len(result) > best_size:
            best_size = len(result)
            best_clique = result
    
    return [i + 1 for i in best_clique]
# EVOLVE-BLOCK-END
