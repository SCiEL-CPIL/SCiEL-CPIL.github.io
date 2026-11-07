"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        u, v = e
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for u in range(n):
        for v in range(n):
            if (u, v) in edge_set:
                adj[u] |= (1 << v)
    
    # Complement adjacency for independent sets
    comp_adj = [0] * n
    for u in range(n):
        # All vertices except u and its neighbors in G are its neighbors in G_bar
        all_mask = (1 << n) - 1
        comp_adj[u] = all_mask & ~(adj[u] | (1 << u))

    # Function to check if a set of vertices (bitmask) is a clique
    def is_clique_mask(mask):
        if mask == 0:
            return True
        # Get first vertex
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        # All other vertices must be adjacent to v
        # adj[v] contains neighbors of v. We check if all bits in 'rest' are in adj[v]
        return (rest & ~adj[v]) == 0

    # Function to check if a set of vertices (bitmask) is an independent set
    def is_independent_mask(mask):
        if mask == 0:
            return True
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        # In independent set, no edges between vertices.
        # So no vertex in 'rest' should be a neighbor of v.
        # i.e., rest & adj[v] should be 0.
        return (rest & adj[v]) == 0

    # Bron-Kerbosch with pivot to find all maximal cliques
    # We represent sets as bitmasks for efficiency in the outer loop, 
    # but standard BK uses sets. For n=14, we can use bitmasks in BK too.
    
    def bron_kerbosch(R, P_mask, X_mask, adj_list):
        """
        R: current clique (bitmask)
        P_mask: candidates (bitmask)
        X_mask: excluded (bitmask)
        adj_list: list of adjacency bitmasks
        Yields maximal cliques as bitmasks
        """
        if P_mask == 0 and X_mask == 0:
            if R != 0:
                yield R
            return
        
        # Pivot selection: choose u in P_mask U X_mask to minimize branching
        # We want to iterate over P_mask \ N(u)
        # If P_mask is empty, we must have X_mask non-empty? No, base case handled.
        # If P_mask is not empty, pick pivot from P_mask U X_mask.
        # To minimize branching, pick u with max degree in P_mask U X_mask?
        # Actually, standard is: pick u in P U X. Branch on v in P \ N(u).
        
        # If P_mask is 0, we should have returned.
        # Let's pick a pivot.
        # For simplicity and speed with n=14, we can just pick the first vertex in P_mask U X_mask.
        # Better: pick u such that |P_mask \ N(u)| is minimized.
        
        union_px = P_mask | X_mask
        if union_px == 0:
            return
            
        # Try all vertices in union_px as pivot? No, just pick one.
        # Heuristic: pick u with max degree in P_mask.
        # For n=14, even picking the first one is fast enough.
        # Let's pick the first set bit in P_mask if P_mask != 0, else first in X_mask.
        # Actually, the pivot can be from P or X.
        # Let's just pick the first available vertex from P_mask if it exists, else X_mask.
        # This is suboptimal but n is small.
        
        pivot_mask = P_mask if P_mask else X_mask
        # If pivot_mask is 0, we return (handled above)
        
        # Get first vertex
        lsb = pivot_mask & -pivot_mask
        u = lsb.bit_length() - 1
        
        # Candidates: v in P_mask such that v is NOT adjacent to u
        # adj_list[u] has bits for neighbors of u.
        # So we want P_mask & ~adj_list[u]
        candidates = P_mask & ~adj_list[u]
        
        while candidates:
            # Pick a vertex v from candidates
            v_lsb = candidates & -candidates
            v = v_lsb.bit_length() - 1
            
            # Recurse
            new_R = R | v_lsb
            new_P = P_mask & adj_list[v]
            new_X = X_mask & adj_list[v]
            
            yield from bron_kerbosch(new_R, new_P, new_X, adj_list)
            
            # Remove v from P, add to X
            P_mask &= ~v_lsb
            X_mask |= v_lsb
            
            # Remove v from candidates
            candidates &= ~v_lsb

    # Find all maximal cliques
    all_cliques = set()
    for c in bron_kerbosch(0, (1 << n) - 1, 0, adj):
        all_cliques.add(c)
        
    # Find all maximal independent sets (maximal cliques in complement)
    all_independent = set()
    for c in bron_kerbosch(0, (1 << n) - 1, 0, comp_adj):
        all_independent.add(c)

    # Combine all homogeneous sets
    homogeneous_masks = list(all_cliques | all_independent)
    
    # Filter out empty sets (shouldn't exist from maximal BK if n>0, but safe)
    homogeneous_masks = [m for m in homogeneous_masks if m != 0]
    
    # Sort by size descending to find large sets first
    homogeneous_masks.sort(key=lambda m: m.bit_count(), reverse=True)
    
    # Map mask to index for quick lookup? Or just use list.
    # Precompute list of vertices for each mask for final output
    mask_to_verts = {}
    for m in homogeneous_masks:
        verts = []
        temp = m
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            verts.append(v)
            temp ^= lsb
        mask_to_verts[m] = sorted(verts)

    # Branch and Bound for Minimum Set Partition
    # State: remaining_mask, current_partition (list of masks), count
    # Goal: cover all vertices with disjoint homogeneous sets
    
    best_count = n + 1
    best_partition = []
    
    # Heuristic: Order of sets in branching
    # We will pick the "most constrained" uncovered vertex.
    
    def solve(remaining_mask, current_partition):
        nonlocal best_count, best_partition
        
        count = len(current_partition)
        
        if count >= best_count:
            return
            
        if remaining_mask == 0:
            if count < best_count:
                best_count = count
                best_partition = current_partition[:]
            return
        
        # Lower bound:
        # At least 1 more set needed.
        # Stronger: ceil(popcount(remaining) / max_set_size_in_remaining)
        # But max_set_size depends on which sets can cover the remaining vertices.
        # Simple LB: 1 if remaining_mask != 0
        
        # Find the vertex in remaining_mask with the fewest candidate sets
        # A candidate set must be a subset of remaining_mask.
        
        # Optimization: If we can cover the rest with 1 set, do it.
        # Check if any homogeneous set is a superset of remaining_mask?
        # No, the set must be exactly the remaining vertices? No, the set must be a subset of remaining, 
        # and the union of chosen sets must be the original set.
        # Wait, the problem is partition. So the chosen sets must be disjoint and their union is the full set.
        # So if we pick a set S, it must be a subset of remaining_mask.
        # The remaining vertices are remaining_mask \ S.
        
        # To speed up, we can check if the entire remaining_mask is itself a homogeneous set.
        if remaining_mask in all_cliques or remaining_mask in all_independent:
            # We can cover it with 1 set
            if count + 1 < best_count:
                best_count = count + 1
                best_partition = current_partition + [remaining_mask]
            return

        # Find the "hardest" vertex
        # For each vertex v in remaining_mask, count how many homogeneous sets S
        # satisfy: S is subset of remaining_mask AND v is in S.
        
        # This is expensive if done naively for every node.
        # Alternative: Just iterate through homogeneous masks.
        # Since n=14, number of homogeneous masks is likely small (< 100).
        
        # Let's find all valid sets for the current remaining_mask
        valid_sets_for_remaining = []
        for m in homogeneous_masks:
            if (m & remaining_mask) == m:
                valid_sets_for_remaining.append(m)
        
        if not valid_sets_for_remaining:
            return # Dead end

        # Pick the vertex with the fewest covering sets
        # We can just pick the first vertex in remaining_mask and branch on sets containing it?
        # That's a valid strategy (Exact Cover).
        # Let's pick the vertex v in remaining_mask.
        # We must pick a set containing v.
        
        # Which v to pick?
        # Let's pick v with the minimum number of valid_sets_for_remaining that contain v.
        
        # Get vertices in remaining_mask
        verts_in_rem = []
        temp = remaining_mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            verts_in_rem.append(v)
            temp ^= lsb
            
        min_count = float('inf')
        target_v = -1
        
        for v in verts_in_rem:
            v_bit = 1 << v
            cnt = 0
            for m in valid_sets_for_remaining:
                if m & v_bit:
                    cnt += 1
            if cnt < min_count:
                min_count = cnt
                target_v = v
                if cnt == 1:
                    break # Can't get better than 1
        
        if target_v == -1:
            return
            
        # Branch on sets that contain target_v
        # Sort these sets by size descending to find large covers first
        candidate_sets = [m for m in valid_sets_for_remaining if (m & (1 << target_v))]
        candidate_sets.sort(key=lambda m: m.bit_count(), reverse=True)
        
        for m in candidate_sets:
            new_remaining = remaining_mask & ~m
            current_partition.append(m)
            solve(new_remaining, current_partition)
            current_partition.pop()

    solve((1 << n) - 1, [])
    
    if best_partition:
        result = []
        for m in best_partition:
            result.append(mask_to_verts[m])
        return result
    else:
        # Fallback
        return [[v] for v in range(n)]

# EVOLVE-BLOCK-END
