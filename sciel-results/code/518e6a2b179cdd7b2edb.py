import time
import random
import sys

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def greedy_set_cover_clique_partition(adj, n, deadline):
        # 1. Enumerate all maximal cliques
        # We use Bron-Kerbosch with pivot to find maximal cliques.
        # We store them as bitmasks.
        max_cliques = []
        
        # We need to be careful with recursion depth or use iterative BK.
        # Given n <= 64, recursion depth is at most 64, which is fine.
        
        def bron_kerbosch(R, P, X):
            if P == 0 and X == 0:
                if R.bit_count() >= 2:
                    max_cliques.append(R)
                return
            
            # Choose pivot u from P | X with max degree in P
            pivot_mask = P | X
            if pivot_mask == 0:
                candidates = P
            else:
                best_pivot = -1
                max_deg = -1
                pm = pivot_mask
                while pm:
                    lsb = pm & -pm
                    u = lsb.bit_length() - 1
                    deg = (adj[u] & P).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        best_pivot = u
                    pm ^= lsb
                
                if best_pivot != -1:
                    # Branch on P \ N(u)
                    candidates = P & ~adj[best_pivot]
                else:
                    candidates = P
            
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                
                # Recurse
                bron_kerbosch(R | lsb, P & adj[v], X & adj[v])
                
                # Update P and X
                P ^= lsb
                X |= lsb
                c ^= lsb
                
                if time.time() > deadline:
                    return

        # Start BK
        # To ensure we find all maximal cliques, we start with R=0, P=All, X=0.
        # However, standard BK finds maximal cliques.
        # We need to make sure we don't time out during enumeration.
        # For n=64, p=0.55, the number of maximal cliques can be large but usually manageable.
        # If it's too slow, we might need a limit, but let's try full enumeration first.
        
        start_time = time.time()
        # Reserve time for the set cover part
        # Let's give 50% of the time to enumeration and 50% to set cover?
        # Or just run enumeration until deadline - buffer?
        # Let's say we have 120s total.
        # Enumeration might be fast for these sizes.
        
        all_vertices_mask = (1 << n) - 1
        bron_kerbosch(0, all_vertices_mask, 0)
        
        # If we timed out during enumeration, max_cliques might be incomplete.
        # We should check if we have enough cliques to cover all edges?
        # Actually, if we timed out, we might miss some edges.
        # To be safe, if we timed out, we should fall back to a simpler method or ensure validity.
        # But let's assume enumeration finishes for these sizes.
        # If not, we can add a safety check: if max_cliques is empty, return edges as cliques.
        
        if not max_cliques:
            # Fallback: just return all edges as cliques
            cliques = []
            for u in range(n):
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if v > u:
                        cliques.append([u, v])
                    mask ^= lsb
            return cliques

        # 2. Greedy Set Cover
        # We want to cover all edges.
        # Each clique C covers the set of edges within C.
        # We maintain a bitmask of uncovered edges?
        # Edges are pairs (u, v). There are n*(n-1)/2 edges.
        # For n=64, that's ~2016 edges. We can use a Python int as a bitset for edges.
        # We need a mapping from edge (u, v) to an index.
        
        edge_indices = {}
        edge_idx = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edge_indices[(u, v)] = edge_idx
                    edge_idx += 1
        
        # Total edges
        total_edges = edge_idx
        
        # Initialize uncovered mask
        # If total_edges > 63, we need big integers, which Python supports natively.
        uncovered = (1 << total_edges) - 1
        
        # For each clique, precompute the mask of edges it covers
        clique_edge_masks = []
        for c in max_cliques:
            # Get vertices in clique
            verts = []
            mask = c
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                verts.append(v)
                mask ^= lsb
            
            # Compute edge mask
            emask = 0
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    u, v = verts[i], verts[j]
                    idx = edge_indices.get((u, v))
                    if idx is not None:
                        emask |= (1 << idx)
            clique_edge_masks.append(emask)
            
            if time.time() > deadline:
                break

        # Greedy selection
        selected_cliques = []
        # We need to select cliques to cover 'uncovered'.
        # At each step, pick the clique that covers the most new edges.
        # To speed this up, we can maintain a list of (clique_idx, mask) and update scores?
        # Or just recompute "new edges covered" each time?
        # Number of cliques can be in the hundreds/thousands.
        # Number of steps is the number of selected cliques.
        # Let's try a simple loop.
        
        remaining_cliques = list(range(len(clique_edge_masks)))
        
        while uncovered and remaining_cliques:
            best_idx = -1
            best_new_count = 0
            best_mask = 0
            
            for idx in remaining_cliques:
                cm = clique_edge_masks[idx]
                new_covered = cm & uncovered
                if new_covered:
                    count = new_covered.bit_count()
                    if count > best_new_count:
                        best_new_count = count
                        best_idx = idx
                        best_mask = new_covered
                        
                        # Early exit if we found a clique that covers all remaining edges?
                        if count == uncovered.bit_count():
                            break
            
            if best_idx == -1:
                # No clique covers any new edges? This shouldn't happen if max_cliques are correct.
                # But if it does, break to avoid infinite loop.
                break
            
            # Select this clique
            selected_cliques.append(max_cliques[best_idx])
            uncovered ^= best_mask # Remove covered edges
            
            # Remove selected clique from remaining (or just mark as used)
            # Since we don't reuse cliques, we can remove it.
            remaining_cliques.remove(best_idx)
            
            # Time check
            if time.time() > deadline:
                break

        # Safety: Cover any remaining edges as size-2 cliques
        # This ensures validity.
        final_cliques = []
        for mask in selected_cliques:
            verts = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            if len(verts) >= 2:
                final_cliques.append(verts)
        
        # Add remaining edges
        # We need to know which edges are still uncovered.
        # We have 'uncovered' bitmask.
        # We need to map back to (u, v).
        # This is a bit expensive if we do it naively, but n is small.
        
        # Let's just iterate through all possible edges and check if they are in 'uncovered'.
        # Or better, iterate through the bits of 'uncovered'.
        
        # We need a reverse mapping from edge index to (u, v).
        # We built edge_indices as (u,v) -> idx.
        # Let's build idx -> (u,v).
        idx_to_edge = [None] * total_edges
        for (u, v), idx in edge_indices.items():
            idx_to_edge[idx] = (u, v)
            
        rem = uncovered
        while rem:
            lsb = rem & -rem
            idx = lsb.bit_length() - 1
            u, v = idx_to_edge[idx]
            final_cliques.append([u, v])
            rem ^= lsb
            
            if time.time() > deadline:
                break

        return final_cliques

    start_time = time.time()
    # Allow 145s for computation, leaving 5s buffer
    deadline = start_time + 145
    
    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)
    
    # Run for instance 1
    cliques_1 = greedy_set_cover_clique_partition(adj_1, 32, deadline)
    
    # Run for instance 2
    # Update deadline? No, deadline is absolute time.
    # But we should check if we are close to deadline.
    # If we are, we might not finish instance 2 properly.
    # Let's give instance 1 half the time?
    # Or just run and hope.
    # Given the sizes, instance 1 is much smaller.
    
    cliques_2 = greedy_set_cover_clique_partition(adj_2, 64, deadline)
    
    return [cliques_1, cliques_2]

if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
    print(f"Total: {len(res[0]) + len(res[1])}")
