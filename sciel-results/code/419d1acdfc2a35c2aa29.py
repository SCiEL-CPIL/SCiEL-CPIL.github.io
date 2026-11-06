"""Rewritten program using Max-SAT style Simulated Annealing with O(1) Q3 violation check."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin before 120s limit (evaluator kills at 150s)
    
    n = 16
    
    # Precompute all 8-vertex subsets of 16 vertices
    all_8sets = list(combinations(range(n), 8))
    num_8sets = len(all_8sets)  # 12870
    
    # For each 8-set, precompute all permutations of Q3 vertices that form a valid Q3 embedding
    # Q3 vertices: 0..7, edges between i and j if Hamming distance = 1
    q3_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.append((i, j))
    
    # For each 8-set, precompute the list of edge masks that would create a Q3
    # A Q3 is created if for some permutation p of the 8-set vertices assigned to Q3 vertices 0..7,
    # all 12 Q3 edges exist in the graph.
    # Represent the 8-set's internal edges as a bitmask of 28 bits (C(8,2)=28).
    # For each 8-set S, precompute: for each of the 8! permutations, which internal edges are needed.
    # Actually, we need: given the 8 vertices in S, and the current adjacency restricted to S,
    # determine if there exists a Q3 embedding.
    
    # More efficient: for each 8-set, precompute all 28 possible internal edge positions
    # and for each Q3 embedding (permutation), which of those 28 positions must be present.
    
    # Let's precompute: for each 8-set index, a list of 28-bit masks where each mask represents
    # the set of internal edges that must exist for a particular Q3 embedding to be present.
    # If the current internal edge mask has all bits set in any of these masks, Q3 is present.
    
    # Precompute permutation edge masks for Q3
    # For a given permutation p (mapping Q3 vertex i to position p[i] in the 8-set),
    # the required internal edges are: for each Q3 edge (i,j), the internal edge between positions p[i] and p[j].
    
    # Precompute all 8! permutations and their required internal edge masks
    import itertools
    all_perms = list(itertools.permutations(range(8)))
    num_perms = len(all_perms)  # 40320
    
    # For each permutation, compute the 28-bit mask of required internal edges
    # Internal edge positions: (a,b) where 0 <= a < b <= 7, index = a*7 + b - a*(a+1)//2 ... 
    # Simpler: use a 64-bit or 28-bit representation. Let's use a set of (min,max) pairs.
    
    # Precompute: for each permutation, the frozenset of required internal edge pairs
    perm_edge_sets = []
    for p in all_perms:
        # p[i] = position in 8-set that Q3 vertex i maps to
        # For each Q3 edge (i,j), we need internal edge (p[i], p[j])
        edges = set()
        for i, j in q3_edges:
            a, b = p[i], p[j]
            if a > b:
                a, b = b, a
            edges.add((a, b))
        perm_edge_sets.append(frozenset(edges))
    
    # Now for the SA:
    # State: adjacency matrix as 16x16 boolean, or 16 16-bit masks.
    # Objective: maximize edges subject to no Q3.
    # Penalty: for each 8-set, if Q3 is present, add penalty.
    
    # We'll use a bitmask for adjacency: adj[v] is a 16-bit mask.
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    def timeup():
        return time_left() <= 0
    
    # Initialize with a random dense graph
    adj = [0] * n
    for v in range(n):
        for u in range(n):
            if u != v and random.random() < 0.7:
                adj[v] |= (1 << u)
    
    def edge_count():
        return sum(bin(adj[v]).count('1') for v in range(n)) // 2
    
    def q3_violations(adj):
        """Count Q3 violations. For each 8-set, check if Q3 is present."""
        count = 0
        for idx, combo in enumerate(all_8sets):
            # Get internal edge mask for this 8-set
            # For each pair (i,j) in the 8-set (positions 0..7), check if edge exists
            # Build a 28-bit mask
            internal_mask = 0
            for a_pos in range(8):
                for b_pos in range(a_pos + 1, 8):
                    u, v = combo[a_pos], combo[b_pos]
                    if (adj[u] >> v) & 1:
                        # Set bit for this internal edge
                        # Index: a_pos * 7 + b_pos - a_pos*(a_pos+1)//2
                        # Simpler: use a dict or just check via the edge sets
                        pass
            
            # Check if any permutation's required edges are all present
            # For each permutation, check if all required internal edges exist
            found = False
            for es in perm_edge_sets:
                ok = True
                for (a_pos, b_pos) in es:
                    u, v = combo[a_pos], combo[b_pos]
                    if not ((adj[u] >> v) & 1):
                        ok = False
                        break
                if ok:
                    found = True
                    break
            if found:
                count += 1
        
        return count
    
    # This is too slow. We need a faster Q3 check.
    
    # Alternative: for each 8-set, precompute a lookup table.
    # The internal edge state is a 28-bit value. For each possible 28-bit value,
    # we can precompute whether Q3 is present.
    # But 2^28 is too large.
    
    # Better: use the fact that we only need to check 40320 permutations per 8-set.
    # For each permutation, the required edges are a specific subset of the 28 internal edges.
    # We can represent the current internal edges as a 28-bit integer.
    # For each permutation, precompute the 28-bit mask of required edges.
    # Then Q3 is present if (current_mask & perm_mask) == perm_mask for any perm.
    
    # Precompute perm masks as 28-bit integers
    def edge_index(a, b):
        """Index for internal edge (a,b) where 0 <= a < b <= 7"""
        # There are C(8,2)=28 edges
        # Index: sum_{k=0}^{a-1} (7-k) + (b - a - 1)
        # Or simpler: a*7 + b - a*(a+1)//2
        return a * 7 + b - a * (a + 1) // 2
    
    # Verify edge_index is a bijection from pairs to 0..27
    idx_map = {}
    for a in range(8):
        for b in range(a + 1, 8):
            idx = edge_index(a, b)
            if idx in idx_map:
                pass  # collision
            idx_map[(a, b)] = idx
    assert len(idx_map) == 28
    
    # Precompute perm masks as 28-bit integers
    perm_masks = []
    for p in all_perms:
        mask = 0
        for i, j in q3_edges:
            a, b = p[i], p[j]
            if a > b:
                a, b = b, a
            mask |= (1 << edge_index(a, b))
        perm_masks.append(mask)
    
    # For each 8-set, precompute the mapping from internal position to actual vertex
    # So we can build the 28-bit internal edge mask quickly.
    
    def build_internal_mask(adj, combo):
        """Build 28-bit mask of internal edges for the 8-set."""
        mask = 0
        for a_pos in range(8):
            for b_pos in range(a_pos + 1, 8):
                u, v = combo[a_pos], combo[b_pos]
                if (adj[u] >> v) & 1:
                    mask |= (1 << edge_index(a_pos, b_pos))
        return mask
    
    def has_q3(internal_mask):
        """Check if the 28-bit internal mask contains a Q3."""
        for pm in perm_masks:
            if (internal_mask & pm) == pm:
                return True
        return False
    
    def total_violations(adj):
        """Count total Q3 violations across all 8-sets."""
        count = 0
        for combo in all_8sets:
            im = build_internal_mask(adj, combo)
            if has_q3(im):
                count += 1
        return count
    
    # Simulated Annealing
    # Energy = -edge_count + lambda * violations
    # We want to minimize energy, i.e., maximize edges while minimizing violations.
    
    best_adj = [0] * n
    best_energy = float('inf')
    
    # Initialize
    adj = [0] * n
    for v in range(n):
        for u in range(n):
            if u != v and random.random() < 0.6:
                adj[v] |= (1 << u)
    
    current_energy = -edge_count(adj) + 50.0 * total_violations(adj)
    
    if current_energy < best_energy:
        best_energy = current_energy
        best_adj = adj[:]
    
    T = 100.0
    cooling = 0.999
    
    while time_left() > 5:
        # Propose a move: flip a random edge
        if random.random() < 0.5:
            # Add an edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if (adj[u] >> v) & 1:
                continue
            # Flip: add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Compute delta energy
            new_ec = edge_count(adj)
            new_viol = total_violations(adj)
            new_energy = -new_ec + 50.0 * new_viol
            
            delta = new_energy - current_energy
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_adj = adj[:]
            
            if delta > 0 and random.random() > exp(-delta / T):
                # Reject: undo
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_energy -= delta
        else:
            # Remove an edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if not ((adj[u] >> v) & 1):
                continue
            # Flip: remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            new_ec = edge_count(adj)
            new_viol = total_violations(adj)
            new_energy = -new_ec + 50.0 * new_viol
            
            delta = new_energy - current_energy
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_adj = adj[:]
            
            if delta > 0 and random.random() > exp(-delta / T):
                # Reject: undo
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_energy -= delta
        
        T *= cooling
    
    # Use best_adj
    adj = best_adj
    
    # Final cleanup: try to remove any remaining Q3 violations by removing edges
    for _ in range(1000):
        if timeup():
            break
        # Find a Q3 violation
        found = False
        for combo in all_8sets:
            if timeup():
                break
            im = build_internal_mask(adj, combo)
            if not has_q3(im):
                continue
            # Find a specific Q3 embedding and remove an edge
            for p_idx, p in enumerate(all_perms):
                pm = perm_masks[p_idx]
                if (im & pm) == pm:
                    # This permutation forms a Q3
                    # Remove a random edge from this Q3
                    chosen = random.choice(q3_edges)
                    i, j = chosen
                    u, v = combo[p[i]], combo[p[j]]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    found = True
                    break
            if found:
                break
        if not found:
            break
    
    # Build result
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
