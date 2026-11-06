import time
import random
import sys
import itertools

DEADLINE = 295.0  # Safety margin before 300s limit

_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0, DEADLINE - _elapsed())

def _paley_37():
    """Construct the Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [[0] * q for _ in range(q)]
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append((i, j))
    return q, adj, edges

def _build_bitsets(adj, n):
    """Build bitset representation of neighbors."""
    neigh = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and adj[i][j]:
                mask |= (1 << j)
        neigh[i] = mask
    return neigh

def _find_all_k_sets(neigh, n, k, is_clique=True):
    """Find all sets of size k that are either cliques or independent sets.
    Returns a list of bitmasks."""
    results = []
    if is_clique:
        def check(subset_mask):
            # Check if all pairs in subset are connected
            # For small k (4), we can just iterate
            vertices = []
            m = subset_mask
            while m:
                v_bit = m & (-m)
                v = v_bit.bit_length() - 1
                vertices.append(v)
                m ^= v_bit
            
            # Check clique
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    if not (neigh[vertices[i]] & (1 << vertices[j])):
                        return False
            return True
    else:
        def check(subset_mask):
            vertices = []
            m = subset_mask
            while m:
                v_bit = m & (-m)
                v = v_bit.bit_length() - 1
                vertices.append(v)
                m ^= v_bit
            
            # Check independent set (no edges between any pair)
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    if (neigh[vertices[i]] & (1 << vertices[j])):
                        return False
            return True

    # Backtracking to find all k-sets
    def backtrack(start, current_mask, current_size):
        if current_size == k:
            results.append(current_mask)
            return
        if start >= n:
            return
        # Pruning: if we can't reach size k, return
        if n - start < k - current_size:
            return
            
        for v in range(start, n):
            new_mask = current_mask | (1 << v)
            # Optimization: only add v if it's compatible with current set
            # For clique: v must be connected to all in current_mask
            # For independent: v must be disconnected from all in current_mask
            if is_clique:
                # Check if v is connected to all vertices in current_mask
                compatible = True
                cm = current_mask
                while cm:
                    u_bit = cm & (-cm)
                    u = u_bit.bit_length() - 1
                    if not (neigh[v] & (1 << u)):
                        compatible = False
                        break
                    cm ^= u_bit
                if not compatible:
                    continue
            else:
                # Check if v is NOT connected to all vertices in current_mask
                compatible = True
                cm = current_mask
                while cm:
                    u_bit = cm & (-cm)
                    u = u_bit.bit_length() - 1
                    if (neigh[v] & (1 << u)):
                        compatible = False
                        break
                    cm ^= u_bit
                if not compatible:
                    continue
                    
            backtrack(v + 1, new_mask, current_size + 1)
            
    backtrack(0, 0, 0)
    return results

def _update_k_sets(neigh, n, k, is_clique, new_vertex_idx, new_neigh_mask, old_k_sets):
    """Update the list of k-sets after adding a new vertex.
    This is a simplified version; for robustness, we might just recompute if n is small.
    Given n <= 42, recomputing might be faster and less error-prone than incremental update."""
    pass # We will recompute instead

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with Paley(37)
    n, adj, edges = _paley_37()
    
    best_n = n
    best_adj = [row[:] for row in adj]
    best_edges = edges[:]
    
    # Precompute K4 and I4 for the initial graph
    neigh = _build_bitsets(adj, n)
    k4_masks = _find_all_k_sets(neigh, n, 4, is_clique=True)
    i4_masks = _find_all_k_sets(neigh, n, 4, is_clique=False)
    
    # Function to check if a candidate neighbor mask is valid
    def is_valid_candidate(candidate_mask, n, k4_masks, i4_masks):
        # Check no K4 is subset of candidate_mask
        for mask in k4_masks:
            if (mask & candidate_mask) == mask:
                return False
        # Check no I4 is subset of ~candidate_mask
        # ~candidate_mask within n bits
        full_mask = (1 << n) - 1
        non_neighbors = full_mask ^ candidate_mask
        for mask in i4_masks:
            if (mask & non_neighbors) == mask:
                return False
        return True

    # Greedy addition loop
    for add_round in range(10): # Try to add up to 10 more vertices
        if _elapsed() > DEADLINE - 20:
            break
        
        found = False
        # Try many random patterns
        # We can bias the density. Paley is 18/37 ~ 0.48.
        # Let's try densities around 0.5.
        
        # To speed up, we can also try "structured" patterns based on existing vertices.
        # But random with high count is often effective for this size.
        
        # We will try 1000 random patterns per vertex addition.
        # Each check is O(|K4| + |I4|). |K4| is ~500-1000. 1000 * 2000 ops = 2M ops. Fast.
        
        for trial in range(2000):
            if _elapsed() > DEADLINE - 20:
                break
            
            # Generate random candidate
            # Using random bits
            cand = 0
            # We can use a random integer and mask it
            # But we want to control density roughly.
            # Let's just pick a random integer from 0 to 2^n - 1
            # This gives uniform distribution over all subsets.
            # For n=37, 2^37 is large, but random.getrandbits(n) is fast.
            
            # To improve success rate, we can try to mimic the density of the graph.
            # But uniform random subsets often work well enough for Ramsey graphs.
            
            cand = random.getrandbits(n)
            
            if is_valid_candidate(cand, n, k4_masks, i4_masks):
                # Valid! Add the vertex.
                # Update adj matrix
                new_n = n + 1
                new_adj = [row[:] for row in adj]
                new_adj.append([0] * new_n)
                for i in range(n):
                    if cand & (1 << i):
                        new_adj[i].append(1)
                        new_adj[new_n-1][i] = 1
                    else:
                        new_adj[i].append(0)
                        new_adj[new_n-1][i] = 0
                new_adj[new_n-1][new_n-1] = 0
                
                # Update edges
                for i in range(n):
                    if cand & (1 << i):
                        edges.append((i, new_n-1))
                
                # Update state
                adj = new_adj
                n = new_n
                neigh = _build_bitsets(adj, n)
                
                # Recompute K4 and I4 lists
                # This is the expensive part, but n is small (<=42).
                # Number of 4-sets in 42 vertices is 42C4 = 111930.
                # Checking each one is O(1) or O(4).
                # 111930 * 4 ~ 450k ops. Very fast.
                k4_masks = _find_all_k_sets(neigh, n, 4, is_clique=True)
                i4_masks = _find_all_k_sets(neigh, n, 4, is_clique=False)
                
                best_n = n
                best_adj = [row[:] for row in adj]
                best_edges = edges[:]
                
                found = True
                break
        
        if not found:
            # If we couldn't add a vertex with random patterns, we might try other strategies
            # But for now, break to save time.
            # We could try a more directed search, but random is usually sufficient for this gap.
            break
            
    return best_n, best_edges

# Entry point
# The construct() function above is the entry point
