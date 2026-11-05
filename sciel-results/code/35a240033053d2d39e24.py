import time
import random
from itertools import combinations

def construct():
    """
    Construct a graph on n vertices with no regular induced subgraph of size >= 6.
    Maximizes n using backtracking with forward checking.
    """
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def _is_regular_mask(adj_mask, mask, n):
        """Check if induced subgraph on mask is regular. Returns True if regular."""
        verts = []
        m = mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            verts.append(idx)
            m ^= lb
        k = len(verts)
        if k < 6:
            return False
        if k == 0:
            return False
        v0 = verts[0]
        deg0 = (adj_mask[v0] & mask).bit_count()
        for v in verts[1:]:
            deg = (adj_mask[v] & mask).bit_count()
            if deg != deg0:
                return False
        return True
    
    def _has_regular_subgraph_fast(adj_mask, n, min_size=6):
        """Check if any subset of size >= min_size induces a regular subgraph."""
        # Check size 6 first (most common violation)
        for comb in combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular_mask(adj_mask, mask, n):
                return True
        # Check size 7
        if n >= 7:
            for comb in combinations(range(n), 7):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if _is_regular_mask(adj_mask, mask, n):
                    return True
        # Check larger sizes only if necessary (expensive)
        for size in range(8, n + 1):
            for comb in combinations(range(n), size):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if _is_regular_mask(adj_mask, mask, n):
                    return True
        return False
    
    def _backtrack_construct(n, seed=42):
        """
        Backtracking construction: build edges (i,j) for i<j sequentially.
        At each step, try adding or not adding the edge.
        Prune if partial assignment creates a regular induced subgraph.
        """
        rng = random.Random(seed)
        
        # Edge list in order: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)
        edges_to_decide = []
        for i in range(n):
            for j in range(i + 1, n):
                edges_to_decide.append((i, j))
        
        adj_mask = [0] * n
        edge_values = [None] * len(edges_to_decide)  # None=undecided, 0=no edge, 1=edge
        
        # Precompute for each edge index, which vertices are involved
        # and what the "active set" looks like when this edge is decided
        
        def _check_partial_validity(adj_mask, n, decided_count):
            """
            Check if the current partial graph (with only decided edges) 
            has a regular induced subgraph of size >= 6.
            We only check subsets where ALL edges within the subset are decided.
            """
            # For efficiency, only check subsets of size 6 and 7
            # A subset is "fully decided" if all pairs within it have been decided
            # Edge (i,j) is decided if max(i,j) < some threshold... actually it's based on order
            # Edge (i,j) is at index: sum of edges before it
            # Let's just check all size-6 subsets and verify all their internal edges are decided
            
            # Precompute: for a given subset, is it fully decided?
            # Edge (i,j) is decided if it appears in edges_to_decide[:decided_count]
            # The index of edge (i,j) in edges_to_decide:
            # edges are ordered: (0,1),(0,2),...,(0,n-1),(1,2),(1,3),...
            # Index of (i,j) where i<j: i*(2*n - i - 3)//2 + (j - i - 1)
            # Let me compute: for vertex i, edges to i+1,...,n-1 are (n-1-i) edges
            # Before vertex i's edges: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
            # Within vertex i's edges, edge to j (j>i) is at position (j - i - 1)
            
            def edge_index(i, j):
                # i < j
                # Number of edges with first vertex < i:
                # For vertex k < i, edges to k+1,...,n-1: (n-1-k) edges
                # Sum = sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
                before = i * (n - 1) - i * (i - 1) // 2
                # Within vertex i, edge to j is at position (j - i - 1)
                return before + (j - i - 1)
            
            # Check all size-6 subsets
            for comb in combinations(range(n), 6):
                # Check if all edges within this subset are decided
                fully_decided = True
                for a in range(6):
                    for b in range(a + 1, 6):
                        idx = edge_index(comb[a], comb[b])
                        if idx >= decided_count:
                            fully_decided = False
                            break
                    if not fully_decided:
                        break
                if not fully_decided:
                    continue
                # All edges in this subset are decided, check regularity
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if _is_regular_mask(adj_mask, mask, n):
                    return False, comb
            return True, None
        
        # Backtracking
        def backtrack(pos):
            if time.time() > deadline - 5:
                return False
            
            if pos == len(edges_to_decide):
                # All edges decided, do a full check
                if _has_regular_subgraph_fast(adj_mask, n):
                    return False
                return True
            
            i, j = edges_to_decide[pos]
            
            # Try adding the edge
            adj_mask[i] |= (1 << j)
            adj_mask[j] |= (1 << i)
            edge_values[pos] = 1
            
            valid, _ = _check_partial_validity(adj_mask, n, pos + 1)
            if valid:
                if backtrack(pos + 1):
                    return True
            
            # Undo
            adj_mask[i] &= ~(1 << j)
            adj_mask[j] &= ~(1 << i)
            edge_values[pos] = 0
            
            # Try not adding the edge
            valid, _ = _check_partial_validity(adj_mask, n, pos + 1)
            if valid:
                if backtrack(pos + 1):
                    return True
            
            edge_values[pos] = None
            return False
        
        # Use randomized backtracking: shuffle edge order? No, keep systematic but add randomness in choice
        # Actually, let's use a hybrid: try both choices with some randomization
        
        def backtrack_random(pos):
            if time.time() > deadline - 5:
                return False
            
            if pos == len(edges_to_decide):
                if _has_regular_subgraph_fast(adj_mask, n):
                    return False
                return True
            
            i, j = edges_to_decide[pos]
            
            # Randomly choose which to try first
            try_add_first = rng.random() < 0.5
            
            choices = []
            if try_add_first:
                choices = [1, 0]  # Try add first, then not-add
            else:
                choices = [0, 1]
            
            for val in choices:
                if val == 1:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
                    edge_values[pos] = 1
                else:
                    adj_mask[i] &= ~(1 << j)
                    adj_mask[j] &= ~(1 << i)
                    edge_values[pos] = 0
                
                valid, _ = _check_partial_validity(adj_mask, n, pos + 1)
                if valid:
                    if backtrack_random(pos + 1):
                        return True
                
                # Undo
                if val == 1:
                    adj_mask[i] &= ~(1 << j)
                    adj_mask[j] &= ~(1 << i)
                else:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
            
            edge_values[pos] = None
            return False
        
        success = backtrack_random(0)
        if success:
            edges = []
            for idx, (i, j) in enumerate(edges_to_decide):
                if edge_values[idx] == 1:
                    edges.append([i, j])
            return n, edges
        return None, None
    
    # Try n=21 first
    for n_target in [21, 20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 15:
            break
        
        # Try multiple seeds
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            n_found, edges = _backtrack_construct(n_target, seed)
            if n_found is not None:
                return {'n': n_found, 'edges': edges}
    
    # Fallback: random local search (from current program)
    def _local_search(adj_mask, n, max_steps=3000, seed=42):
        rng = random.Random(seed)
        adj_mask = adj_mask[:]
        
        def _find_violation(adj_mask, n):
            for comb in combinations(range(n), 6):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if _is_regular_mask(adj_mask, mask, n):
                    return mask, comb
            for comb in combinations(range(n), 7):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if _is_regular_mask(adj_mask, mask, n):
                    return mask, comb
            return 0, []
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            mask, verts = _find_violation(adj_mask, n)
            if mask == 0:
                # Full check
                if _has_regular_subgraph_fast(adj_mask, n):
                    break
                return adj_mask, True
            
            # Flip a random edge within the violating subset
            if len(verts) >= 2:
                u, v = rng.sample(verts, 2)
                if adj_mask[u] & (1 << v):
                    adj_mask[u] ^= (1 << v)
                    adj_mask[v] ^= (1 << u)
                else:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        
        return adj_mask, False
    
    # Random init + local search fallback
    for n_target in [20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 10:
            break
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            rng = random.Random(seed)
            adj_mask = [0] * n_target
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if rng.random() < 0.5:
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
            adj_mask, valid = _local_search(adj_mask, n_target, max_steps=1000, seed=seed)
            if valid:
                edges = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if adj_mask[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n_target, 'edges': edges}
    
    # Last resort: n=5
    return {'n': 5, 'edges': []}
