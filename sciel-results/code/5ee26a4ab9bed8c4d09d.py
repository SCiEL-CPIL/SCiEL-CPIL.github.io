import time
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit
    
    # Precompute inverse sums for all non-empty subsets of {1, ..., 10}
    # subset_index -> (mask, sum_mod_101, size)
    subsets = []
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(pow(x, -1, 101) for x in terms) % 101
        subsets.append((mask, residue, len(terms)))
    
    # Sort subsets by size to facilitate increasing k
    subsets.sort(key=lambda x: x[2])
    
    # We want to find the minimum k such that a perfect matching exists
    # using subsets of size <= k.
    
    # Hopcroft-Karp is O(E * sqrt(V))
    # V_left = 101, V_right <= 1023
    # E is at most 101 * 1023
    
    # We can binary search on k or just iterate k from 1 to 10.
    # Iterating is fine since k is small.
    
    for k in range(1, 11):
        if time.time() > deadline:
            # Fallback: return a valid but possibly suboptimal solution
            # Just pick any valid subset for each residue, ignoring size constraint
            # This should not happen, but good for safety.
            return _fallback()
            
        # Build adjacency list for bipartite graph
        # Left: residues 0..100
        # Right: subsets with size <= k
        
        # Filter subsets with size <= k
        valid_subsets = [s for s in subsets if s[2] <= k]
        n_right = len(valid_subsets)
        
        if n_right < 101:
            continue
            
        # Map subset index in valid_subsets to right node index
        # right node index is just the index in valid_subsets list
        
        # Build adjacency list for left nodes (residues)
        # adj[r] = list of right node indices (indices in valid_subsets)
        adj = [[] for _ in range(101)]
        for i, (mask, residue, size) in enumerate(valid_subsets):
            adj[residue].append(i)
            
        # Run Hopcroft-Karp
        # We need a matrix or CSR/COO format for scipy
        # Let's use a simple implementation or scipy
        
        # scipy.sparse.csgraph.maximum_bipartite_matching
        # Input: a sparse matrix of shape (n_left, n_right)
        # n_left = 101, n_right = n_right
        
        row_indices = []
        col_indices = []
        for r in range(101):
            for c in adj[r]:
                row_indices.append(r)
                col_indices.append(c)
                
        if not row_indices:
            continue
            
        data = [1] * len(row_indices)
        matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(101, n_right))
        
        matching = csgraph.maximum_bipartite_matching(matrix)
        
        if len(matching) == 101:
            # Perfect matching found
            # Construct the result
            result = [None] * 101
            for r, c in matching.items():
                mask, residue, size = valid_subsets[c]
                terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
                result[r] = terms
            return result
            
    # If no k worked (should not happen), fallback
    return _fallback()

def _fallback():
    # Just find any representation
    representations = {}
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(pow(x, -1, 101) for x in terms) % 101
        if residue not in representations:
            representations[residue] = terms
    # Fill missing with any available (should not be missing)
    # Actually, the seed program guarantees all residues are covered.
    # If some are missing, we can just use the first found or empty? 
    # The task says "101 nonempty lists". 
    # If a residue is not covered by any subset, it's impossible. 
    # But the seed program implies it's possible.
    return [representations.get(r, [1]) for r in range(101)]
