import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Order vertices by degree descending
    # Higher degree vertices are more constrained
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # Map original vertex to index in order
    # Actually, we just iterate through 'order'
    
    # State for parts:
    # Each part is a dict: {'mask': int, 'is_clique': bool, 'is_indep': bool, 'vertices': list}
    # Using lists for vertices helps in reconstruction, but mask is faster for checks.
    # We can reconstruct vertices from mask at the end.
    
    best_parts = None
    best_count = n + 1
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin for 60s limit
    
    # To speed up, we can pre-calculate if a set of vertices is a clique/indep?
    # No, dynamic check is fast enough with bitmasks.
    
    def dfs(idx, parts):
        nonlocal best_parts, best_count
        
        # Time check
        if time.time() > deadline:
            return
        
        if len(parts) >= best_count:
            return
            
        if idx == n:
            if len(parts) < best_count:
                best_count = len(parts)
                # Convert masks to lists of original vertex indices
                # The vertices in 'order' are the original indices
                # We need to map mask bits back to original indices
                # The bit position in the mask corresponds to the original vertex index
                new_parts = []
                for p in parts:
                    mask = p['mask']
                    verts = [i for i in range(n) if (mask & (1 << i))]
                    new_parts.append(verts)
                best_parts = new_parts
            return
        
        v = order[idx]
        v_bit = 1 << v
        
        # Try adding v to existing parts
        for i in range(len(parts)):
            p = parts[i]
            mask = p['mask']
            is_c = p['is_clique']
            is_i = p['is_indep']
            
            new_mask = mask | v_bit
            
            # Check clique constraint
            new_is_c = False
            if is_c:
                # v must be adjacent to all in mask
                if (adj[v] & mask) == mask:
                    new_is_c = True
                    
            # Check independent set constraint
            new_is_i = False
            if is_i:
                # v must not be adjacent to any in mask
                if (adj[v] & mask) == 0:
                    new_is_i = True
            
            if new_is_c or new_is_i:
                # Valid addition
                parts[i] = {'mask': new_mask, 'is_clique': new_is_c, 'is_indep': new_is_i}
                dfs(idx + 1, parts)
                # Backtrack
                parts[i] = p
        
        # Try creating a new part
        # Symmetry breaking: 
        # We only create a new part if we haven't created a new part for the previous vertex 
        # that is "equivalent"? 
        # Actually, a simpler symmetry breaking: 
        # If we create a new part for v, it becomes the last part.
        # For the next vertex, we can add to this new part or others.
        # We don't need to explicitly break symmetry for "new part" vs "new part" 
        # because the order of creation is fixed by the DFS path.
        # However, if we have two empty parts (which we don't, parts are non-empty), 
        # symmetry is an issue.
        # The main symmetry is: Part A and Part B are indistinguishable if they have same type and size?
        # With bitmasks, the content distinguishes them.
        # The only symmetry is that the order of parts in the list doesn't matter.
        # By always appending new parts to the end, and iterating existing parts in order,
        # we generate each partition exactly once (up to the order of parts in the list, 
        # which is canonical by creation order).
        
        # Check if we should prune based on lower bound?
        # Remaining vertices: n - idx - 1
        # Current parts: len(parts)
        # If len(parts) + 1 >= best_count, we still might find a solution with len(parts)+1 
        # if best_count is larger, but we prune if len(parts) >= best_count.
        # So creating a new part leads to len(parts)+1.
        # If len(parts)+1 >= best_count, we can prune this branch?
        # Yes, if len(parts) + 1 >= best_count, any solution extending this will have 
        # at least len(parts)+1 parts, which is not better than best_count.
        
        if len(parts) + 1 < best_count:
            new_part = {'mask': v_bit, 'is_clique': True, 'is_indep': True}
            parts.append(new_part)
            dfs(idx + 1, parts)
            parts.pop()
            
    dfs(0, [])
    
    if best_parts is None:
        # Fallback to greedy or single vertex parts if search fails (should not happen)
        return [[i] for i in range(n)]
        
    # Ensure types are correct
    # best_parts is list of lists of ints
    # Check for numpy types? The problem says convert. 
    # Our construction uses Python ints.
    
    return best_parts

# EVOLVE-BLOCK-END
