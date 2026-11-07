import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    deadline = time.time() + 85.0
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Pre-compute edge indices and adjacency for fast access
    edge_idx = [(u, v) for u, v in edges]
    
    # Build incidence matrix for GF(2) cycle space check
    # We need a basis for the cycle space to check if a sign vector is in the cut space.
    # A sign vector x (0 for +1, 1 for -1) corresponds to a disconnected lift iff x is in the cut space.
    # Cut space is the orthogonal complement of the cycle space.
    # So x is in cut space iff x . c == 0 for all c in cycle space basis.
    
    # Let's find a basis for the cycle space.
    # The graph has n vertices and E = 3m edges.
    # Dimension of cycle space = E - n + 1 (assuming connected, which prism graphs are).
    # dim = 3m - 2m + 1 = m + 1.
    
    # We can find a spanning tree and use the non-tree edges to form fundamental cycles.
    # Let's just do a simple BFS to find a spanning tree and then the fundamental cycles.
    
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edge_idx):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Find spanning tree
    parent = [-1] * n
    parent_edge = [-1] * n
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    tree_edges = set()
    
    while queue:
        u = queue.popleft()
        for v, ei in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                parent_edge[v] = ei
                tree_edges.add(ei)
                queue.append(v)
    
    # Fundamental cycles are formed by each non-tree edge (u,v) + path in tree from u to v.
    # Let's construct the cycle space basis matrix B (dim x E) over GF(2).
    # Rows of B are the cycles.
    # We need to check if x is orthogonal to all rows of B.
    
    non_tree_edges = [i for i in range(num_edges) if i not in tree_edges]
    num_cycles = len(non_tree_edges)
    
    # B is a list of bitmasks or lists of 0/1.
    # Since E is small, we can use integers as bitmasks.
    cycle_masks = []
    
    for ei in non_tree_edges:
        u, v = edge_idx[ei]
        # Find path from u to v in tree
        # We can trace parents up to LCA, but simpler: just trace from u to root and v to root.
        # Or even simpler: since it's a tree, just trace from u to root, then v to root.
        
        # Let's just do a BFS on the tree from u to v.
        # Actually, since we have parent pointers, we can lift.
        # But n is small. Let's just collect the path.
        
        # Path from u to root
        path_u = []
        curr = u
        while curr != -1:
            path_u.append(curr)
            curr = parent[curr]
        
        # Path from v to root
        path_v = []
        curr = v
        while curr != -1:
            path_v.append(curr)
            curr = parent[curr]
        
        # Find LCA
        # Convert to sets for intersection? Or just walk.
        # Since it's a tree, the path is unique.
        # The cycle consists of the non-tree edge ei, and the tree edges on the path u-v.
        
        # Let's find the tree edges on the path u-v.
        # We can mark nodes in path_u.
        path_u_set = set(path_u)
        
        # Walk from v up until we hit a node in path_u_set
        path_v_edges = []
        curr = v
        while curr not in path_u_set:
            path_v_edges.append(parent_edge[curr])
            curr = parent[curr]
        
        # Now curr is the LCA (or on the path to LCA).
        # Walk from u up to curr.
        path_u_edges = []
        curr = u
        while curr != -1 and curr != path_u_set.__iter__().__next__() if False else True: # Just walk up
            # Actually, we need to stop when we reach the common ancestor.
            # Let's just collect edges until we hit a node that is in path_v?
            # No, let's just collect all edges from u to root and v to root, then remove common part.
            pass
        
        # Simpler approach:
        # Get list of edges on path u->root and v->root.
        edges_u = []
        curr = u
        while curr != -1:
            edges_u.append(parent_edge[curr])
            curr = parent[curr]
            
        edges_v = []
        curr = v
        while curr != -1:
            edges_v.append(parent_edge[curr])
            curr = parent[curr]
            
        # The path u-v in the tree is the symmetric difference of the paths u-root and v-root?
        # No, it's the union of the two paths minus the common part (root to LCA).
        # Actually, the set of edges in the path u-v is {e | e in path_u_root XOR e in path_v_root} is wrong.
        # It is (path_u_root U path_v_root) \ (path_lca_root U path_lca_root).
        # Basically, the edges that are in one path but not the other? No.
        
        # Let's just do a BFS on the tree to find the path u-v.
        # Tree is small.
        
        def find_path_tree(start, end):
            if start == end:
                return []
            parent_local = {start: -1}
            edge_local = {start: -1}
            q = deque([start])
            while q:
                node = q.popleft()
                if node == end:
                    break
                for nei, ei in adj[node]:
                    if ei in tree_edges: # Only traverse tree edges
                        if nei not in parent_local:
                            parent_local[nei] = node
                            edge_local[nei] = ei
                            q.append(nei)
            # Reconstruct
            path_edges = []
            curr = end
            while curr != start:
                path_edges.append(edge_local[curr])
                curr = parent_local[curr]
            return path_edges
        
        tree_path_edges = find_path_tree(u, v)
        
        # The cycle is tree_path_edges + [ei]
        mask = 1 << ei
        for te in tree_path_edges:
            mask ^= (1 << te)
            
        cycle_masks.append(mask)
    
    # Now we have cycle_masks.
    # x is in cut space iff (x & mask) has even parity for all masks in cycle_masks.
    # Parity of (x & mask) is the number of 1s in the bitwise AND.
    
    def is_disconnected(x_mask):
        for mask in cycle_masks:
            # Count bits in x_mask & mask
            if bin(x_mask & mask).count('1') % 2 == 0:
                pass # Wait, if ALL are even, it's in cut space.
            else:
                # If ANY is odd, it's NOT in cut space, so it IS connected.
                return False
        return True
    
    # Optimization: Precompute parity?
    # Since m is small, we can just use this.
    # Even better: combine all cycle masks into a single check?
    # No, they are independent constraints.
    
    # Actually, we can just check if x is in the row space of the incidence matrix.
    # But the cycle mask check is fine.
    
    # Let's refine is_disconnected for speed.
    # If num_cycles is small, this is fast.
    
    def compute_radius_from_mask(x_mask):
        signs = np.ones(num_edges, dtype=np.int8)
        for i in range(num_edges):
            if (x_mask >> i) & 1:
                signs[i] = -1
        
        S = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edge_idx):
            s = signs[i]
            S[u, v] = s
            S[v, u] = s
            
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))
    
    def evaluate(x_mask):
        if is_disconnected(x_mask):
            return None
        return compute_radius_from_mask(x_mask)
    
    best_radius = np.inf
    best_mask = None
    
    # Generate initial candidates
    rng = np.random.default_rng(3)
    
    # 1. Try single bit flips (one -1 edge)
    for i in range(num_edges):
        mask = 1 << i
        r = evaluate(mask)
        if r is not None and r < best_radius:
            best_radius = r
            best_mask = mask
            
    # 2. Try random valid masks
    for _ in range(1000):
        # Generate random mask
        mask = int(rng.integers(0, 1 << num_edges))
        r = evaluate(mask)
        if r is not None and r < best_radius:
            best_radius = r
            best_mask = mask
            
    if best_mask is None:
        # Fallback: find any valid mask
        # Try all masks with 2 bits set?
        for i in range(num_edges):
            for j in range(i+1, num_edges):
                mask = (1 << i) | (1 << j)
                r = evaluate(mask)
                if r is not None:
                    best_radius = r
                    best_mask = mask
                    break
            if best_mask is not None:
                break
        
        if best_mask is None:
             # This should not happen for prism graphs with m>=4
             # But just in case, return something
             return np.ones(num_edges, dtype=np.int8)

    current_mask = best_mask
    current_radius = best_radius
    
    # Local Search
    # We flip bits. If the new mask is disconnected, we skip it.
    # To improve, we can try to flip bits that are "likely" to reduce radius.
    # But simple random restarts + local search is robust.
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Start from current best or a random valid neighbor
        if restart == 0:
            start_mask = best_mask
        else:
            # Generate a random valid mask near best? Or just random?
            # Let's try random valid mask.
            found = False
            for _ in range(100):
                # Flip 1-3 random bits from best_mask to diversify
                trial = best_mask
                for _ in range(rng.integers(1, 4)):
                    bit = rng.integers(0, num_edges)
                    trial ^= (1 << bit)
                if not is_disconnected(trial):
                    start_mask = trial
                    found = True
                    break
            if not found:
                start_mask = best_mask
                
        curr_mask = start_mask
        curr_rad = evaluate(curr_mask)
        if curr_rad is None:
            continue
            
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try all single bit flips
            for i in range(num_edges):
                new_mask = curr_mask ^ (1 << i)
                if is_disconnected(new_mask):
                    continue
                
                new_rad = compute_radius_from_mask(new_mask)
                
                if new_rad < curr_rad - 1e-10:
                    curr_mask = new_mask
                    curr_rad = new_rad
                    improved = True
                    
                    if curr_rad < best_radius:
                        best_radius = curr_rad
                        best_mask = curr_mask
                        
                    # Restart the inner loop to explore from the new point
                    # This is "First Improvement" strategy.
                    # Or we can continue to check other bits.
                    # First improvement is usually faster to converge.
                    break
        
        # Optional: 2-flip moves if stuck
        # For now, single flip + restarts is strong.
        
    # Convert best_mask to signs array
    best_signs = np.ones(num_edges, dtype=np.int8)
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            best_signs[i] = -1
            
    return best_signs

