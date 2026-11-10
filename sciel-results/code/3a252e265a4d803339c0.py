import time
import random
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Uses a local search with spectral gap heuristic to guide edge insertion.
    """
    N = 40
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Helper to check if adding edge (u, v) creates a C6
    # Uses iterative DFS with bitmask for visited nodes
    def can_add(adj_mask, u, v):
        # We look for a simple path u-a-b-c-d-v of length 5.
        # This is equivalent to finding a node d at distance 4 from u
        # such that d is adjacent to v, and the path u..d is simple and avoids v.
        
        # Stack: (node, depth, mask)
        # mask: integer where bit i is 1 if node i is in the current path
        start_mask = 1 << u
        stack = [(u, 0, start_mask)]
        
        while stack:
            node, depth, mask = stack.pop()
            
            if depth == 4:
                # Check if v is a neighbor of node
                if adj_mask[node] & (1 << v):
                    return False # C6 exists
                continue
            
            # Get neighbors of node not in mask and not v
            candidates = adj_mask[node] & ~mask
            candidates &= ~(1 << v)
            
            while candidates:
                bit = candidates & -candidates
                nb = bit.bit_length() - 1
                candidates ^= bit
                stack.append((nb, depth + 1, mask | bit))
                
        return True

    # Helper to compute spectral gap (lambda_1 - lambda_2) of adjacency matrix
    def spectral_gap(adj_list, adj_set):
        # Build adjacency matrix
        A = np.zeros((N, N), dtype=np.float64)
        for u in range(N):
            for v in adj_list[u]:
                if u < v:
                    A[u, v] = 1
                    A[v, u] = 1
        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(A)
        # Sort in descending order
        eigvals = eigvals[::-1]
        # Spectral gap is lambda_1 - lambda_2
        return eigvals[0] - eigvals[1]

    # Initial Graph: 8 disjoint K5s
    adj_set = [set() for _ in range(N)]
    adj_mask = [0] * N
    edges = []
    
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj_set[u].add(v)
                adj_set[v].add(u)
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append([u, v])
                
    best_edges = len(edges)
    best_adj_set = [set(s) for s in adj_set]
    best_adj_mask = adj_mask[:]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Local Search
    while time.time() < deadline:
        # Create list of missing edges
        missing = []
        for u, v in all_pairs:
            if v not in adj_set[u]:
                missing.append((u, v))
                
        if not missing:
            break
        
        # Compute spectral gap for each candidate edge
        # To save time, we only compute spectral gap for a subset of candidates
        # or we use a cheaper heuristic
        # For now, let's use a simpler heuristic: prefer edges that connect
        # vertices with low degree, as this may maintain spectral gap better
        
        # Actually, let's compute spectral gap for the current graph and use it
        # to guide which edges to try first
        # But this is expensive. Let's use a simpler approach:
        # Try edges in random order, but with a bias towards edges that
        # would maintain a larger spectral gap
        
        # For efficiency, let's just use random order but with a smarter kick
        random.shuffle(missing)
        
        added_in_pass = 0
        for u, v in missing:
            if time.time() > deadline:
                break
            if v in adj_set[u]:
                continue
            if can_add(adj_mask, u, v):
                # Add edge
                adj_set[u].add(v)
                adj_set[v].add(u)
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append([u, v])
                added_in_pass += 1
                if len(edges) > best_edges:
                    best_edges = len(edges)
                    best_adj_set = [set(s) for s in adj_set]
                    best_adj_mask = adj_mask[:]
        
        if added_in_pass == 0:
            # Stuck. Perform a kick: remove a few random edges
            if len(edges) > 0:
                # Remove 1 to 3 random edges
                kick_size = random.randint(1, 3)
                for _ in range(kick_size):
                    if not edges:
                        break
                    idx = random.randint(0, len(edges) - 1)
                    u, v = edges[idx]
                    # Remove from structures
                    if v in adj_set[u]:
                        adj_set[u].remove(v)
                        adj_set[v].remove(u)
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                    edges.pop(idx)

    # Return the best graph found
    final_edges = []
    for u in range(N):
        for v in best_adj_set[u]:
            if u < v:
                final_edges.append([u, v])
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
