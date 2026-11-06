import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_result = {'n': 0, 'edges': []}
    
    # Try descending n values
    # Exoo's record is 35. We aim for 35, then 34, etc.
    for n in range(35, 19, -1):
        if time.time() > deadline:
            break
        
        # Try multiple restarts for this n
        # We don't know how many restarts are needed, so we loop until time runs out
        # or we find a solution.
        found = False
        while time.time() < deadline:
            res = local_search(n, deadline)
            if res:
                best_result = res
                found = True
                break
        
        if found:
            return best_result

    # Fallback: Return a small valid graph if nothing better found
    # A single vertex is always valid.
    if best_result['n'] == 0:
        best_result = {'n': 1, 'edges': []}
        
    return best_result

def local_search(n, deadline):
    """
    Attempts to find a K4-free graph with alpha <= 5 on n vertices.
    """
    # Initialize random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Precompute complement masks
    # comp[i] has bit j set if (i,j) is NOT an edge and i != j
    all_mask = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = all_mask ^ (1 << i) ^ adj[i]
    
    # Main local search loop
    max_iters = 2000
    for it in range(max_iters):
        if time.time() > deadline:
            return None
        
        # 1. Check for K4
        k4 = find_k4(adj, n)
        if k4:
            # Remove a random edge from the K4
            # Edges in K4: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            u, v = random.choice([(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]), 
                                  (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])])
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            comp[u] ^= (1 << v)
            comp[v] ^= (1 << u)
            continue
        
        # 2. Check for Independent Set of size 6
        ind6 = find_ind_6(comp, n)
        if ind6:
            # Add a random edge to the independent set to break it
            # Pairs in 6-set: 15 edges
            pairs = [
                (ind6[0], ind6[1]), (ind6[0], ind6[2]), (ind6[0], ind6[3]), (ind6[0], ind6[4]), (ind6[0], ind6[5]),
                (ind6[1], ind6[2]), (ind6[1], ind6[3]), (ind6[1], ind6[4]), (ind6[1], ind6[5]),
                (ind6[2], ind6[3]), (ind6[2], ind6[4]), (ind6[2], ind6[5]),
                (ind6[3], ind6[4]), (ind6[3], ind6[5]),
                (ind6[4], ind6[5])
            ]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            comp[u] ^= (1 << v)
            comp[v] ^= (1 << u)
            continue
        
        # 3. No violations found, graph is valid
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return {'n': n, 'edges': edges}
    
    return None

def find_k4(adj, n):
    """
    Finds a K4 in the graph. Returns tuple of 4 vertices or None.
    """
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                for k in range(j+1, n):
                    if (adj[i] & (1 << k)) and (adj[j] & (1 << k)):
                        # i, j, k form a triangle
                        # Check for a fourth vertex l connected to all three
                        common = adj[i] & adj[j] & adj[k]
                        # Mask out i, j, k to be safe (though they shouldn't be in adj[i] etc)
                        # Actually, adj[i] does not have bit i.
                        # So common contains vertices l != i, j, k that are connected to i, j, k.
                        if common:
                            l = (common & -common).bit_length() - 1
                            return (i, j, k, l)
    return None

def find_ind_6(comp, n):
    """
    Finds an independent set of size 6 (clique of size 6 in complement).
    Uses backtracking with heuristic vertex ordering.
    Returns tuple of 6 vertices or None.
    """
    def backtrack(cand, size, current):
        if size == 0:
            return tuple(current)
        
        # Pruning: not enough candidates
        if cand.bit_count() < size:
            return None
        
        # Heuristic: Order candidates by degree in the induced subgraph (comp)
        # Higher degree in comp means more connections to other candidates, 
        # which is good for finding a clique.
        verts = []
        c = cand
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            # Degree of v in the candidate set within the complement graph
            deg = (cand & comp[v]).bit_count()
            verts.append((deg, v))
        
        # Sort by degree descending
        verts.sort(reverse=True)
        
        for deg, v in verts:
            # Try adding v to the clique
            # New candidates must be neighbors of v in comp
            new_cand = (cand ^ (1 << v)) & comp[v]
            res = backtrack(new_cand, size - 1, current + [v])
            if res:
                return res
        
        return None
    
    # Initial candidates: all vertices
    return backtrack((1 << n) - 1, 6, [])

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Found graph with n={res['n']}, edges={len(res['edges'])}")
