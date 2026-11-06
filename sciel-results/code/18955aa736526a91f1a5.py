import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    best_edges = 0
    best_graph = None
    
    def popcount(x):
        return bin(x).count('1')
        
    def has_k33(neighbors):
        # Check if any triple has >= 3 common neighbors
        for i in range(n):
            for j in range(i+1, n):
                common_ij = neighbors[i] & neighbors[j]
                if popcount(common_ij) < 2:
                    continue
                for k in range(j+1, n):
                    common_ijk = common_ij & neighbors[k]
                    if popcount(common_ijk) >= 3:
                        return True, (i, j, k)
        return False, None
        
    def find_blocking_k33s(neighbors, u, v):
        """
        Returns a list of sets of edges. Each set is a K3,3 that would be formed by adding (u,v).
        We return at most a few to keep it fast.
        """
        blockers = []
        Cv = neighbors[v] & ~(1 << u)
        Cu = neighbors[u] & ~(1 << v)
        
        # Find C4 in bipartite graph between Cv and Cu
        # Vertices a in Cv, c in Cu.
        # a~c, b~c, a~d, b~d with a!=b in Cv, c!=d in Cu.
        
        # For each pair a, b in Cv, check common neighbors in Cu
        # If common neighbors >= 2, we have a C4 (actually K2,2 subgraph which is C4)
        # The K3,3 parts are {u, a, b} and {v, c, d}.
        
        a_list = []
        mask = Cv
        while mask:
            a = (mask & -mask).bit_length() - 1
            a_list.append(a)
            mask &= mask - 1
            
        for i in range(len(a_list)):
            for j in range(i+1, len(a_list)):
                a = a_list[i]
                b = a_list[j]
                # Common neighbors of a and b in Cu
                common = neighbors[a] & neighbors[b] & Cu
                if popcount(common) >= 2:
                    # Pick two distinct c, d from common
                    verts = []
                    cmask = common
                    while cmask:
                        c = (cmask & -cmask).bit_length() - 1
                        verts.append(c)
                        if len(verts) == 2:
                            break
                        cmask &= cmask - 1
                    if len(verts) == 2:
                        c, d = verts
                        # The K3,3 edges are:
                        # (u,v) - the new edge
                        # (u,c), (u,d) - in Cu, so exist
                        # (v,a), (v,b) - in Cv, so exist
                        # (a,c), (a,d) - a~c, a~d exist
                        # (b,c), (b,d) - b~c, b~d exist
                        # The existing edges in this K3,3 (excluding u,v) are:
                        # (u,c), (u,d), (v,a), (v,b), (a,c), (a,d), (b,c), (b,d)
                        edge_set = {
                            (min(u,c), max(u,c)), (min(u,d), max(u,d)),
                            (min(v,a), max(v,a)), (min(v,b), max(v,b)),
                            (min(a,c), max(a,c)), (min(a,d), max(a,d)),
                            (min(b,c), max(b,c)), (min(b,d), max(b,d))
                        }
                        blockers.append(edge_set)
                        if len(blockers) >= 5: # Limit to avoid overhead
                            return blockers
        return blockers

    def can_add_edge(neighbors, u, v):
        # Check if adding (u, v) creates a K3,3
        if neighbors[u] & (1 << v):
            return True # Already exists
        
        Cv = neighbors[v] & ~(1 << u)
        Cu = neighbors[u] & ~(1 << v)
        
        # Check for C4 in bipartite graph between Cv and Cu
        a_mask = Cv
        while a_mask:
            a = (a_mask & -a_mask).bit_length() - 1
            a_mask &= a_mask - 1
            
            c_mask = Cu & neighbors[a]
            while c_mask:
                c = (c_mask & -c_mask).bit_length() - 1
                c_mask &= c_mask - 1
                
                # We have a~c.
                # Need b in Cv, b != a, b~c
                # Need d in Cu, d != c, d~a
                # And b~d
                
                b_cands = (Cv & ~(1 << a)) & neighbors[c]
                d_cands = (Cu & ~(1 << c)) & neighbors[a]
                
                b_mask = b_cands
                while b_mask:
                    b = (b_mask & -b_mask).bit_length() - 1
                    b_mask &= b_mask - 1
                    if neighbors[b] & d_cands:
                        return False
        
        return True

    def add_edge(neighbors, u, v):
        neighbors[u] |= (1 << v)
        neighbors[v] |= (1 << u)
        
    def remove_edge(neighbors, u, v):
        neighbors[u] &= ~(1 << v)
        neighbors[v] &= ~(1 << u)

    for restart in range(500):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        neighbors = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    neighbors[i] |= (1 << j)
                    neighbors[j] |= (1 << i)
        
        # Repair phase
        changed = True
        while changed:
            if time.time() > deadline:
                break
            has, triple = has_k33(neighbors)
            if not has:
                break
            # Remove a random edge from the K3,3
            i, j, k = triple
            common = neighbors[i] & neighbors[j] & neighbors[k]
            verts = []
            mask = common
            while mask:
                v = (mask & -mask).bit_length() - 1
                verts.append(v)
                if len(verts) == 3:
                    break
                mask &= mask - 1
            # Remove edge (i, verts[0])
            remove_edge(neighbors, i, verts[0])
            changed = True
        # Ensure changed is False if we broke due to time or no K33
        if time.time() > deadline:
            break
        if not has_k33(neighbors)[0]:
            changed = False
        else:
            # If still has K33, something went wrong or loop exited early
            # Force break to avoid infinite loop
            break

        # Optimization Phase 1: Greedy Add
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (neighbors[u] & (1 << v)):
                        if can_add_edge(neighbors, u, v):
                            add_edge(neighbors, u, v)
                            improved = True
                            if time.time() > deadline:
                                break
                if time.time() > deadline:
                    break

        # Optimization Phase 2: 2-Swap Search
        # Try to swap edges to escape local optima
        # For each non-edge, if blocked, try to remove an edge from a blocker and add it.
        swapped = True
        while swapped:
            if time.time() > deadline:
                break
            swapped = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (neighbors[u] & (1 << v)):
                        if not can_add_edge(neighbors, u, v):
                            # Try to unblock
                            blockers = find_blocking_k33s(neighbors, u, v)
                            for edge_set in blockers:
                                for e in edge_set:
                                    # Try removing e and adding (u,v)
                                    # Make a copy
                                    new_neighbors = neighbors[:]
                                    remove_edge(new_neighbors, e[0], e[1])
                                    if can_add_edge(new_neighbors, u, v):
                                        add_edge(new_neighbors, u, v)
                                        # Check if this move is beneficial
                                        # It maintains edge count, but might allow future additions.
                                        # We accept it to diversify.
                                        neighbors = new_neighbors
                                        swapped = True
                                        break
                                if swapped:
                                    break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
        
        # Count edges
        edges = sum(popcount(neighbors[i]) for i in range(n)) // 2
        
        if edges > best_edges:
            best_edges = edges
            best_graph = [ [i, j] for i in range(n) for j in range(i+1, n) if neighbors[i] & (1 << j) ]
            
    if best_graph is None:
        # Fallback to seed
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_graph
