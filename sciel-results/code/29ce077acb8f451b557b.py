import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    n = 20
    
    # Helper: Check if graph is K3,3-free
    def is_k33_free(adj_bits):
        for i in range(n):
            for j in range(i+1, n):
                ij = adj_bits[i] & adj_bits[j]
                if ij.bit_count() < 3:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (ij & adj_bits[k]).bit_count() >= 3:
                        return False
        return True

    def count_edges(adj_bits):
        total = 0
        for i in range(n):
            total += adj_bits[i].bit_count()
        return total // 2

    def add_edge(adj_bits, u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    def remove_edge(adj_bits, u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    def count_k33s(adj_bits):
        """
        Count the number of K3,3 subgraphs.
        A K3,3 is a set of 6 vertices {a1,a2,a3,b1,b2,b3} such that
        all edges between {a1,a2,a3} and {b1,b2,b3} exist.
        
        We count the number of pairs of disjoint triples (A, B) such that
        N(A) contains B.
        To avoid overcounting due to symmetry (A,B) vs (B,A), we can just count
        all such pairs. Since we only care about relative changes, absolute count
        is fine as long as it's consistent.
        
        Actually, a simpler metric for SA:
        Energy = -100 * edges + count_k33s
        But counting exact K3,3s is O(n^6) if naive.
        
        Optimized counting:
        For each triple A, let C_A be the set of common neighbors.
        The number of triples B in C_A is C(|C_A|, 3).
        Sum this over all A.
        This counts each K3,3 exactly twice (once for A, once for B).
        So total K3,3s = 1/2 * sum_A C(|N(A)|, 3).
        
        Wait, N(A) are vertices connected to all in A.
        If |N(A)| >= 3, any 3 vertices in N(A) form a K3,3 with A.
        So yes, sum_A C(|N(A)|, 3) / 2 is the number of K3,3s.
        
        Complexity: O(n^3 * cost_of_intersection).
        n=20, n^3 = 8000. Very fast.
        """
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = adj_bits[i] & adj_bits[j] & adj_bits[k]
                    cnt = common.bit_count()
                    if cnt >= 3:
                        # Number of ways to choose 3 from cnt
                        total += cnt * (cnt - 1) * (cnt - 2) // 6
        return total // 2

    def energy(adj_bits):
        """
        Objective: Maximize edges, subject to K3,3-free.
        For SA, we need a scalar.
        Let E = count_k33s.
        If E == 0, valid.
        We want to maximize edges.
        Let's define Energy = E * 1000 - edges.
        We minimize Energy.
        """
        e = count_k33s(adj_bits)
        m = count_edges(adj_bits)
        return e * 1000 - m

    # Initialize with a random dense graph? 
    # Or start from a known good structure?
    # Let's start from a random graph with p=0.5 and let SA refine.
    # Or start from complete graph and let SA remove edges?
    # Complete graph has huge E. SA might get stuck removing random edges.
    # Better: Start from a sparse graph and add edges?
    # Or use a hybrid: Start from a random graph, run SA.
    
    # Let's try starting from a random graph with p=0.7
    adj_bits = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.7:
                add_edge(adj_bits, i, j)
                
    best_adj = adj_bits[:]
    best_energy = energy(adj_bits)
    best_edges = count_edges(adj_bits)
    
    current_adj = adj_bits[:]
    current_energy = best_energy
    
    T = 100.0  # Initial temperature
    cooling = 0.999
    
    while time.time() < deadline:
        if T < 0.1:
            break
            
        # Generate a random move: add or remove a random edge
        u, v = random.sample(range(n), 2)
        has_edge = (current_adj[u] >> v) & 1
        
        if has_edge:
            remove_edge(current_adj, u, v)
        else:
            add_edge(current_adj, u, v)
            
        new_energy = energy(current_adj)
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta/T)
        if delta_e < 0 or random.random() < np.exp(-delta_e / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_adj = current_adj[:]
                best_edges = count_edges(current_adj)
        else:
            # Revert
            if has_edge:
                add_edge(current_adj, u, v)
            else:
                remove_edge(current_adj, u, v)
                
        T *= cooling
        
    # If we ended up in an invalid state (E > 0), we need to repair.
    # The best_adj might also be invalid.
    # Let's do a final greedy repair on best_adj if it's invalid.
    final_adj = best_adj[:]
    
    # Check if best_adj is valid
    if count_k33s(final_adj) > 0:
        # Greedy repair: remove edges from most critical K3,3s
        # This is a fallback. Ideally SA finds a valid state.
        for _ in range(1000):
            if count_k33s(final_adj) == 0:
                break
            # Find a K3,3 and remove a random edge from it
            found = False
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        common = final_adj[i] & final_adj[j] & final_adj[k]
                        if common.bit_count() >= 3:
                            # Get 3 vertices from common
                            verts = []
                            mask = common
                            while len(verts) < 3 and mask:
                                lsb = mask & -mask
                                v = lsb.bit_length() - 1
                                verts.append(v)
                                mask ^= lsb
                            # Remove a random edge between {i,j,k} and verts
                            a, b, c = i, j, k
                            v1, v2, v3 = verts
                            # Pick random edge
                            u, v = random.choice([(a,v1), (a,v2), (a,v3), (b,v1), (b,v2), (b,v3), (c,v1), (c,v2), (c,v3)])
                            remove_edge(final_adj, u, v)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break

    # Try to add more edges to final_adj if valid
    if count_k33s(final_adj) == 0:
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (final_adj[i] >> j) & 1]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 2:
                break
            add_edge(final_adj, u, v)
            if count_k33s(final_adj) == 0:
                pass # Keep edge
            else:
                remove_edge(final_adj, u, v)

    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (final_adj[i] >> j) & 1:
                result.append([i, j])
                
    return result

# EVOLVE-BLOCK-END
