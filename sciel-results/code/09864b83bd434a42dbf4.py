import time
import random
import math

def is_k33_free(adj):
    """
    Check if the graph defined by adjacency bitmasks is K3,3-free.
    A K3,3 exists if there are 3 vertices u, v, w and 3 common neighbors.
    Equivalently, for any triple of vertices {u, v, w}, the number of common neighbors must be < 3.
    """
    n = len(adj)
    for i in range(n):
        for j in range(i + 1, n):
            # Intersection of neighbors of i and j
            common_ij = adj[i] & adj[j]
            # If they have fewer than 3 common neighbors, no K3,3 can involve this pair
            # as the '3 common neighbors' side.
            if common_ij.bit_count() < 3:
                continue
            
            # We need to check if there is a third vertex k such that
            # |N(k) ∩ N(i) ∩ N(j)| >= 3.
            # This is equivalent to checking if the intersection of N(i), N(j), and N(k) has size >= 3.
            # Instead of iterating k, we can just check all pairs (i, j) and see if 
            # there are 3 vertices k1, k2, k3 that are all in N(i) ∩ N(j) AND 
            # all connected to each other? No, that's not right.
            
            # Definition of K3,3: Bipartition {a,b,c} and {x,y,z}.
            # Edges exist between every a in {a,b,c} and every x in {x,y,z}.
            # This means |N(a) ∩ N(b) ∩ N(c)| >= 3 is NOT the condition.
            # The condition is: There exist 3 distinct vertices u, v, w such that
            # they have at least 3 common neighbors.
            # Let's re-verify.
            # If u, v, w have common neighbors x, y, z, then we have edges (u,x), (u,y), (u,z), 
            # (v,x), (v,y), (v,z), (w,x), (w,y), (w,z).
            # This forms a K3,3 with parts {u,v,w} and {x,y,z}.
            # So yes, checking if any triple {u,v,w} has >= 3 common neighbors is correct.
            
            # Optimization:
            # If common_ij has bit_count >= 3, we need to see if any vertex k
            # is connected to at least 3 of the vertices in common_ij.
            # Actually, it's simpler:
            # For each k, check if (adj[k] & common_ij).bit_count() >= 3.
            # But we can be smarter.
            
            # Let's iterate k.
            for k in range(j + 1, n):
                if (adj[k] & common_ij).bit_count() >= 3:
                    return False
    return True

def construct():
    n = 20
    # Deadline: 150s limit. Use 140s to be safe.
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_adj = None
    best_edges = 0
    
    # Helper to count edges
    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    # Helper to convert adjacency bitmasks to list of edges
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m ^= lsb
        return edges

    # 1. Initialization Strategy
    # We want a dense starting point. 
    # A known construction for K3,3-free graphs is the polarity graph of a projective plane,
    # but for n=20, we can just start with a random dense graph and repair it,
    # or start with a specific structure.
    
    # Let's try a few different initialization methods in a multi-start fashion.
    
    def init_random_dense():
        adj = [0] * n
        # Probability 0.6 is quite dense
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.6:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def init_polarity_like():
        # Construct a graph based on a finite field or similar structure if possible,
        # but for n=20, a random repair is often sufficient.
        # Let's try a "grid" like structure which is often K3,3 free?
        # Actually, let's just use random dense and repair.
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.7:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Repair function: Remove edges randomly until valid
    def repair(adj):
        while not is_k33_free(adj):
            # Find an edge to remove. 
            # Ideally, remove an edge that is part of a K3,3.
            # For simplicity, just pick a random edge.
            edges = []
            for i in range(n):
                m = adj[i]
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges.append((i, j))
                    m ^= lsb
            
            if not edges:
                break
            
            u, v = random.choice(edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        return adj

    # Simulated Annealing
    def simulated_annealing(initial_adj, time_budget):
        adj = initial_adj[:]
        current_edges = count_edges(adj)
        best_local_adj = adj[:]
        best_local_edges = current_edges
        
        # Temperature schedule
        T = 1.0
        cooling = 0.995
        
        while time.time() < time_budget:
            # Pick a random move: add edge, remove edge, or swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Add edge
                non_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (adj[i] & (1 << j)):
                            non_edges.append((i, j))
                if not non_edges:
                    continue
                u, v = random.choice(non_edges)
                
                # Apply move
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                new_edges = current_edges + 1
                
                # Check validity
                if is_k33_free(adj):
                    # Accept
                    current_edges = new_edges
                    if current_edges > best_local_edges:
                        best_local_edges = current_edges
                        best_local_adj = adj[:]
                else:
                    # Reject (revert)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
            elif move_type < 0.7:
                # Remove edge
                edges = []
                for i in range(n):
                    m = adj[i]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append((i, j))
                        m ^= lsb
                if not edges:
                    continue
                u, v = random.choice(edges)
                
                # Apply move
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                new_edges = current_edges - 1
                
                # Always accept removal (it keeps graph valid if it was valid)
                # But we might want to keep it if it helps escape local optima?
                # In SA, we accept bad moves with prob exp(-delta/T).
                # Here delta = new_edges - current_edges = -1.
                # Prob = exp(1/T). Since T starts at 1, prob is high.
                # But removal is always valid, so we just accept it.
                # However, if we always accept removal, we drift to empty graph.
                # We should only accept removal if it allows future additions?
                # Standard SA: accept with prob exp(-(E_new - E_old)/T).
                # E is "energy" = -edges. So delta E = 1.
                # Prob = exp(-1/T).
                if random.random() < math.exp(-1.0 / T):
                    current_edges = new_edges
                    if current_edges > best_local_edges:
                        best_local_edges = current_edges
                        best_local_adj = adj[:]
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
            else:
                # Swap: remove (u,v), add (x,y)
                edges = []
                for i in range(n):
                    m = adj[i]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append((i, j))
                        m ^= lsb
                if len(edges) < 2:
                    continue
                
                (u, v), (x, y) = random.sample(edges, 2)
                if len({u, v, x, y}) < 4:
                    continue
                
                # Apply
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                adj[x] |= (1 << y)
                adj[y] |= (1 << x)
                
                if is_k33_free(adj):
                    # Edge count remains same
                    if current_edges > best_local_edges:
                        best_local_edges = current_edges
                        best_local_adj = adj[:]
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    adj[x] &= ~(1 << y)
                    adj[y] &= ~(1 << x)
            
            T *= cooling
            
        return best_local_adj, best_local_edges

    # Multi-start
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Vary initialization density
        p = 0.5 + 0.05 * r
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair
        adj = repair(adj)
        
        # Run SA for a portion of the remaining time
        # Divide time among restarts, but give more time to later ones?
        # Or just run SA until a sub-deadline.
        sub_deadline = start_time + (deadline - start_time) * ((r + 1) / num_restarts)
        
        local_best_adj, local_best_edges = simulated_annealing(adj, sub_deadline)
        
        if local_best_edges > best_edges:
            best_edges = local_best_edges
            best_adj = local_best_adj

    # Final Polish: Greedy addition of edges
    if best_adj is None:
        return []
        
    # Try to add any remaining valid edges
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if not (best_adj[i] & (1 << j)):
                    # Try adding
                    best_adj[i] |= (1 << j)
                    best_adj[j] |= (1 << i)
                    if is_k33_free(best_adj):
                        changed = True
                    else:
                        best_adj[i] &= ~(1 << j)
                        best_adj[j] &= ~(1 << i)

    return adj_to_edges(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
