import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit (keep 40s buffer)
    
    # Known lower bound is 31.
    # We will try to find a valid tournament for n starting from 31 upwards.
    # Strategy:
    # 1. For each n from 31 to 80:
    #    a. Generate a random tournament.
    #    b. Use vectorized local search to minimize the number of invalid 4-sets.
    #    c. If 0 invalid 4-sets are found, return the tournament.
    # 2. If time runs out, return the best valid tournament found so far 
    #    (or fallback to a known construction if none found, though we expect to find one).
    
    # Representation:
    # adj: numpy array of shape (n, n), where adj[i, j] = 1 if i -> j, 0 if j -> i.
    # Note: The task requires orientation bits for i<j.
    # 0 means i -> j, 1 means j -> i.
    # So if adj[i, j] = 1, the bit is 0. If adj[i, j] = 0, the bit is 1.
    
    def count_invalid_4sets(adj):
        """
        Count the number of 4-sets with score sequence (2,2,1,1).
        This is equivalent to counting 4-sets where no vertex has out-degree 3.
        
        A 4-set {a,b,c,d} is invalid if the induced subgraph is isomorphic to the
        unique non-transitive tournament on 4 vertices without a source, which has
        score sequence (2,2,1,1).
        
        We can count the number of valid 4-sets (those with a source) and subtract
        from total C(n,4).
        
        A 4-set has a source if there is a vertex that points to the other 3.
        Let's count the number of 4-sets where a specific vertex v is the unique source?
        No, a 4-set can have at most one source (if it has two, they point to each other? 
        No, if u and v are both sources, u->v and v->u, contradiction).
        So the number of valid 4-sets is the sum over all vertices v of the number of 
        3-sets in N^+(v) (out-neighbors of v).
        
        Wait, is it possible for a 4-set to have a source?
        Yes, if the score sequence is (3,1,1,1) or (3,2,1,0) etc.
        Actually, any 4-set with a vertex of out-degree 3 is valid.
        Since in a tournament on 4 vertices, max out-degree is 3.
        If max out-degree is 3, it's valid.
        If max out-degree is 2, it's invalid (score seq must be 2,2,1,1).
        
        So, Valid 4-sets = Sum_{v} C(out_degree(v), 3).
        Why? For each vertex v, let S_v be the set of 3-subsets of N^+(v).
        For any 3-subset {a,b,c} of N^+(v), the set {v,a,b,c} has v pointing to a,b,c.
        So v is a source in this 4-set.
        Does any 4-set get counted more than once?
        If a 4-set has two sources, say v and w, then v->w and w->v, impossible.
        So each valid 4-set has exactly one source.
        Thus, the number of valid 4-sets is exactly Sum_{v} C(out_degree(v), 3).
        
        Total 4-sets = C(n, 4).
        Invalid 4-sets = C(n, 4) - Sum_{v} C(out_degree(v), 3).
        """
        n = adj.shape[0]
        out_degrees = np.sum(adj, axis=1) # Shape (n,)
        # C(k, 3) = k * (k-1) * (k-2) / 6
        # Handle k < 3
        valid_count = 0
        # Vectorized computation of C(out_degrees, 3)
        od = out_degrees.astype(np.int64)
        # Avoid negative numbers in product
        c3 = np.zeros_like(od)
        mask = od >= 3
        c3[mask] = od[mask] * (od[mask] - 1) * (od[mask] - 2) // 6
        total_valid = np.sum(c3)
        
        total_4sets = n * (n - 1) * (n - 2) * (n - 3) // 24
        
        return total_4sets - total_valid

    def get_invalid_4sets_and_edges(adj):
        """
        Not needed if we just flip random edges and check delta.
        But for local search, we need to know which edges to flip.
        We can just flip a random edge and see if the violation count decreases.
        """
        pass

    def local_search(adj, n, max_iter=2000):
        """
        Hill climbing on the number of invalid 4-sets.
        """
        current_adj = adj.copy()
        current_invalid = count_invalid_4sets(current_adj)
        
        if current_invalid == 0:
            return current_adj, 0
        
        best_adj = current_adj.copy()
        best_invalid = current_invalid
        
        # Precompute indices for random edge selection
        # There are n*(n-1)/2 edges.
        # We can pick i, j randomly.
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random edge (i, j) with i != j
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            # Flip edge i -> j
            # current_adj[i, j] is 1 if i->j.
            # If we flip, i->j becomes j->i.
            # So current_adj[i, j] becomes 1 - current_adj[i, j]
            # And current_adj[j, i] becomes 1 - current_adj[j, i]
            
            old_ij = current_adj[i, j]
            old_ji = current_adj[j, i]
            
            current_adj[i, j] = 1 - old_ij
            current_adj[j, i] = 1 - old_ji
            
            new_invalid = count_invalid_4sets(current_adj)
            
            if new_invalid < best_invalid:
                best_invalid = new_invalid
                best_adj = current_adj.copy()
                
                if best_invalid == 0:
                    break
            else:
                # Revert if not better (or even if equal, to keep it simple/stable, 
                # though accepting equal moves can help escape plateaus. 
                # Let's accept equal moves with some probability to allow drift).
                if new_invalid == best_invalid and random.random() < 0.1:
                    # Accept
                    best_invalid = new_invalid
                    best_adj = current_adj.copy()
                else:
                    # Revert
                    current_adj[i, j] = old_ij
                    current_adj[j, i] = old_ji
        
        return best_adj, best_invalid

    best_n = None
    best_orient = None
    
    # Try n from 31 to 80
    # Start with smaller n as it's the goal.
    for n in range(31, 81):
        if time.time() > deadline:
            break
        
        # Number of restarts decreases as n increases? 
        # Or fixed budget?
        # Let's give each n a chance.
        # For small n, local search might get stuck in local minima with >0 violations.
        # We need multiple restarts.
        
        max_restarts = 5 if n < 40 else 2
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Generate random tournament
            # adj[i, j] = 1 if i -> j
            adj = np.zeros((n, n), dtype=np.int8)
            # Fill upper triangle
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 0
                    else:
                        adj[i, j] = 0
                        adj[j, i] = 1
            
            # Local search
            adj_opt, invalid_count = local_search(adj, n, max_iter=1000)
            
            if invalid_count == 0:
                # Found a valid tournament!
                best_n = n
                # Convert to orientation list
                orientation = []
                for i in range(n):
                    for j in range(i + 1, n):
                        # 0 means i -> j (adj[i, j] == 1)
                        # 1 means j -> i (adj[i, j] == 0)
                        orientation.append(0 if adj_opt[i, j] == 1 else 1)
                return {'n': n, 'orientation': orientation}
            
            # If we found a tournament with fewer violations than current best for this n?
            # We don't track "best for this n" separately, we just want valid.
            # If no valid found, continue.

    # Fallback: If we didn't find a valid tournament for n < 81, 
    # we should return the best we found? 
    # The task says "Minimize n". 
    # If we don't find a valid one, we must return *a* valid one.
    # The problem statement says "The smallest known has 67 vertices".
    # We should ensure we return a valid tournament.
    # If our search failed, we can use a known construction or a larger random one.
    # But random ones for large n are likely valid?
    # Let's check n=83 (Paley) or just try a larger n with random + local search.
    
    # Let's try n=83 with a few restarts to be safe.
    n = 83
    for r in range(5):
        if time.time() > deadline:
            break
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 0
                else:
                    adj[i, j] = 0
                    adj[j, i] = 1
        
        adj_opt, invalid_count = local_search(adj, n, max_iter=500)
        if invalid_count == 0:
            orientation = []
            for i in range(n):
                for j in range(i + 1, n):
                    orientation.append(0 if adj_opt[i, j] == 1 else 1)
            return {'n': n, 'orientation': orientation}
            
    # Absolute fallback: Paley 83
    # Paley tournament of order p (prime, p=3 or 7 mod 4)
    # p=83 is 3 mod 4.
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            # i -> j if (j - i) % p is a quadratic residue
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
    return {'n': p, 'orientation': orientation}
