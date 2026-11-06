import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Identify candidates
    # Must share factor with 255 (3, 5, 17)
    # 255 is required.
    # Others must be in [1, 254] and gcd(x, 255) > 1.
    
    nodes = []
    for i in range(1, 255): # 1 to 254
        if math.gcd(i, 255) > 1:
            nodes.append(i)
            
    n = len(nodes)
    if n == 0:
        return [255]
        
    # 2. Build adjacency bitmasks
    # adj[i] is a bitmask of neighbors of nodes[i]
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(nodes[i], nodes[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 3. Bron-Kerbosch with pivot
    # R: current clique (list of indices)
    # P: potential candidates (bitmask)
    # X: excluded candidates (bitmask)
    
    max_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique, deadline
        
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            if len(R) > len(max_clique):
                max_clique = R[:]
            return
        
        # Pivot selection: choose u in P union X with max degree in P
        # We want to maximize |P & adj[u]| to minimize recursion
        # If P is empty, we already returned.
        
        # Simple pivot: pick first element in P? Or one with max neighbors in P?
        # Let's pick u from P U X that maximizes popcount(P & adj[u])
        
        # To speed up, we can just iterate. n is small (~130).
        # But doing this at every node is expensive.
        # Standard optimization: pick u from P if P is not empty, else from X.
        # Actually, Bron-Kerbosch with pivot picks u from P U X.
        
        # Let's implement a simple pivot:
        # If P is not empty, pick u from P.
        # To be efficient, we can pick the u that has the most neighbors in P.
        
        # For n=130, a linear scan for pivot is $O(n)$.
        # Recursion depth is up to n.
        
        # Let's try a simpler approach first: standard BK without complex pivot heuristics,
        # just picking the first available u in P U X.
        
        # However, for dense graphs, picking a high-degree pivot is crucial.
        
        # Let's find u in P U X maximizing popcount(P & adj[u]).
        # This is O(|P| + |X|) which is O(n).
        
        # Optimization: If P is not empty, we can just pick u from P.
        # Is it better to pick from X? No, picking from P is standard.
        
        # Let's implement:
        # u = argmax_{v in P U X} popcount(P & adj[v])
        
        # If P is empty, we returned. So P is not empty.
        # We can restrict u to P for simplicity, or P U X.
        # P U X is safer.
        
        # To avoid O(n) scan every time, we can use a heuristic:
        # Pick u from P.
        
        # Let's just iterate through bits in P U X.
        
        # Actually, for n=130, even O(n^2) or O(n^3) total might be okay if pruned well.
        
        # Let's use a recursive helper that returns the max clique size or set.
        
        # To handle the time limit, we check time periodically.
        
        # Implementation of BK:
        
        def bk(R, P, X):
            nonlocal max_clique
            if time.time() > deadline:
                return
            
            if P == 0 and X == 0:
                if len(R) > len(max_clique):
                    max_clique = R[:]
                return
            
            # Pivot
            # Choose u from P U X
            # We'll pick the one with max neighbors in P
            # To save time, if P is not empty, we can just pick the first one?
            # No, that's bad.
            
            # Let's try to pick u from P.
            # If P is not empty:
            #   u = one of the bits in P
            #   To optimize, we can pick the one with highest degree in P.
            
            # Let's just iterate.
            # This is a bottleneck if done naively.
            
            # Alternative: Use a library? No, std lib only.
            
            # Let's write a fast popcount. int.bit_count() is available in Python 3.8+.
            
            # Find u
            # We want to minimize the number of recursive calls.
            # The number of calls is bounded by the number of neighbors of u in P.
            # So we want u with max neighbors in P.
            
            # Candidate set for u: P | X
            # But if P is not empty, it's usually better to pick u from P.
            # Actually, Bron-Kerbosch theorem says any u in P U X works.
            
            # Let's pick u from P if P != 0, else from X.
            
            if P != 0:
                # Find u in P with max popcount(P & adj[u])
                # Iterate bits in P
                temp_P = P
                best_u = -1
                best_count = -1
                while temp_P:
                    # Get lowest set bit
                    lsb = temp_P & -temp_P
                    idx = lsb.bit_length() - 1
                    temp_P ^= lsb
                    
                    count = (P & adj[idx]).bit_count()
                    if count > best_count:
                        best_count = count
                        best_u = idx
                        if count == P.bit_count(): # Can't get better
                            break
                u = best_u
            else:
                # P is 0, so X must be non-empty (else returned)
                temp_X = X
                lsb = temp_X & -temp_X
                u = lsb.bit_length() - 1
                # P is 0, so P & adj[u] is 0.
                # This branch is only reached if we didn't return earlier?
                # Wait, if P==0 and X!=0, we return?
                # Standard BK: if P==0 and X==0, report clique.
                # If P==0 and X!=0, we can prune? No, we just return?
                # Actually, if P==0, there are no more candidates.
                # If X!=0, it means we are in a branch where we've already "seen" these nodes.
                # So we don't need to recurse further if P==0.
                # So the base case `if P == 0 and X == 0` is for reporting.
                # If P == 0, we should return immediately regardless of X?
                # Yes, because R is a maximal clique in the subgraph induced by R U P U X?
                # No. R is a clique. P are nodes that can extend R. X are nodes that were in previous branches.
                # If P is empty, R cannot be extended. It is a maximal clique.
                # We only report it if it is a *maximal* clique?
                # BK reports all maximal cliques.
                # We want the maximum one.
                # So if P==0, R is a maximal clique. We check if it's the best.
                # Then return.
                # So we don't need to handle u from X if P==0.
                pass

            # Recurse on neighbors of u in P
            # P \ N(u)
            # For each v in P \ N(u):
            #   bk(R U {v}, P & N(v), X & N(v))
            #   P = P \ {v}
            #   X = X U {v}
            
            # Wait, the standard algorithm iterates over `P \ N(u)`.
            # Let `candidates = P & ~adj[u]`
            
            candidates = P & ~adj[u]
            
            while candidates:
                if time.time() > deadline:
                    return
                
                lsb = candidates & -candidates
                v_idx = lsb.bit_length() - 1
                candidates ^= lsb
                
                # New R
                R.append(v_idx)
                
                # New P
                new_P = P & adj[v_idx]
                
                # New X
                new_X = X & adj[v_idx]
                
                bk(R, new_P, new_X)
                
                R.pop()
                
                # Update P and X
                P &= ~lsb
                X |= lsb
                
        # Initial call
        # R = []
        # P = all nodes bitmask
        # X = 0
        
        initial_P = (1 << n) - 1
        bk([], initial_P, 0)
        
    # If BK didn't finish or found nothing, fallback?
    # BK should find something.
    
    # Construct result
    result_indices = max_clique
    result_values = [nodes[i] for i in result_indices]
    result_values.append(255)
    
    return result_values
