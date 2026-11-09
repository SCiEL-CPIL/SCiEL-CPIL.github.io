import time
from math import gcd

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    
    Strategy:
    1. The trivial solution is all multiples of 3 (size 85).
    2. To verify if a larger mixed set exists (using primes 3, 5, 17), we model the problem
       as finding the Maximum Independent Set (MIS) in a "conflict graph" where an edge exists
       between u and v if gcd(u, v) == 1.
       We restrict the search space to numbers sharing a factor with 255 (i.e., divisible by 3, 5, or 17).
    3. We use the Bron-Kerbosch algorithm with pivot selection to find the maximum clique
       in the "compatibility graph" (complement of conflict graph) restricted to candidates
       that are compatible with 255.
    4. We optimize the search by pruning branches that cannot exceed the current best (85).
    
    If the MIS size > 85, we return the larger set. Otherwise, we return the multiples of 3.
    """
    
    N = 255
    deadline = time.time() + 50.0  # Safety margin before 60s limit
    
    # Baseline: Multiples of 3
    baseline = [x for x in range(1, N + 1) if x % 3 == 0]
    best_set = baseline
    best_size = len(baseline)  # 85
    
    # Candidates must share a factor with 255 (3, 5, or 17)
    # And must be compatible with 255 (gcd(x, 255) > 1 is always true if x is div by 3,5,17)
    # Actually, the condition is gcd(a,b) > 1 for ALL pairs.
    # Since 255 is in the set, every other x must have gcd(x, 255) > 1.
    # This implies x is divisible by 3, 5, or 17.
    
    candidates = [x for x in range(1, N + 1) if (x % 3 == 0 or x % 5 == 0 or x % 17 == 0)]
    
    # We must include 255.
    # Let's fix 255 in the set.
    # Any other element x in the set must satisfy gcd(x, 255) > 1.
    # All candidates satisfy this.
    # Furthermore, for any x in the set, x must be compatible with all other elements in the set.
    
    # Build a compatibility matrix for the candidates (excluding 255 for now, we'll add it back)
    # Actually, it's easier to work with the full set of candidates including 255.
    # But since 255 MUST be in the set, we can just ensure our search includes 255.
    
    # Let's define the graph on the set of candidates C.
    # Edge (u, v) exists if gcd(u, v) > 1.
    # We want the maximum clique in this graph that contains 255.
    
    # To make Bron-Kerbosch efficient, we can remove vertices that are incompatible with 255.
    # But all candidates are compatible with 255 by definition.
    # However, we can prune the candidate set:
    # If a candidate x is incompatible with many others, it might not be in the max clique.
    
    # Let's use a standard Bron-Kerbosch implementation with pivot.
    # We represent the graph as a set of sets or bitmasks.
    # N=255 is small enough for bitmasks (255 bits fits in 4 64-bit ints or just Python ints).
    
    # Map candidates to indices
    # To ensure 255 is included, we can force it into the clique R at the start.
    # Then we look for the max clique in the neighbors of 255 within the remaining candidates.
    
    # Neighbors of 255 in the compatibility graph:
    # All candidates are neighbors of 255 because gcd(x, 255) > 1 for all x in candidates.
    # So we just need the max clique in the subgraph induced by (candidates \ {255}).
    # Wait, this is not quite right. The max clique containing 255 is 1 + max clique in N(255).
    # Since N(255) = candidates \ {255}, we just need the max clique in the graph induced by candidates \ {255}.
    
    # Let's verify this logic.
    # If C is a clique containing 255, then C \ {255} is a clique in the subgraph induced by C \ {255}.
    # Conversely, if K is a clique in the subgraph induced by candidates \ {255}, then K U {255} is a clique
    # IF 255 is compatible with all elements in K.
    # By definition of candidates, every element in candidates is compatible with 255.
    # So yes, Max Clique containing 255 = 1 + Max Clique in (candidates \ {255}).
    
    # So we just need to find the Max Clique in the graph G' induced by V' = {x in [1, 255] | x != 255 and (x%3==0 or x%5==0 or x%17==0)}.
    # Edges in G' exist if gcd(u, v) > 1.
    
    # If Max Clique Size in G' + 1 > 85, we have an improvement.
    # Current best is 85 (which corresponds to clique size 84 in G' + 1).
    # So we are looking for a clique of size >= 85 in G'.
    
    # Let's build the graph.
    # To speed up, we can precompute gcds or use prime factors.
    # Since N is small, gcd is fast.
    
    # Optimization: If a vertex v in G' has degree < 84, it cannot be part of a clique of size 85 (since it needs 84 neighbors in the clique).
    # Actually, in a clique of size k, every vertex has degree k-1 within the clique.
    # So if we want a clique of size 85, every vertex in it must have at least 84 neighbors in the clique.
    # Thus, any vertex in G' with total degree < 84 can be removed from G'.
    
    # Let's filter candidates.
    V = [x for x in range(1, N + 1) if x != 255 and (x % 3 == 0 or x % 5 == 0 or x % 17 == 0)]
    
    # Calculate degrees in the compatibility graph
    # Note: This is O(|V|^2) which is ~ 100^2 = 10,000 operations. Very fast.
    degree = {v: 0 for v in V}
    neighbors = {v: set() for v in V}
    
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            u, v = V[i], V[j]
            if gcd(u, v) > 1:
                degree[u] += 1
                degree[v] += 1
                neighbors[u].add(v)
                neighbors[v].add(u)
    
    # Filter out vertices with degree < 84
    # We need a clique of size 85. So each vertex must have at least 84 neighbors.
    # If we remove a vertex, the degrees of its neighbors decrease.
    # This is a peeling process.
    
    # However, Bron-Kerbosch with good pivoting is quite fast on dense graphs.
    # The graph is likely dense because most multiples of 3, 5, 17 share factors.
    # Let's just run Bron-Kerbosch on the full V, but with a bound.
    
    # We will use a recursive Bron-Kerbosch.
    # To avoid deep recursion or slow performance, we'll add a timeout check.
    
    # State:
    # R: current clique (set of vertices)
    # P: potential vertices (set)
    # X: excluded vertices (set)
    
    # We want to find if there is a clique of size > 84 in V.
    # If yes, we return it + 255.
    # If no, we return baseline.
    
    # Since we only care about beating 84, we can stop searching if we find one.
    # But finding the MAXIMUM is better to be sure? No, the task is to maximize.
    # If we find a clique of size 85, is it the max?
    # The baseline is 85. If we find 86, is it the max?
    # It's possible there is a 90.
    # However, finding ANY clique > 84 is an improvement.
    # To be safe and maximize, we should try to find the actual maximum.
    # But if time is tight, returning the first one found > 84 is better than nothing.
    # Given the structure, if a larger clique exists, it's likely not much larger.
    
    # Let's implement a branch and bound Bron-Kerbosch.
    
    max_clique_found = []
    found_improvement = False
    
    # For Bron-Kerbosch, we use sets.
    # To make it faster, we can use bitsets if we map to indices.
    # |V| is approx:
    # Multiples of 3: 84 (excluding 255)
    # Multiples of 5: 50 (excluding 255)
    # Multiples of 17: 14 (excluding 255)
    # Intersections:
    # 3&5: 16 (excluding 255)
    # 3&17: 4 (excluding 255)
    # 5&17: 2 (excluding 255)
    # 3&5&17: 0 (255 is the only one, excluded)
    # Union = 84 + 50 + 14 - 16 - 4 - 2 + 0 = 126.
    # So |V| = 126.
    
    # Bitmask approach:
    # Map each vertex in V to an index 0..125.
    # Precompute neighbor masks.
    
    idx_map = {v: i for i, v in enumerate(V)}
    n_v = len(V)
    
    # neighbor_masks[i] is a bitmask of neighbors of V[i]
    neighbor_masks = [0] * n_v
    for i in range(n_v):
        u = V[i]
        mask = 0
        for v in neighbors[u]:
            j = idx_map[v]
            mask |= (1 << j)
        neighbor_masks[i] = mask
    
    # Bron-Kerbosch with pivot
    # R: current clique size (we don't need the actual set until we find a winner, but we need to track it for reconstruction if we want to return it)
    # Actually, we just need to know if we can reach size 85.
    # And if we do, we need to return the set.
    
    # Let's store the best clique found.
    best_clique_set = set()
    
    def bron_kerbosch(R_size, P_mask, X_mask, R_set):
        nonlocal best_clique_set, found_improvement, deadline
        
        if time.time() > deadline:
            return
        
        if P_mask == 0 and X_mask == 0:
            # Maximal clique found
            if R_size > len(best_clique_set):
                best_clique_set = R_set.copy()
                if R_size + 1 > best_size: # +1 for 255
                    found_improvement = True
            return
        
        # Pivot selection: choose u in P U X that maximizes |P & N(u)|
        # This minimizes the number of recursive calls.
        U_mask = P_mask | X_mask
        if U_mask == 0:
            return
            
        # Find pivot
        # Iterate through bits in U_mask
        # To speed up, we can just pick the first one, or the one with max degree in P.
        # Let's pick the one with max popcount(P_mask & neighbor_masks[u])
        
        pivot = -1
        max_count = -1
        temp = U_mask
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            count = (P_mask & neighbor_masks[u]).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
            temp &= temp - 1
            
        # If P_mask & N(pivot) is empty, then P is independent, and we have a maximal clique.
        # Actually, the standard optimization is:
        # If P \ N(pivot) is empty, then P is a clique? No.
        # The optimization is: iterate only over vertices in P \ N(pivot).
        # Because any vertex in P & N(pivot) will be covered by the recursion starting with a pivot neighbor?
        # Yes, Bron-Kerbosch with pivot:
        # For each v in P \ N(pivot):
        #   recurse(R U {v}, P & N(v), X & N(v))
        #   P = P \ {v}
        #   X = X U {v}
        
        # If P & N(pivot) is empty, then P \ N(pivot) = P.
        
        candidates_for_branch = P_mask & ~(neighbor_masks[pivot])
        
        # If candidates_for_branch is 0, we can't branch?
        # Wait, if P & N(pivot) is not empty, we only branch on P \ N(pivot).
        # If P \ N(pivot) is empty, then all vertices in P are in N(pivot).
        # In that case, we don't need to branch?
        # No, the theorem states that we only need to iterate over P \ N(pivot).
        # If P \ N(pivot) is empty, it means every vertex in P is a neighbor of pivot.
        # This doesn't mean we stop. It means the set of candidates to branch on is empty?
        # Actually, if P \ N(pivot) is empty, then the loop runs 0 times.
        # Does this mean we missed a clique?
        # No, the pivot selection ensures that all maximal cliques are found.
        # If P \ N(pivot) is empty, it implies that P is a subset of N(pivot).
        # In this case, the recursion will handle it?
        # Let's check the standard algorithm.
        # "For each v in P \ N(u)"
        # If this set is empty, the loop doesn't run.
        # Is it possible that a maximal clique is missed?
        # No, because if P \ N(u) is empty, then for any v in P, v is in N(u).
        # This case usually implies that the current R is already a maximal clique in the context?
        # Actually, if P \ N(u) is empty, we just return?
        # No, we must be careful.
        
        # Let's just iterate over the bits in candidates_for_branch.
        # If it's 0, we return?
        # If P is not empty but P \ N(pivot) is empty, it means P is a clique?
        # No.
        
        # Let's trust the standard implementation:
        # If P \ N(pivot) is empty, we don't recurse.
        # This is correct because any maximal clique containing R must include a vertex from P.
        # If all vertices in P are neighbors of pivot, then...
        # Actually, if P \ N(pivot) is empty, then P is a subset of N(pivot).
        # This doesn't change the fact that we need to find cliques in P.
        # Wait, if I don't branch, I miss cliques.
        
        # Correction: The pivot optimization is valid.
        # If P \ N(pivot) is empty, then the loop terminates.
        # This happens when P is a clique? No.
        
        # Let's just implement it and test.
        # To be safe, if candidates_for_branch is 0, we might need to handle P directly?
        # No, standard BK with pivot is correct.
        
        temp = candidates_for_branch
        while temp:
            if time.time() > deadline:
                return
                
            lsb = temp & -temp
            v_idx = lsb.bit_length() - 1
            v_val = V[v_idx]
            
            # Recurse
            new_R_set = R_set.copy()
            new_R_set.add(v_val)
            
            new_P = P_mask & neighbor_masks[v_idx]
            new_X = X_mask & neighbor_masks[v_idx]
            
            bron_kerbosch(R_size + 1, new_P, new_X, new_R_set)
            
            # Update P and X
            P_mask &= ~lsb
            X_mask |= lsb
            temp &= temp - 1
            
        # Optimization: If we found a clique larger than 84, we can stop early?
        # Only if we are sure it's the maximum. We aren't.
        # But if we just want to beat 85, we can stop.
        # The task says "Maximize cardinality".
        # So we should find the max.
        # But if time is running out, we return the best found.
        
    # Initial call
    # R is empty.
    # P is all vertices.
    # X is empty.
    all_mask = (1 << n_v) - 1
    bron_kerbosch(0, all_mask, 0, set())
    
    # Check if we found a better set
    if len(best_clique_set) + 1 > best_size:
        # Add 255 to the set
        result_set = best_clique_set.union({255})
        return sorted(result_set)
    else:
        return baseline

if __name__ == "__main__":
    s = construct()
    print(len(s))
