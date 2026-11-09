import time
import math
import sys

def construct():
    """
    Find the maximum cardinality subset S of [1, 512] such that lcm(a, b) <= 512 for all a, b in S.
    This is equivalent to finding the Maximum Clique in the graph where edge (u, v) exists if lcm(u, v) <= 512.
    We use the Bron-Kerbosch algorithm with Pivot Selection, optimized using Python integers as bitmasks.
    """
    start_time = time.time()
    LIMIT = 512
    N = LIMIT
    # Safety margin: return before 90s. Let's aim for 85s.
    deadline = start_time + 85

    # 1. Precompute adjacency bitmasks
    # adj[i] is a bitmask where bit j is 1 if lcm(i, j) <= 512
    # We use 0-indexed bits for numbers 1..512. 
    # Bit i corresponds to number i+1.
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(i, N + 1):
            # Optimization: if i > 512, lcm is > 512. But i <= 512.
            # lcm(i, j) = i * j // gcd(i, j)
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                # Set bit for j (index j-1) and bit for i (index i-1)
                # We will fill the full mask in a second pass or do it symmetrically here.
                # Since we iterate j >= i, we set bit j-1.
                mask |= (1 << (j - 1))
        # We need symmetric adjacency. 
        # The loop above sets bits for j >= i.
        # We need to ensure adj[i] has bits for all j such that lcm(i,j) <= 512.
        # The inner loop covers j >= i. What about j < i?
        # lcm(i, j) is symmetric. If we compute for all pairs i <= j, we can fill adj.
        # Let's just do a full double loop or fill symmetrically.
        # Given N=512, a double loop is 512*512 = 262k operations, which is trivial.
        pass

    # Redo adjacency construction properly
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                # Set bit for i-1 in adj[j] and bit for j-1 in adj[i]
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))
                # Self-loop? lcm(i,i)=i <= 512. 
                # For clique finding, we usually consider simple graphs.
                # A clique of size 1 is valid.
                # We don't strictly need self-loops for the BK algorithm logic 
                # if we handle the base case, but it doesn't hurt.
                # Actually, standard BK works on simple graphs.
                # We should NOT include self-loops in the adjacency mask for the 
                # "neighbors" set P, because P contains candidates not yet in R.
                # If i is in R, i is not in P. So self-loop doesn't affect P intersection.
                # However, when we pick a pivot p, we look at N(p).
                # If we include self-loops, N(p) includes p.
                # The standard BK pivot logic:
                # If P is empty, update best.
                # Else, pick pivot u in P U X.
                # For each v in P \ N(u):
                #   BK(R U {v}, P \ N(v), X \ N(v))
                #   P = P \ {v}
                #   X = X U {v}
                
                # If N(u) includes u, and u is in P U X, then P \ N(u) excludes u.
                # This is correct behavior.
                
                # Let's ensure adj[i] does NOT have bit i-1 set to keep it a simple graph representation 
                # for the logic "neighbors of v in P".
                # Actually, it's cleaner if adj[i] only contains j != i.
                # The loop above sets adj[i] bit j-1 and adj[j] bit i-1.
                # It does NOT set adj[i] bit i-1. So it's a simple graph.

    best_clique_size = 0
    best_clique_mask = 0

    # Bron-Kerbosch with Pivot
    # R: current clique (bitmask)
    # P: candidates (bitmask)
    # X: excluded (bitmask)
    
    # Helper to get list of vertices from mask
    def mask_to_list(mask):
        return [i + 1 for i in range(N) if (mask >> i) & 1]

    # To speed up pivot selection, we might want to precompute something, 
    # but bit operations are fast.
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique_size, best_clique_mask, deadline
        
        if time.time() > deadline:
            return False # Signal timeout
        
        # If P is empty, we have a maximal clique
        if P == 0:
            size = bin(R).count('1')
            if size > best_clique_size:
                best_clique_size = size
                best_clique_mask = R
            return True
        
        # Select pivot u from P U X
        # We want u that maximizes |N(u) \ P| or minimizes |P \ N(u)|
        # Standard heuristic: max degree in P U X
        # Since we are using bitmasks, we can just pick the one with highest popcount of (adj[u] & (P | X))
        # Or simply pick the first one? No, pivot is crucial for performance.
        
        PX = P | X
        # Find pivot u
        # Iterate through bits in PX
        # We can just pick a vertex with high degree in the induced subgraph.
        # For simplicity and speed in Python, we can just pick the first vertex 
        # or the one with the highest index? 
        # Better: pick the vertex u in PX such that (adj[u] & P) has max popcount.
        # This minimizes the candidates P \ N(u).
        
        max_pop = -1
        pivot = -1
        temp = PX
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            u_idx = (lsb.bit_length() - 1)
            u = u_idx + 1
            # Count neighbors of u in P
            # adj[u] is bitmask of neighbors
            # We want popcount(adj[u] & P)
            pop = (adj[u] & P).bit_count()
            if pop > max_pop:
                max_pop = pop
                pivot = u
            temp ^= lsb
            
        if pivot == -1:
            # PX was empty? But P is not empty, so this shouldn't happen.
            return True

        # Candidates are P \ N(pivot)
        # N(pivot) is adj[pivot]
        # We want vertices v in P such that v is NOT neighbor of pivot.
        # Note: pivot itself is in P (since P subset PX). 
        # Is pivot a neighbor of pivot? No, simple graph.
        # So pivot is in P \ N(pivot).
        
        candidates = P & ~adj[pivot]
        
        # Iterate through candidates
        while candidates:
            if time.time() > deadline:
                return False
            
            # Pick one candidate v
            lsb = candidates & -candidates
            v_idx = (lsb.bit_length() - 1)
            v = v_idx + 1
            
            # Recurse
            # New R = R | {v}
            # New P = P & ~adj[v]  (neighbors of v in current P)
            # New X = X & ~adj[v]
            
            new_R = R | lsb
            new_P = P & ~adj[v]
            new_X = X & ~adj[v]
            
            if not bron_kerbosch(new_R, new_P, new_X):
                return False # Timeout propagated
            
            # Update P and X for next iteration
            P = P & ~lsb
            X = X | lsb
            candidates = candidates & ~lsb
            
        return True

    # Initial call
    # R = 0
    # P = all vertices (1..512)
    # X = 0
    initial_P = (1 << N) - 1
    # We want to start with a good upper bound if possible to prune?
    # BK doesn't prune based on size directly in the standard form, 
    # but we can stop if max possible size <= best_clique_size.
    # Max possible size from current state is |R| + |P|.
    # If |R| + |P| <= best_clique_size, we can prune.
    
    # Let's add this pruning to the recursion.
    # We need to modify bron_kerbosch to check this.
    
    def bron_kerbosch_opt(R, P, X):
        nonlocal best_clique_size, best_clique_mask, deadline
        
        if time.time() > deadline:
            return False
        
        # Pruning: if current clique size + remaining candidates <= best, stop
        if bin(R).count('1') + bin(P).count('1') <= best_clique_size:
            return True
            
        if P == 0:
            size = bin(R).count('1')
            if size > best_clique_size:
                best_clique_size = size
                best_clique_mask = R
            return True
        
        PX = P | X
        if PX == 0:
            # No pivot possible? Should not happen if P is not empty.
            # If P is not empty, PX is not empty.
            return True

        # Select pivot u from P U X to minimize candidates
        max_pop = -1
        pivot = -1
        temp = PX
        while temp:
            lsb = temp & -temp
            u_idx = (lsb.bit_length() - 1)
            u = u_idx + 1
            pop = (adj[u] & P).bit_count()
            if pop > max_pop:
                max_pop = pop
                pivot = u
            temp ^= lsb
        
        if pivot == -1:
            return True

        candidates = P & ~adj[pivot]
        
        while candidates:
            if time.time() > deadline:
                return False
            
            lsb = candidates & -candidates
            v_idx = (lsb.bit_length() - 1)
            v = v_idx + 1
            
            new_R = R | lsb
            new_P = P & ~adj[v]
            new_X = X & ~adj[v]
            
            if not bron_kerbosch_opt(new_R, new_P, new_X):
                return False
            
            P = P & ~lsb
            X = X | lsb
            candidates = candidates & ~lsb
            
        return True

    # Run the search
    # To handle potential timeouts gracefully, we run it.
    # If it times out, we return the best found so far.
    # Since we want the MAXIMUM, and BK is exact, if it finishes, it's optimal.
    # If it times out, the result is a lower bound.
    
    # We can also try to seed with a greedy solution to set a high initial best_clique_size.
    # This helps with pruning.
    
    # Greedy Seed
    def greedy_seed():
        # Order by degree descending
        degrees = [(adj[i].bit_count(), i) for i in range(1, N + 1)]
        degrees.sort(reverse=True)
        S = set()
        for deg, i in degrees:
            # Check if i is compatible with all in S
            # i is compatible with all in S if adj[i] & mask(S) == mask(S)
            # We can maintain a mask of S
            pass
        
        # Simpler greedy:
        # Start with empty set.
        # At each step, add the vertex with highest degree in the remaining compatible set.
        R_mask = 0
        P_mask = (1 << N) - 1
        
        while P_mask:
            # Find vertex in P with max degree in P
            best_v = -1
            best_deg = -1
            temp = P_mask
            while temp:
                lsb = temp & -temp
                v_idx = (lsb.bit_length() - 1)
                v = v_idx + 1
                # Degree of v in P is popcount(adj[v] & P_mask)
                d = (adj[v] & P_mask).bit_count()
                if d > best_deg:
                    best_deg = d
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
            
            # Add best_v to R
            R_mask |= (1 << (best_v - 1))
            # Update P: remove best_v and its non-neighbors?
            # No, for greedy clique, we keep only neighbors of best_v.
            P_mask = P_mask & adj[best_v]
            P_mask &= ~(1 << (best_v - 1)) # Remove best_v itself
            
        return R_mask

    seed_mask = greedy_seed()
    best_clique_size = bin(seed_mask).count('1')
    best_clique_mask = seed_mask
    
    # Now run BK
    # We can restrict the search space? No, we need global max.
    # But we can use the seed size for pruning.
    
    success = bron_kerbosch_opt(0, (1 << N) - 1, 0)
    
    # If success is False, it timed out. We still have best_clique_mask.
    # Convert mask to list
    result = mask_to_list(best_clique_mask)
    
    # Fallback if result is empty (shouldn't happen)
    if not result:
        result = [1]
        
    return result

# Run construct to test
# print(construct())
