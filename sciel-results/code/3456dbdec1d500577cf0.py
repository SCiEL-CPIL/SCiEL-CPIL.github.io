import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    def time_check():
        return time.time() > deadline

    # We will attempt to find a valid graph for n=30, then n=31, etc.
    # Start with a random graph and use simulated annealing to minimize violations.
    
    # Objective: Minimize (count_K4 + count_Ind6)
    # If we reach 0, the graph is valid.
    
    def get_initial_graph(n):
        # Initialize with a random graph or a structured one (e.g. random regular)
        # A random graph with p=0.5 is a good start for local search
        adj = [[False]*n for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = True
                    adj[j][i] = True
                    edges.append([i, j])
        return adj, edges

    def count_k4(adj, n):
        # Efficiently count K4s
        # For each pair (i,j), check common neighbors for edges
        count = 0
        # Precompute common neighbors for speed? 
        # For n=30, O(n^4) is 810k, which is fast.
        # But we can do better.
        # Iterate over triangles, then check 4th vertex.
        # Triangles: O(n^3) = 27k. For each triangle, check common neighbors.
        # Let's just do the simple O(n^4) loop with early exit or simple counting.
        # Actually, for local search, we need the count to change.
        
        # Optimization: 
        # For each i < j < k < l, check if all 6 edges exist.
        # This is too slow if called frequently in a tight loop with high n.
        
        # Better: 
        # 1. Find all triangles.
        # 2. For each triangle (i,j,k), count edges between them and other vertices? No.
        
        # Let's use the standard method:
        # For each i:
        #   For each j in adj[i] where j > i:
        #     For each k in adj[i] & adj[j] where k > j:
        #       For each l in adj[i] & adj[j] & adj[k] where l > k:
        #         count++
        
        # To speed up, we can precompute intersection sets.
        # But since n is small (30-35), set intersections are fast.
        
        # We'll implement a fast counter that also allows us to see *which* K4s exist
        # for potential targeted repair, but for now just count.
        
        k4s = []
        for i in range(n):
            ni = adj[i]
            for j in range(i+1, n):
                if not ni[j]: continue
                nij = [k for k in range(j+1, n) if ni[k] and adj[j][k]]
                for k in nij:
                    nijk = [l for l in range(k+1, n) if ni[l] and adj[j][l] and adj[k][l]]
                    if nijk:
                        count += len(nijk)
                        # For repair, we might want to store these, but storing is expensive.
                        # We'll just return count for now.
        return count

    def count_ind6(adj, n):
        # Count independent sets of size 6.
        # This is equivalent to counting K6 in the complement graph.
        # Comp graph: adj_comp[i][j] = not adj[i][j] and i!=j
        
        # We can use the same logic as K4 but on complement.
        # However, building complement every time is O(n^2).
        # We can pass adj and check `not adj[i][j]`.
        
        count = 0
        # Independent set of size 6 means 6 vertices with NO edges between them.
        # i < j < k < l < m < p
        # check all pairs are non-edges.
        
        # This is O(n^6) if naive. n=30 -> 593,775,500. Too slow for frequent calls.
        
        # We need a faster way to count or estimate independent sets.
        # Or, we can use a recursive backtracking that stops early?
        # No, we need the total count for the penalty.
        
        # Alternative:
        # Use the fact that we only need to know if count > 0.
        # For local search, we can use a "hard" constraint approach:
        # If we find a K4 or Ind6, we add a penalty.
        # To speed up counting Ind6, we can sample? No, must be exact for validity.
        # But for the *search*, we can use a heuristic penalty that correlates with the count.
        
        # Let's try a different angle:
        # Instead of counting ALL Ind6s, we can just check if ANY exists.
        # If we use a binary objective (0 if valid, 1 if invalid), local search gets stuck easily.
        # We need a graded penalty.
        
        # How to count Ind6 fast?
        # We can precompute for each vertex v, the set of non-neighbors.
        # Then an Ind6 is a clique of size 6 in the graph of non-edges.
        
        # Let's optimize the counting by using bitsets (integers) for adjacency.
        # This will be much faster for n<=64.
        
        return 0 # Placeholder, will rewrite with bitsets

    # --- Rewrite using Bitsets for Speed ---
    
    def to_bitsets(adj, n):
        bits = [0] * n
        for i in range(n):
            b = 0
            for j in range(n):
                if adj[i][j]:
                    b |= (1 << j)
            bits[i] = b
        return bits

    def count_k4_bits(bits, n):
        count = 0
        for i in range(n):
            bi = bits[i]
            # j > i
            # We iterate j
            for j in range(i+1, n):
                if not (bi >> j) & 1: continue
                bij = bi & bits[j]
                # k > j
                # We need k in bij, k > j
                # And l in bij & bits[k], l > k
                # This is still O(n^3) or worse per pair.
                
                # Let's just iterate k
                # To avoid double counting and ensure i<j<k<l:
                # We can iterate k from j+1 to n
                # If k in bij:
                #   bijk = bij & bits[k]
                #   Count bits in bijk that are > k
                #   mask = ((1 << n) - 1) ^ ((1 << (k+1)) - 1)
                #   count += bin(bijk & mask).count('1')
                
                for k in range(j+1, n):
                    if not (bij >> k) & 1: continue
                    bijk = bij & bits[k]
                    # Mask out bits <= k
                    mask = ((1 << n) - 1) ^ ((1 << (k+1)) - 1)
                    count += bin(bijk & mask).count('1')
        return count

    def count_ind6_bits(bits, n):
        # Count K6 in complement.
        # Comp bits: ~bits[i] & ((1<<n)-1) & ~(1<<i)
        # Precompute comp bits?
        comp = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = full_mask ^ bits[i] ^ (1 << i)
            
        count = 0
        for i in range(n):
            ci = comp[i]
            for j in range(i+1, n):
                if not (ci >> j) & 1: continue
                cij = ci & comp[j]
                for k in range(j+1, n):
                    if not (cij >> k) & 1: continue
                    cijk = cij & comp[k]
                    for l in range(k+1, n):
                        if not (cijk >> l) & 1: continue
                        cijkl = cijk & comp[l]
                        for m in range(l+1, n):
                            if not (cijkl >> m) & 1: continue
                            cijklm = cijkl & comp[m]
                            # Count bits in cijklm > m
                            mask = ((1 << n) - 1) ^ ((1 << (m+1)) - 1)
                            count += bin(cijklm & mask).count('1')
        return count

    # This counting is still O(n^6) for Ind6. n=30 is ~500M ops. Too slow for 120s if called many times.
    # We need a smarter penalty or a different search.
    
    # Alternative Strategy: 
    # Use the "Repair" approach.
    # 1. Start with a random graph.
    # 2. Find a K4 or Ind6.
    # 3. Flip an edge to break it.
    # 4. Repeat.
    
    # This avoids counting ALL violations, just finding ONE.
    # Finding a K4 is fast. Finding an Ind6 is the bottleneck.
    
    def find_k4(bits, n):
        for i in range(n):
            bi = bits[i]
            for j in range(i+1, n):
                if not (bi >> j) & 1: continue
                bij = bi & bits[j]
                for k in range(j+1, n):
                    if not (bij >> k) & 1: continue
                    bijk = bij & bits[k]
                    # Check if any bit > k is set
                    mask = ((1 << n) - 1) ^ ((1 << (k+1)) - 1)
                    if bijk & mask:
                        # Find the bit
                        l_mask = bijk & mask
                        # Get lowest set bit
                        l = (l_mask & -l_mask).bit_length() - 1
                        return (i, j, k, l)
        return None

    def find_ind6(bits, n):
        # Find K6 in complement.
        comp = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = full_mask ^ bits[i] ^ (1 << i)
            
        for i in range(n):
            ci = comp[i]
            for j in range(i+1, n):
                if not (ci >> j) & 1: continue
                cij = ci & comp[j]
                for k in range(j+1, n):
                    if not (cij >> k) & 1: continue
                    cijk = cij & comp[k]
                    for l in range(k+1, n):
                        if not (cijk >> l) & 1: continue
                        cijkl = cijk & comp[l]
                        for m in range(l+1, n):
                            if not (cijkl >> m) & 1: continue
                            cijklm = cijkl & comp[m]
                            mask = ((1 << n) - 1) ^ ((1 << (m+1)) - 1)
                            if cijklm & mask:
                                p_mask = cijklm & mask
                                p = (p_mask & -p_mask).bit_length() - 1
                                return (i, j, k, l, m, p)
        return None

    # This is still O(n^6) worst case for find_ind6.
    # But if the graph is "close" to valid, it might find one quickly?
    # Or if it's random, it might find one quickly?
    # Actually, for a random graph, Ind6 is very likely to exist and be found quickly.
    # As we approach validity, Ind6s become rare and hard to find.
    
    # To handle the "hard to find" phase, we can use a heuristic:
    # If we can't find an Ind6 in X seconds, assume it's gone? No, dangerous.
    
    # Let's try a different representation: 
    # We want to MAXIMIZE n.
    # Known: Paley(29) is valid.
    # Can we extend Paley(29) to 30?
    # The current program tries to do this via hitting sets.
    
    # Let's try a simpler local search on n=30.
    # Initialize with Paley(29) + 1 random vertex.
    # Then flip edges incident to the new vertex, and also edges in the base graph if needed.
    
    # Actually, let's just do general local search on n=30.
    # Penalty: 10 * (1 if K4 else 0) + 10 * (1 if Ind6 else 0)
    # This is a binary penalty. Local search with binary penalty is like random walk with acceptance.
    
    # Better Penalty:
    # Count of K4s + Count of Ind6s.
    # But counting Ind6s is slow.
    
    # Compromise:
    # Use "Find One" logic.
    # If K4 found: penalty += 100
    # If Ind6 found: penalty += 100
    # If neither: penalty = 0.
    
    # Move: Flip a random edge.
    # Accept if new_penalty < old_penalty.
    # If equal, accept with probability p (simulated annealing).
    
    def evaluate(bits, n):
        k4 = find_k4(bits, n)
        if k4:
            return 100, k4
        ind6 = find_ind6(bits, n)
        if ind6:
            return 100, ind6
        return 0, None

    best_n = 0
    best_edges = []
    
    # Try n=30 first, as it's the next integer after 29.
    # If we succeed, try n=31.
    
    for target_n in [30, 31]:
        if time_check():
            break
            
        # Initialize
        # Use Paley(29) as base for n=30?
        # Paley(29) is on vertices 0..28.
        # For n=30, we have vertices 0..29.
        
        # Let's just start random for simplicity, but biased towards Paley structure if possible.
        # Actually, random start for n=30 is fine.
        
        adj, edges = get_initial_graph(target_n)
        bits = to_bitsets(adj, target_n)
        
        current_penalty, _ = evaluate(bits, target_n)
        
        if current_penalty == 0:
            best_n = target_n
            best_edges = edges[:]
            break # Found valid graph for this n, try next? Or keep searching for higher?
            # The task says "Maximize n". So if we find 30, we should try 31.
            # But if we find 30, we should save it.
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        max_steps = 50000 # Adjust based on time
        
        step = 0
        while step < max_steps and not time_check():
            step += 1
            if step % 1000 == 0:
                T *= alpha
                if T < T_min:
                    T = T_min
            
            # Pick random edge to flip
            i = random.randint(0, target_n - 1)
            j = random.randint(0, target_n - 1)
            if i == j: continue
            
            # Flip edge (i, j)
            # Update adj
            old_val = adj[i][j]
            adj[i][j] = not old_val
            adj[j][i] = not old_val
            
            # Update bits
            if old_val:
                bits[i] &= ~(1 << j)
                bits[j] &= ~(1 << i)
                if [i, j] in edges: edges.remove([i, j])
                elif [j, i] in edges: edges.remove([j, i])
                # Edge list maintenance is slow. 
                # Let's just reconstruct edges at the end if valid.
                # Or maintain a set of edges.
            else:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
            
            new_penalty, _ = evaluate(bits, target_n)
            
            delta = new_penalty - current_penalty
            
            if delta < 0:
                current_penalty = new_penalty
                if new_penalty == 0:
                    # Found a valid graph!
                    # Reconstruct edges
                    final_edges = []
                    for u in range(target_n):
                        for v in range(u+1, target_n):
                            if adj[u][v]:
                                final_edges.append([u, v])
                    best_n = target_n
                    best_edges = final_edges
                    # We can break out of this n loop and try next n?
                    # Yes.
                    break
            else:
                # delta >= 0
                if random.random() < pow(2.71828, -delta / T):
                    # Accept
                    current_penalty = new_penalty
                else:
                    # Revert
                    adj[i][j] = old_val
                    adj[j][i] = old_val
                    if old_val:
                        bits[i] |= (1 << j)
                        bits[j] |= (1 << i)
                    else:
                        bits[i] &= ~(1 << j)
                        bits[j] &= ~(1 << i)
            
            if current_penalty == 0:
                final_edges = []
                for u in range(target_n):
                    for v in range(u+1, target_n):
                        if adj[u][v]:
                            final_edges.append([u, v])
                best_n = target_n
                best_edges = final_edges
                break

    # If we didn't find anything better than 29, return Paley(29)
    if best_n < 29:
        q = 29
        n = q
        squares = set(x * x % q for x in range(1, q))
        adj = [set() for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in squares:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}
