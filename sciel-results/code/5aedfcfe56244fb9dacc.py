import time
import random
import sys

def count_violations(adj, n):
    """
    adj: list of two lists of ints. adj[c][u] is bitmask of neighbors of u in color c.
    n: number of vertices.
    Returns number of monochromatic K2,2,2 subgraphs.
    """
    total = 0
    for c in [0, 1]:
        A = adj[c]
        for u in range(n):
            for v in range(u + 1, n):
                # Common neighbors of u and v in color c
                common = A[u] & A[v]
                # We need to count K2,2 in the subgraph induced by 'common'
                # Vertices in this subgraph are the set bits in 'common'
                # Let H be this subgraph.
                # Number of K2,2 in H is sum_{i<j in V(H)} binom(|N_H(i) cap N_H(j)|, 2)
                
                # Optimization: if popcount(common) < 4, no K2,2 possible
                if common.bit_count() < 4:
                    continue
                
                # Extract vertices
                # This list comprehension is O(n)
                v_list = [i for i in range(n) if (common >> i) & 1]
                L = len(v_list)
                
                count_c4 = 0
                # Iterate pairs in v_list
                # Pre-compute masked adjacency for these vertices?
                # A[i] & common is N_H(i)
                # Let's just compute on the fly.
                for a_idx in range(L):
                    i = v_list[a_idx]
                    Ai = A[i] & common
                    for b_idx in range(a_idx + 1, L):
                        j = v_list[b_idx]
                        # N_H(j) is A[j] & common
                        # Intersection is Ai & (A[j] & common)
                        # Since Ai is already masked, Ai & A[j] is the intersection within H
                        # Wait, Ai & A[j] might have bits outside common?
                        # No, Ai = A[i] & common. So Ai is subset of common.
                        # So Ai & A[j] is subset of common.
                        # So (Ai & A[j]).bit_count() is correct.
                        cnt = (Ai & A[j]).bit_count()
                        if cnt >= 2:
                            count_c4 += cnt * (cnt - 1) // 2
                total += count_c4
    return total // 3

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Try n from 29 down to 17
    # If we find a valid one, return it.
    # We prioritize 29.
    
    for n in range(29, 16, -1):
        if time.time() > deadline:
            break
            
        # If n <= 17, we have a known construction (Paley)
        if n == 17:
            squares = {x * x % 17 for x in range(1, 17)}
            return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}
        
        # For n > 17, use SA
        best_colors = None
        best_violations = float('inf')
        
        # Multiple restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize random coloring
            # colors list of length n*(n-1)//2
            # adj[0] and adj[1] bitmasks
            adj = [[0] * n for _ in range(2)]
            colors = [0] * (n * (n - 1) // 2)
            
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = random.randint(0, 1)
                    colors[idx] = c
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            
            v = count_violations(adj, n)
            
            if v == 0:
                return {'n': n, 'colors': colors}
                
            if v < best_violations:
                best_violations = v
                best_colors = colors[:]
                
            # Simulated Annealing
            T = 1.0
            # Run for a short time or fixed steps
            steps_in_sa = 2000
            for step in range(steps_in_sa):
                if time.time() > deadline:
                    break
                    
                # Pick random edge
                idx = random.randint(0, len(colors) - 1)
                # Find i, j from idx
                # idx = i*n - i*(i+1)/2 + (j - i - 1)
                # Inverse is tricky, so let's just iterate or use a lookup table
                # For n=29, edges=406.
                # We can precompute a list of (i, j, idx)
                
                # Let's precompute edge list for this n outside the restart loop?
                # No, n changes. But inside n loop, it's constant.
                # Let's generate edge_map once per n.
                # Wait, I'm inside the restart loop.
                # Better to precompute edge_map for current n before restarts.
                
                # Let's restructure.
                pass # Placeholder, will refactor below

            # If we broke out of SA loop due to time, break restarts
            if time.time() > deadline:
                break

        # If we exited SA loop without finding 0, but have best_colors
        # We can check if best_colors is valid (it might have been improved)
        # But we tracked best_violations. If best_violations == 0, we would have returned.
        # So if we get here, we didn't find a solution for this n in this restart batch.
        # We can try more restarts if time permits.
        
    # Fallback
    # Return the best valid one found? 
    # The loop returns immediately if valid.
    # If we finish the n loop without returning, we must return something.
    # The task says "Maximize n". If we couldn't find 29, we should return the largest n we *did* find valid.
    # But my loop structure breaks if time is up.
    # I should keep track of the best valid n found.
    
    # Refactored construct
    return {'n': 17, 'colors': []} # Placeholder

# Let's write the actual code properly.
