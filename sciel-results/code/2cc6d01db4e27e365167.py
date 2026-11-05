import time
from math import gcd
import numpy as np

N = 512
DEADLINE = 85.0  # Safety margin from 90s limit

def construct():
    start_time = time.time()
    
    # 1. Precompute LCMs and build conflict graph
    # Nodes are 0..N-1 representing values 1..N
    # Edge (u, v) exists if lcm(u+1, v+1) > N
    
    # For efficiency, we can precompute lcm for all pairs or compute on the fly.
    # N=512, so N^2 = 262,144 pairs. This is small.
    # We will use a list of bitmasks for the adjacency matrix.
    # adj[i] is a bitmask of nodes j such that lcm(i+1, j+1) > N.
    
    adj = [0] * N
    
    # Symmetry: if lcm(i,j) > N, then both i cannot be with j.
    # We only need to check i < j to set bits, but the graph is undirected.
    
    # Optimization: 
    # lcm(a, b) = a * b // gcd(a, b)
    # lcm(a, b) > N  <=>  a * b > N * gcd(a, b)
    
    for i in range(N):
        a = i + 1
        # We can restrict j to i+1..N-1 and set bits for both
        for j in range(i + 1, N):
            b = j + 1
            g = gcd(a, b)
            # Check if lcm > N
            # a * b // g > N
            # To avoid large integers (though Python handles big ints, 
            # a*b is at most 512*512 = 262144, which is fine)
            if (a * b) // g > N:
                # Conflict
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Solve Maximum Independent Set (MIS) on this graph
    # An independent set in the conflict graph is a valid subset S such that 
    # no pair in S has lcm > N.
    
    # We use a recursive branch-and-bound algorithm with bitsets.
    # This is a standard approach for small-to-medium N.
    
    # Sort nodes by degree descending to improve pruning? 
    # Or just use natural order. Tomita's algorithm often benefits from ordering.
    # Let's stick to natural order first, as N=512 is small enough that 
    # bit operations are fast.
    
    best_size = 0
    best_set = []
    
    # Global variables for the recursive solver
    # To avoid passing large lists around, we use a stack-based or recursive approach 
    # with bitmasks representing the candidate set R.
    
    # We'll implement a simple recursive B&B.
    # state: R (bitmask of candidates), size (current size of independent set)
    
    # To speed up, we can use a greedy coloring bound on the subgraph induced by R.
    # However, implementing full coloring bound in Python might be slow if not careful.
    # Given N=512, a simpler bound (size of R) might be sufficient if the graph is dense enough 
    # or if we prune well.
    
    # Let's try a standard recursive B&B with a greedy bound first.
    # If it's too slow, we can optimize.
    
    # Actually, for N=512, the conflict graph might be sparse or dense?
    # Many numbers have small LCMs. 
    # Example: powers of 2. lcm(2^a, 2^b) = 2^max(a,b) <= 256. So powers of 2 are compatible.
    # The graph is likely not extremely dense.
    
    # Let's use a more robust B&B.
    
    # We will use a list for the current set and a bitmask for candidates.
    
    def solve(R_mask, current_size, current_set):
        nonlocal best_size, best_set, start_time
        
        # Deadline check
        if time.time() - start_time > DEADLINE:
            return
        
        if R_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Bound: current_size + popcount(R_mask) <= best_size -> prune
        # Count bits in R_mask
        # Python int has no native popcount in older versions, but 3.10+ has bit_count()
        # Assuming Python 3.8+ usually available. Let's use bin(x).count('1') for compatibility or bit_count if available.
        try:
            rem = R_mask.bit_count()
        except AttributeError:
            rem = bin(R_mask).count('1')
            
        if current_size + rem <= best_size:
            return
        
        # Choose a vertex to branch on
        # Heuristic: pick the vertex with the highest degree in the induced subgraph R?
        # Or just pick the lowest index?
        # Picking highest degree in R often prunes faster.
        
        # To find highest degree in R:
        # deg(v) in R = popcount(adj[v] & R_mask)
        
        # Iterate through bits in R_mask
        # We can extract the first set bit, but let's find the max degree one.
        # This loop is O(N) per node.
        
        best_v = -1
        max_deg = -1
        
        temp_mask = R_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            
            # Calculate degree of v in R
            # adj[v] & R_mask
            deg = (adj[v] & R_mask).bit_count() if hasattr(int, 'bit_count') else bin(adj[v] & R_mask).count('1')
            
            if deg > max_deg:
                max_deg = deg
                best_v = v
                
            temp_mask ^= lsb
            
        if best_v == -1:
            # Should not happen if R_mask != 0
            return
            
        # Branch 1: Exclude best_v
        # R remains R_mask, but we remove best_v from consideration for this branch?
        # No, if we exclude best_v, it just means best_v is not in the set.
        # The remaining candidates are R_mask without best_v.
        R_excl = R_mask & ~(1 << best_v)
        solve(R_excl, current_size, current_set)
        
        # Branch 2: Include best_v
        # If we include best_v, we must exclude all its neighbors in R.
        # New R = R_mask & ~(adj[best_v] | (1 << best_v))
        # Note: adj[best_v] contains neighbors. We need to remove them and best_v itself.
        # Also, we add best_v to current_set.
        
        R_incl = R_mask & ~(adj[best_v] | (1 << best_v))
        current_set.append(best_v + 1) # Store value (1-indexed)
        solve(R_incl, current_size + 1, current_set)
        current_set.pop()
        
    # Initial call
    # R_mask includes all nodes 0..N-1
    initial_mask = (1 << N) - 1
    solve(initial_mask, 0, [])
    
    # Fallback if B&B didn't finish or found nothing (shouldn't happen, size >= 1)
    if best_size == 0:
        # Return a valid small set, e.g., {1}
        return [1]
        
    # The best_set contains 1-indexed values
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a // gcd(a, b)) * b > 512:
                print(f"Invalid pair: {a}, {b}")
                valid = False
                break
        if not valid:
            break
    if valid:
        print("All pairs valid.")
