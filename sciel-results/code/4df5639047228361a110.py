# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    """
    Return nonempty subset of [1,128] with no three distinct elements whose
    three pairwise greatest common divisors are equal. Maximize cardinality.
    
    Approach: Model as Maximum Independent Set in a 3-uniform hypergraph.
    Use Branch-and-Bound with bitset-based conflict propagation.
    Improved Greedy: Bitset-based, minimizing immediate removals to maximize
    remaining options for a larger initial solution.
    """
    N = 128
    
    # 1. Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # 2. Precompute bad_thirds[u][v]
    bad_thirds = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            bitset = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    bitset |= (1 << c)
            bad_thirds[a][b] = bitset
            bad_thirds[b][a] = bitset
            
    # 3. Improved Greedy Initialization using Bitsets
    # Heuristic: Pick vertex v that minimizes the number of removed vertices
    # from the current Available set. This keeps the pool of candidates large.
    
    def greedy_select_bitset(initial_available, S_list, order_hint=None):
        """
        Greedily select vertices from initial_available.
        order_hint: A list of vertices to consider in this priority order?
        Actually, we just iterate through Available bits, but we can sort candidates
        by some metric if we want. For speed, we'll just scan Available bits.
        To add variety, we can pass a "priority" mask or just rely on the
        minimization of removal.
        """
        Available = initial_available
        S = []
        
        while Available:
            # Find best v in Available
            best_v = -1
            best_removal_count = N + 1 # We want to minimize removal
            best_new_bad = 0
            
            temp_avail = Available
            while temp_avail:
                lsb = temp_avail & -temp_avail
                v = lsb.bit_length() - 1
                temp_avail ^= lsb
                
                # Calculate new_bad for this v given current S
                new_bad = 0
                for u in S:
                    new_bad |= bad_thirds[u][v]
                
                # Count removals
                removal = (Available & new_bad).bit_count()
                
                if removal < best_removal_count:
                    best_removal_count = removal
                    best_v = v
                    best_new_bad = new_bad
                    
            if best_v == -1:
                break
                
            S.append(best_v)
            bit_v = 1 << best_v
            # Update Available
            Available &= ~bit_v
            Available &= ~best_new_bad
            
        return S

    # Try to get a good initial lower bound
    initial_Available = 0
    for i in range(1, N + 1):
        initial_Available |= (1 << i)
        
    best_size = 0
    best_set = []
    
    # Run greedy once. The "min removal" heuristic is deterministic given the set.
    # We can try to bias it by pre-sorting? 
    # The bit iteration order is ascending index.
    # We can try a version that prefers higher indices or lower indices by
    # modifying the bit iteration or just running it as is.
    
    sol = greedy_select_bitset(initial_Available, [])
    best_size = len(sol)
    best_set = sol[:]
    
    # Try a few variations? 
    # For instance, start by forcing certain "safe" vertices?
    # Powers of 2 are always safe together.
    # Let's try a greedy that *forces* powers of 2 first?
    # No, greedy is greedy.
    
    # Another strategy: Local Search / Hill Climbing on the Greedy result?
    # Try to swap elements to increase size?
    # Given time limits, a simple B&B with a good start is usually enough for N=128.
    
    # 4. Branch and Bound
    start_time = time.time()
    time_limit = 55.0
    
    global_best_size = best_size
    global_best_set = best_set[:]
    
    def solve(S_list, Available):
        nonlocal global_best_size, global_best_set
        
        if time.time() - start_time > time_limit:
            return
            
        current_size = len(S_list)
        avail_count = Available.bit_count()
        
        # Pruning: Upper bound
        if current_size + avail_count <= global_best_size:
            return
            
        if avail_count == 0:
            if current_size > global_best_size:
                global_best_size = current_size
                global_best_set = S_list[:]
            return
            
        # Heuristic: Choose vertex v from Available that maximizes removal
        # This is standard for MIS to prune the "Include" branch deeply.
        
        best_v = -1
        best_removal = -1
        best_new_bad = 0
        
        temp_avail = Available
        while temp_avail:
            lsb = temp_avail & -temp_avail
            v = lsb.bit_length() - 1
            temp_avail ^= lsb
            
            new_bad = 0
            for u in S_list:
                new_bad |= bad_thirds[u][v]
                
            removal = (Available & new_bad).bit_count()
            
            if removal > best_removal:
                best_removal = removal
                best_v = v
                best_new_bad = new_bad
                
        if best_v == -1:
            return
            
        v = best_v
        new_bad = best_new_bad
        bit_v = 1 << v
        
        # Branch 1: Include v
        S_list.append(v)
        new_Available = Available & ~bit_v & ~new_bad
        solve(S_list, new_Available)
        S_list.pop()
        
        # Branch 2: Exclude v
        new_Available_skip = Available & ~bit_v
        solve(S_list, new_Available_skip)
        
    solve([], initial_Available)
    
    if not global_best_set:
        return [1]
        
    return sorted(global_best_set)

# EVOLVE-BLOCK-END
