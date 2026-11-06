"""Program for large integer sets avoiding equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    """
    Return nonempty subset of [1,128] with no three distinct elements whose
    three pairwise greatest common divisors are equal. Maximize cardinality.
    
    Approach: Model as Maximum Independent Set in a 3-uniform hypergraph.
    Use Branch-and-Bound with bitset-based conflict propagation.
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
    # bad_thirds[u][v] is a bitset of all w such that {u, v, w} is a bad triple.
    # We assume u < v for indexing, but we can just make it a full (N+1)x(N+1) matrix for simplicity.
    bad_thirds = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            bitset = 0
            for c in range(b + 1, N + 1):
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    bitset |= (1 << c)
            # Also check c < a and a < c < b? 
            # The triple is unordered. The condition is symmetric.
            # My loop above only checks c > b.
            # Let's just iterate all c != a, c != b.
            bitset = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    bitset |= (1 << c)
            bad_thirds[a][b] = bitset
            bad_thirds[b][a] = bitset
            
    # 3. Greedy Initialization
    # Try a few greedy orders to get a good lower bound for best_size
    def greedy_select(order):
        S = []
        # Maintain a set of "forbidden" vertices based on S
        # forbidden[w] is True if w cannot be added to S
        # A vertex w is forbidden if there exist u, v in S such that {u,v,w} is bad.
        # We can maintain a bitset of forbidden vertices.
        forbidden = 0
        for x in order:
            if (forbidden >> x) & 1:
                continue
            # Check if x can be added
            # If we add x, new forbidden vertices are those z such that exists u in S with {u, x, z} bad.
            # This is OR_{u in S} bad_thirds[u][x].
            # Also, x itself is removed from available (not strictly forbidden for others, just taken).
            
            # Quick check: is x already in S? No, we iterate unique order.
            # Is x forbidden by previous choices? Yes, checked above.
            
            # Calculate new forbidden
            new_forbidden_part = 0
            for u in S:
                new_forbidden_part |= bad_thirds[u][x]
            
            # If adding x doesn't create immediate conflicts with itself? 
            # The definition of bad_thirds[u][x] includes all z != u, x.
            # So new_forbidden_part contains all z that would form a bad triple with x and some u in S.
            
            # We can add x if x is not in new_forbidden_part? 
            # Actually, bad_thirds[u][x] does NOT include x (since z != x).
            # So x is not in new_forbidden_part.
            # However, we must ensure that no pair in S already forbids x.
            # That is exactly what `forbidden` tracks.
            
            # So we can always add x if it's not in `forbidden`.
            S.append(x)
            forbidden |= new_forbidden_part
            forbidden |= (1 << x) # Mark x as used/removed from consideration
            
        return S

    # Try ascending, descending, and sorted by "conflict count"
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1))
    ]
    
    # Calculate conflict count for a heuristic order
    conflict_count = [0] * (N + 1)
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            # Count how many w form a bad triple with u and v
            # This is just popcount(bad_thirds[u][v])
            cnt = bad_thirds[u][v].bit_count()
            conflict_count[u] += cnt
            conflict_count[v] += cnt
            
    orders.append(sorted(range(1, N + 1), key=lambda x: conflict_count[x]))
    
    best_size = 0
    best_set = []
    
    for order in orders:
        sol = greedy_select(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
            
    # 4. Branch and Bound
    start_time = time.time()
    time_limit = 55.0  # Leave buffer for overhead
    
    # Global variables for the search
    global_best_size = best_size
    global_best_set = best_set[:]
    
    def solve(S_list, Available, current_bad):
        nonlocal global_best_size, global_best_set
        
        # Time check
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
        # We want to pick v such that adding it removes the most vertices from Available.
        # Removal count = (Available & (current_bad | OR_{u in S} bad_thirds[u][v])).bit_count()
        # Since current_bad is constant, we maximize (Available & OR_{u in S} bad_thirds[u][v]).bit_count()
        
        best_v = -1
        best_incremental_bad = 0
        best_removal = -1
        
        temp_avail = Available
        while temp_avail:
            lsb = temp_avail & -temp_avail
            v = lsb.bit_length() - 1
            temp_avail ^= lsb
            
            # Calculate incremental bad for this v
            incremental_bad = 0
            for u in S_list:
                incremental_bad |= bad_thirds[u][v]
                
            # Total bad if we pick v is current_bad | incremental_bad
            # Vertices removed from Available are those in Available that are in total_bad
            # Note: v itself is removed, but v is not in bad_thirds[u][v] (z != v).
            # So we must explicitly account for v.
            # removed = (Available & (current_bad | incremental_bad)) | (1 << v)
            # But we only care about the count for heuristic.
            # count = ((Available & (current_bad | incremental_bad)).bit_count() + 1) 
            # Actually, if v is in current_bad, it's already counted? 
            # current_bad contains vertices that form bad triples with pairs in S.
            # v might be in current_bad.
            # The set of vertices that become unavailable is {v} U (Available & (current_bad | incremental_bad)).
            
            removal = ((Available & (current_bad | incremental_bad)).bit_count())
            # v is always removed. If v was in current_bad, it's counted above. If not, it's not.
            # So total removed = removal + (1 if not ((current_bad | incremental_bad) >> v) & 1 else 0)
            # Simpler: just compute the bitset of removed vertices and count bits.
            # But we want speed.
            # Let's just use the count of available vertices that conflict.
            # If v doesn't conflict with anything in S, incremental_bad is 0.
            # Then removal is count of Available & current_bad.
            
            # For heuristic, maximizing removal is good.
            if removal > best_removal:
                best_removal = removal
                best_v = v
                best_incremental_bad = incremental_bad
                
        if best_v == -1:
            return
            
        v = best_v
        incremental_bad = best_incremental_bad
        bit_v = 1 << v
        
        # Branch 1: Include v
        S_list.append(v)
        new_current_bad = current_bad | incremental_bad
        # New Available: remove v, and remove any vertex in new_current_bad
        new_Available = Available & ~bit_v & ~new_current_bad
        solve(S_list, new_Available, new_current_bad)
        S_list.pop()
        
        # Branch 2: Exclude v
        # current_bad remains the same
        new_Available_skip = Available & ~bit_v
        solve(S_list, new_Available_skip, current_bad)
        
    # Initial call
    initial_Available = 0
    for i in range(1, N + 1):
        initial_Available |= (1 << i)
        
    # Start with empty set and no bad triples
    solve([], initial_Available, 0)
    
    if not global_best_set:
        return [1]
        
    return sorted(global_best_set)

# EVOLVE-BLOCK-END
