import time
import random
from collections import defaultdict

NMAX = 256
RANGE = list(range(2, NMAX + 1))
N = len(RANGE)
# Map value to index 0..254
val_to_idx = {v: i for i, v in enumerate(RANGE)}
idx_to_val = RANGE

# Precompute divisors for all numbers up to 256
# divisors[n] is a list of divisors d of n such that 2 <= d <= 256
divisors = [[] for _ in range(NMAX + 1)]
for n in range(2, NMAX + 1):
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            divisors[n].append(d)
            if d != n // d:
                divisors[n].append(n // d)
    divisors[n].append(n)

# Precompute conflict sets
# conflict_pair[b_idx][c_idx] = set of a_indices such that a divides b*c
# Since b and c are distinct, we only need to store this for b < c or just compute on the fly?
# b*c can be up to 256*255 = 65280.
# We can precompute for each product P = b*c, the set of a in [2, 256] that divide P.
# Let's map product to list of indices of divisors in our range.
# There are 255*254/2 ~ 32000 pairs.
# We can just compute divisors of b*c on the fly if needed, but precomputation is safer for speed.

# Let's precompute `divs_of_product[b_idx][c_idx]` as a bitmask or list of indices.
# Given N=255, a bitmask is 255 bits, which fits in a Python int (arbitrary precision).
# This will make intersection and counting very fast.

divs_mask = [[0] * N for _ in range(N)]
for i in range(N):
    b = idx_to_val[i]
    for j in range(N):
        if i == j:
            continue
        c = idx_to_val[j]
        prod = b * c
        mask = 0
        for d in range(2, NMAX + 1):
            if prod % d == 0:
                mask |= (1 << val_to_idx[d])
        divs_mask[i][j] = mask

# Also precompute for each a, the pairs (b,c) that it conflicts with?
# No, the greedy removal works by counting conflicts.
# Conflict count for a is sum over all pairs (b,c) in S\{a} of [a divides b*c].
# This is hard to update incrementally if we remove b or c.
# However, we can compute the "degree" of a node a in the current hypergraph H(S).
# Degree(a) = number of pairs (b,c) in S, b!=c, b!=a, c!=a, such that a | b*c.

def compute_degrees(S_mask):
    """
    S_mask: int bitmask of active vertices.
    Returns list of degrees for each vertex.
    """
    degrees = [0] * N
    # Iterate over all pairs (b, c) in S
    # If a divides b*c, increment degree[a]
    # To do this efficiently, we can iterate over b in S, c in S (b!=c)
    # and for each pair, find all a in S that divide b*c, and increment their degree.
    # But we only care about a in S.
    # So: for each pair (b,c) in S, count how many a in S are in divs_mask[b][c].
    # Then add that count to... wait.
    # The definition of degree of a is the number of edges containing a.
    # An edge is a set {a,b,c}.
    # So for a fixed a, degree(a) = number of pairs (b,c) in S\{a} such that a | b*c.
    
    # Alternative:
    # For each a in S:
    #   count = 0
    #   for each b in S\{a}:
    #       for each c in S\{a,b}:
    #           if a | b*c: count++
    #   degree[a] = count
    
    # This is O(N^3). With N=255, N^3 ~ 16 million.
    # In Python, 16M operations might take ~1-2 seconds.
    # We do this in a loop. We need it to be fast.
    
    # Let's try to optimize.
    # For a fixed a, the condition is a | b*c.
    # This is equivalent to: for all prime factors p, v_p(a) <= v_p(b) + v_p(c).
    
    # Let's stick to the bitmask approach for checking, but counting is the issue.
    # We can precompute for each a, a list of pairs (b,c) that it conflicts with?
    # No, that's too much memory.
    
    # Let's just do the O(N^3) check but optimized with bitmasks?
    # Actually, we can compute degrees in O(N^2 * (N/word_size)) using bit operations?
    # For each pair (b,c), let M = divs_mask[b][c].
    # The number of a in S that divide b*c is popcount(S_mask & M).
    # But this gives the number of a's that form an edge with THIS specific pair (b,c).
    # The degree of a specific node a is the number of edges containing a.
    # So, Degree(a) = Sum_{b,c in S\{a}} [a | b*c].
    
    # Let's swap the loops.
    # For each a in S:
    #   deg = 0
    #   for b in S\{a}:
    #      for c in S\{a,b}:
    #         if a | b*c: deg++
    #   degrees[a] = deg
    
    # To speed this up, we can precompute for each a and b, the set of c such that a | b*c.
    # valid_c[a][b] = bitmask of c such that a | b*c.
    # Then for fixed a, b: count = popcount(S_mask & valid_c[a][b]).
    # Then Degree(a) = Sum_{b in S\{a}} count.
    
    # Precomputation:
    # valid_c_mask[a_idx][b_idx] = bitmask of c indices such that a | b*c.
    # Note: c must be distinct from a and b?
    # The problem states a,b,c are distinct.
    # So when counting edges containing a, we consider pairs (b,c) where b!=a, c!=a, b!=c.
    # If we use valid_c_mask, it includes all c. We must exclude c=a and c=b.
    
    pass # We will implement the O(N^3) with early termination or optimization in the main loop if needed, 
         # but let's try to precompute valid_c_mask first.

# Precompute valid_c_mask
# valid_c_mask[a_idx][b_idx] = bitmask of c in [2..256] such that a | b*c.
valid_c_mask = [[0] * N for _ in range(N)]
for i in range(N):
    a = idx_to_val[i]
    for j in range(N):
        b = idx_to_val[j]
        mask = 0
        prod = b
        # We need c such that a | b*c.
        # This is equivalent to (a / gcd(a,b)) | c.
        g = __import__('math').gcd(a, b)
        req = a // g
        # c must be a multiple of req.
        for c_val in range(2, NMAX + 1):
            if c_val % req == 0:
                mask |= (1 << val_to_idx[c_val])
        valid_c_mask[i][j] = mask

def get_degrees(S_mask):
    degrees = [0] * N
    # For each a in S
    a_list = [i for i in range(N) if (S_mask >> i) & 1]
    for i in a_list:
        a_idx = i
        deg = 0
        # For each b in S, b != a
        for j in a_list:
            if j == i:
                continue
            b_idx = j
            # Count c in S such that c != a, c != b, and a | b*c
            # Mask of valid c's: valid_c_mask[a_idx][b_idx]
            # Exclude a (bit i) and b (bit j)
            valid_mask = valid_c_mask[a_idx][b_idx]
            # We want c in S_mask, c != i, c != j
            # So: S_mask & valid_mask & ~(1<<i) & ~(1<<j)
            count_mask = S_mask & valid_mask & ~(1 << i) & ~(1 << j)
            deg += bin(count_mask).count('1')
        degrees[a_idx] = deg
    return degrees

def greedy_removal(deadline, tie_breaker='largest'):
    S_mask = (1 << N) - 1
    # Initialize degrees
    degrees = get_degrees(S_mask)
    
    while time.time() < deadline:
        # Find node with max degree
        # Tie breaker: if degrees are equal, pick based on tie_breaker
        max_deg = -1
        best_idx = -1
        for i in range(N):
            if not (S_mask >> i) & 1:
                continue
            deg = degrees[i]
            if deg > max_deg:
                max_deg = deg
                best_idx = i
            elif deg == max_deg:
                # Tie breaking
                if best_idx == -1:
                    best_idx = i
                else:
                    val_i = idx_to_val[i]
                    val_best = idx_to_val[best_idx]
                    if tie_breaker == 'largest':
                        if val_i > val_best:
                            best_idx = i
                    elif tie_breaker == 'smallest':
                        if val_i < val_best:
                            best_idx = i
                    else: # random
                        if random.random() < 0.5:
                            best_idx = i
        
        if best_idx == -1:
            break
            
        # If max degree is 0, we can stop?
        # If degree is 0, removing it doesn't remove any conflicts.
        # But we want to MAXIMIZE size.
        # The greedy removal for Max Independent Set usually removes nodes with HIGHEST degree.
        # If all degrees are 0, the graph is empty (no conflicts), so we can keep all.
        # Wait, the greedy removal algorithm for MIS is:
        # While V is not empty:
        #   Pick v with max degree
        #   Add v to Independent Set? NO.
        #   The standard greedy for MIS is:
        #   Add v to IS if no neighbors in IS.
        #   OR
        #   Remove v from V, add to IS if...
        
        # Let's re-evaluate the strategy.
        # "Greedy removal" usually refers to the algorithm for finding a large independent set by starting with all vertices and removing vertices that have edges.
        # Specifically:
        # 1. Start with S = V.
        # 2. While there is an edge (conflict) in S:
        #    a. Find the vertex v in S that is part of the most edges in S.
        #    b. Remove v from S.
        # 3. Return S.
        
        # This guarantees S is an independent set (no conflicts).
        # Does it maximize size? Not necessarily, but it's a good heuristic.
        
        if max_deg == 0:
            break
            
        # Remove best_idx
        S_mask &= ~(1 << best_idx)
        
        # Update degrees
        # This is expensive. O(N^2).
        # We can just recompute?
        # If N=255, recomputing degrees takes ~1ms?
        # Let's check.
        degrees = get_degrees(S_mask)
        
    # Convert mask to list
    res = []
    for i in range(N):
        if (S_mask >> i) & 1:
            res.append(idx_to_val[i])
    return res

def local_search(S, deadline):
    """
    Try to improve S by swapping elements.
    S is a list of ints.
    """
    S_set = set(S)
    S_mask = 0
    for v in S:
        S_mask |= (1 << val_to_idx[v])
    
    best_S = S[:]
    best_size = len(S)
    
    # Get removed elements
    all_elements = set(RANGE)
    removed = list(all_elements - S_set)
    
    # Precompute degrees for current S
    degrees = get_degrees(S_mask)
    
    # We want to try to add a removed element x.
    # If we add x, it might create conflicts.
    # To add x, we must remove all y in S that conflict with x?
    # Or swap x with y?
    
    # Strategy:
    # For each x in removed:
    #   Find all y in S such that x conflicts with y (i.e., there exists z in S such that x|y*z or y|x*z or z|x*y).
    #   Actually, x conflicts with the set S if there exists a pair (b,c) in S such that x|b*c, OR
    #   there exists b,c in S such that b|x*c (which implies x and c are "close"), etc.
    #   
    # Let's define conflict(x, S) = number of triples (a,b,c) in S U {x} that are invalid and involve x.
    # If conflict(x, S) == 0, we can add x.
    # If conflict(x, S) > 0, we can try to remove one of the conflicting elements and add x.
    
    # To speed this up, we can just try to add x if possible.
    # If not, try to swap x with a "low degree" element in S.
    
    for _ in range(100): # Limit iterations
        if time.time() > deadline:
            break
            
        improved = False
        # Shuffle removed to randomize
        random.shuffle(removed)
        
        for x in removed:
            x_idx = val_to_idx[x]
            
            # Check if x can be added directly
            # This requires checking all pairs in S.
            # We can use the mask logic.
            # x can be added if for all b,c in S, b!=c:
            # x does not divide b*c
            # b does not divide x*c
            # c does not divide x*b
            
            # Let's check x | b*c
            # We need to check if there exists b,c in S such that x | b*c.
            # This is equivalent to: is there any pair (b,c) in S such that b*c is a multiple of x?
            # We can precompute for each x, the set of pairs (b,c) that it conflicts with?
            # No, let's just check.
            
            # Optimization:
            # If x is large, it's less likely to divide b*c.
            # If x is small, it's more likely.
            
            # Let's just do a quick check using the valid_c_mask logic in reverse?
            # No, let's just iterate.
            
            can_add = True
            # Check x | b*c
            # For each b in S:
            #   For each c in S, c!=b:
            #      if b*c % x == 0: can_add = False; break
            
            # This is O(N^2).
            # We can optimize by checking if x divides b*c.
            # Let g = gcd(x, b). Then x/g must divide c.
            
            S_list = list(S_set)
            
            # Check 1: x | b*c
            for b in S_list:
                g = __import__('math').gcd(x, b)
                req = x // g
                # Is there a c in S, c!=b, such that c % req == 0?
                # We can precompute counts of multiples of req in S?
                # Too expensive to precompute for all req.
                # Just check.
                found_c = False
                for c in S_list:
                    if c == b:
                        continue
                    if c % req == 0:
                        found_c = True
                        break
                if found_c:
                    can_add = False
                    break
            
            if not can_add:
                # Check b | x*c
                for b in S_list:
                    g = __import__('math').gcd(b, x)
                    req = b // g
                    found_c = False
                    for c in S_list:
                        if c == b:
                            continue
                        if c % req == 0:
                            found_c = True
                        break
                    if found_c:
                        can_add = False
                        break
                if not can_add:
                    # Check c | x*b (same as b | x*c, just swapped roles, already covered if S is symmetric in check? 
                    # Wait, b | x*c and c | x*b are different conditions.
                    # But if we iterate all b and all c, we cover both.
                    # Actually, my loop "for b in S: for c in S" covers all ordered pairs.
                    # So if I check "x | b*c" for all b,c, and "b | x*c" for all b,c, I cover everything.
                    pass
            
            if can_add:
                S_set.add(x)
                S_mask |= (1 << x_idx)
                removed.remove(x)
                if len(S_set) > best_size:
                    best_S = list(S_set)
                    best_size = len(S_set)
                    improved = True
                break # Restart loop? Or continue?
                
            # If cannot add, try swap
            # Find y in S with lowest degree that conflicts with x?
            # Or just pick a random y in S and try to swap.
            
            # Pick y with min degree
            min_deg = float('inf')
            y_idx = -1
            for y in S_list:
                deg = degrees[val_to_idx[y]]
                if deg < min_deg:
                    min_deg = deg
                    y_idx = val_to_idx[y]
            
            if y_idx != -1:
                y = idx_to_val[y_idx]
                # Try swapping y with x
                # New S = S - {y} + {x}
                # Check if x can be added to S - {y}
                S_temp_set = S_set - {y}
                S_temp_list = list(S_temp_set)
                
                can_add_x_to_temp = True
                for b in S_temp_list:
                    g = __import__('math').gcd(x, b)
                    req = x // g
                    for c in S_temp_list:
                        if c == b: continue
                        if c % req == 0:
                            can_add_x_to_temp = False
                            break
                    if not can_add_x_to_temp: break
                
                if can_add_x_to_temp:
                    # Also need to check b | x*c and c | x*b for b,c in S_temp
                    for b in S_temp_list:
                        g = __import__('math').gcd(b, x)
                        req = b // g
                        for c in S_temp_list:
                            if c == b: continue
                            if c % req == 0:
                                can_add_x_to_temp = False
                                break
                        if not can_add_x_to_temp: break
                        
                    if can_add_x_to_temp:
                        # Perform swap
                        S_set.remove(y)
                        S_set.add(x)
                        S_mask &= ~(1 << y_idx)
                        S_mask |= (1 << x_idx)
                        removed.remove(x)
                        removed.append(y)
                        # Update degrees?
                        degrees = get_degrees(S_mask)
                        improved = True
                        break
        
        if not improved:
            break
            
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best = []
    best_size = 0
    
    # Run greedy removal with different tie breakers
    for tb in ['largest', 'smallest', 'random']:
        if time.time() > deadline - 10:
            break
        c = greedy_removal(deadline - 10, tb)
        if len(c) > best_size:
            best = c
            best_size = len(c)
            
    # Local search on best
    if time.time() < deadline - 5:
        c = local_search(best, deadline - 5)
        if len(c) > best_size:
            best = c
            best_size = len(c)
            
    # Final check and return
    # Ensure valid
    res = list(best)
    # Verify
    n = len(res)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            for k in range(n):
                if i == k or j == k: continue
                if (res[j] * res[k]) % res[i] == 0:
                    # Invalid, fallback
                    # Find a valid subset?
                    # Just return a small valid set
                    return [256]
    return res

# Run construct to test? No, just define.
