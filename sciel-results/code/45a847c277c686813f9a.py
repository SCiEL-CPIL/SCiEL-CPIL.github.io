import time
import math
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 88.0  # Safety margin from 90s limit

    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCD matrix
    gcd_mat = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            gcd_mat[i][j] = math.gcd(elements[i], elements[j])

    # Build the 2-uniform conflict graph.
    # An edge exists between u and v if there exists a w such that {u, v, w} is a bad triple.
    # Wait, the constraint is: no three distinct elements whose three pairwise gcds are equal.
    # Let E be the set of edges in the conflict graph.
    # If u and v are connected by an edge in this graph, it means they CANNOT both be in the independent set?
    # No. The constraint is on TRIPLES.
    # If we select a set S, it is valid if for all distinct a,b,c in S, it is NOT the case that 
    # gcd(a,b) == gcd(b,c) == gcd(c,a).
    
    # The current program treats it as a 3-uniform hypergraph.
    # The prompt suggests: "Formulate the problem as a maximum weight independent set in a 2-uniform conflict graph 
    # where an edge exists between u and v if they are part of any bad triple".
    
    # Let's analyze this reduction.
    # If u and v are in a bad triple {u, v, w}, then we cannot have {u, v, w} in our set.
    # This is a 3-uniform constraint.
    # Can we reduce it to a 2-uniform one?
    # Generally, no. A 3-uniform constraint cannot be represented exactly as a 2-uniform constraint on the same nodes 
    # without adding auxiliary nodes or changing the problem structure significantly (e.g., Helly properties, which don't apply here).
    
    # HOWEVER, the prompt's "Strategy to try" says: 
    # "Formulate the problem as a maximum weight independent set in a 2-uniform conflict graph where an edge exists 
    # between u and v if they are part of any bad triple, and solve it using a specialized branch-and-bound algorithm..."
    
    # This specific reduction is INVALID for this problem. 
    # If u and v are in a bad triple with w, u and v can still be in the set together, as long as w is not.
    # If we put an edge between u and v, we forbid {u, v} entirely, which is too restrictive.
    
    # Let's re-read carefully: "where an edge exists between u and v if they are part of any bad triple".
    # This implies the proposed strategy in the directive might be flawed or I am misinterpreting "part of any bad triple".
    # Maybe it means: Construct a graph where nodes are PAIRS? No, that's complex.
    
    # Let's look at the inspirations.
    # Inspiration 1: CSP with Backtracking.
    # Inspiration 2: Structural initial construction + Local Search.
    # Inspiration 3: Structural Analysis.
    
    # The current program is a greedy hypergraph approach.
    # The directive asks for a "STRUCTURAL change".
    # The suggested strategy in the directive text is: "Formulate ... as MWIS in 2-uniform ...".
    # If I follow the directive literally, I will produce a suboptimal or incorrect result because the reduction is wrong.
    # BUT, the directive also says: "Idea target: design the new strategy so that its result answers this open question: Does operator 'diff' change the chance that a child improves its island...".
    # And "STAGNATION: ... Make a STRUCTURAL change".
    
    # Perhaps the "2-uniform" hint refers to a different modeling?
    # Actually, there is a known trick: If the bad triples are "cliques" in some sense, or if the structure allows it.
    # But for GCD equality, it's not simple.
    
    # Let's try a different structural change: **Exact Branch and Bound on the 3-uniform hypergraph**.
    # The current program is Greedy + Local Search.
    # A Branch and Bound (B&B) algorithm can find the *maximum* independent set if the search space is small enough or pruned effectively.
    # N=128 is too large for naive B&B.
    # However, we can use **Bitset-based B&B** with strong upper bounds.
    
    # Upper Bound Strategy:
    # The maximum independent set size is bounded by the size of the largest independent set in the "conflict graph" 
    # where we connect nodes that CANNOT be together?
    # No, that's the 2-uniform reduction which is too strong (forbids pairs).
    
    # Let's try **Simulated Annealing** with a very good initial state and a robust acceptance criterion, 
    # but optimized with bitsets for speed. This is a different optimization method (stochastic global optimization vs deterministic greedy/local).
    
    # OR, let's try **Maximum Clique** on the complement graph?
    # The complement of the "bad triple" constraint is not a graph.
    
    # Let's go with **Branch and Bound with Bitsets** specifically designed for this 3-uniform case, 
    # but using a "Core" reduction.
    # 1. Identify a "Core" set of elements that are highly connected (part of many bad triples).
    # 2. Elements outside the core that don't form bad triples with each other or the core can be added freely?
    # No, they might form bad triples with two elements in the core.
    
    # Let's try a different structural approach: **Integer Linear Programming (ILP)** relaxation?
    # No SciPy/ILP solver guaranteed (only standard lib, numpy, scipy). SciPy has `linprog` but it's for LP, not ILP.
    
    # Let's stick to the most robust structural change: **Bitset-based Backtracking with Forward Checking and MRV**,
    # similar to Inspiration 1, but highly optimized with Python bit operations.
    # The current program uses Greedy. Backtracking is fundamentally different: it explores the solution space systematically with pruning.
    
    # Algorithm:
    # 1. Precompute `conflict_mask[i]`? No, we need triples.
    # 2. We maintain a set `S` (selected), `C` (candidates), `F` (forbidden).
    # 3. At each step, pick a candidate `v`.
    #    - If we include `v`:
    #       - Check if adding `v` creates any bad triple with pairs in `S`.
    #       - If yes, `v` cannot be included (branch pruned).
    #       - If no, add `v` to `S`.
    #       - Update `C`: Remove any `u` in `C` that would form a bad triple with `v` and some `w` in `S`?
    #         Actually, if `u` and `w` in `S` form a bad triple with `v`, then `u` cannot be added later if `v` is in `S`?
    #         No, if `v` is in `S`, and we later add `u`, we check `{u, v, w}`.
    #         The constraint is global.
    #    
    #    Let's define the state by `S` (selected) and `C` (available candidates).
    #    A candidate `c` is "blocked" by `S` if there exists `s1, s2` in `S` such that `{c, s1, s2}` is a bad triple.
    #    If `c` is blocked, it cannot be added.
    #    
    #    So, we can precompute for each pair `(s1, s2)` in `S`, which candidates are blocked.
    #    This is expensive to update dynamically.
    
    # Alternative: **Randomized Local Search with Bitset Acceleration and Multiple Restarts**.
    # The current program has a weak local search.
    # I will implement a robust **Simulated Annealing** that:
    # 1. Starts with a random valid set (using the greedy logic from the seed).
    # 2. Moves by adding, removing, or swapping elements.
    # 3. Uses a bitset-based energy calculation that is O(N/word_size) or faster.
    # 4. Runs for the entire time budget, keeping the best valid set found.
    # 
    # This is a different optimization method (SA vs Greedy+LS).
    
    N = 128
    # Precompute bad triples representation
    # For each pair (i, j), we need to know which k's make it bad.
    # bad_k[i][j] is a bitmask of k's.
    bad_k = [[0] * N for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            g = gcd_mat[i][j]
            mask = 0
            for k in range(N):
                if k == i or k == j:
                    continue
                if gcd_mat[i][k] == g and gcd_mat[j][k] == g:
                    mask |= (1 << k)
            bad_k[i][j] = mask

    # Function to calculate the number of bad triples in a set represented by a mask.
    # This is O(N^2 / word_size) if we are careful, but Python integers are big ints.
    # Let's try to optimize the energy calculation.
    
    def count_bad_triples(mask):
        # Iterate through all pairs (i, j) in the set?
        # No, that's O(N^2).
        # We can iterate through i in set, then j in set where j > i.
        # For each pair, check if bad_k[i][j] & mask has any bits?
        # No, bad_k[i][j] contains k's. If k is in mask, then {i, j, k} is a bad triple.
        # So energy = sum_{i<j in mask} popcount(bad_k[i][j] & mask) / 3 ?
        # Each bad triple {a,b,c} is counted 3 times: (a,b,c), (b,c,a), (c,a,b) in the pairs?
        # Pair (a,b): checks k=c.
        # Pair (b,c): checks k=a.
        # Pair (c,a): checks k=b.
        # Yes, each bad triple contributes 3 to the sum.
        # So energy = sum // 3.
        
        total = 0
        temp = mask
        while temp:
            lsb_i = temp & -temp
            i = lsb_i.bit_length() - 1
            temp ^= lsb_i
            
            # Iterate j in mask where j > i to avoid double counting pairs?
            # If we iterate all j in mask, j != i, we count each pair twice.
            # Let's just iterate all j in mask.
            temp_j = mask & ~(1 << i)
            while temp_j:
                lsb_j = temp_j & -temp_j
                j = lsb_j.bit_length() - 1
                temp_j ^= lsb_j
                
                # Check if there is a k in mask such that {i, j, k} is bad.
                # k must be in mask.
                # bad_k[i][j] & mask gives the set of such k's.
                # But we must ensure k != i and k != j.
                # bad_k[i][j] already excludes i and j in the precomputation loop?
                # Yes: `if k == i or k == j: continue`.
                # So we just count the bits.
                total += (bad_k[i][j] & mask).bit_count()
                
        return total // 3

    # This O(N^2) energy calculation is too slow for SA with many steps.
    # We need incremental updates.
    
    # Let's use a different representation for the state: a list of elements.
    # And maintain a "conflict count" for each potential addition?
    
    # Strategy: **Greedy + Bitset Local Search with Swaps**
    # But make the Local Search much more aggressive and exhaustive within the time limit.
    # The current program does very few swaps.
    
    # I will implement a **Variable Neighborhood Search (VNS)** or **Iterated Local Search (ILS)** with bitset optimization.
    # Key optimization: Precompute `bad_k` as a list of lists of ints (bitmasks).
    # Use a `set` for the current solution to allow O(1) membership checks? No, bitmask is faster for iteration.
    
    # Let's try to improve the Greedy phase first, then do a deep Local Search.
    
    def greedy_construct(order):
        curr_mask = 0
        # We maintain a list of "blocked" elements?
        # If we add x, we might block some future elements?
        # No, adding x only creates bad triples with pairs already in the set.
        # So the validity of adding y depends on the current set S.
        # count_conflicts_add(y, S) = sum_{a in S} popcount(bad_k[y][a] & S) // 2
        
        for x in order:
            # Check if x can be added
            # If curr_mask is 0, always add.
            if curr_mask == 0:
                curr_mask |= (1 << x)
                continue
            
            # Calculate conflicts
            # We can optimize this:
            # conflicts = 0
            # temp = curr_mask
            # while temp:
            #   lsb = temp & -temp
            #   a = lsb.bit_length() - 1
            #   conflicts += (bad_k[x][a] & curr_mask).bit_count()
            #   temp ^= lsb
            # if conflicts == 0: add
            
            # To speed this up, we can precompute for each x, the "potential" conflicts?
            # No, it depends on S.
            
            # Let's just do the check. N=128, so at most 128 iterations per add.
            # 128 * 128 = 16k operations per greedy step. 128 steps = 2M ops. Fast enough.
            
            conflicts = 0
            temp = curr_mask
            while temp:
                lsb = temp & -temp
                a = lsb.bit_length() - 1
                conflicts += (bad_k[x][a] & curr_mask).bit_count()
                temp ^= lsb
                if conflicts > 0: # Early exit
                    break
            
            if conflicts == 0:
                curr_mask |= (1 << x)
                
        return curr_mask

    best_mask = 0
    best_size = 0
    
    # Multiple random restarts with Greedy + Deep Local Search
    max_restarts = 200 # We have time budget
    
    for restart in range(max_restarts):
        if time.time() > DEADLINE - 5:
            break
            
        rng = random.Random(42 + restart)
        order = list(range(N))
        rng.shuffle(order)
        
        curr_mask = greedy_construct(order)
        curr_size = curr_mask.bit_count()
        
        # Local Search Phase
        # We try to add any element that fits.
        # Then we try to swap.
        # Then we try to remove and add multiple?
        
        improved = True
        while improved:
            improved = False
            # 1. Try to add
            for x in range(N):
                if not (curr_mask & (1 << x)):
                    # Check if x fits
                    conflicts = 0
                    temp = curr_mask
                    while temp:
                        lsb = temp & -temp
                        a = lsb.bit_length() - 1
                        conflicts += (bad_k[x][a] & curr_mask).bit_count()
                        temp ^= lsb
                        if conflicts > 0:
                            break
                    if conflicts == 0:
                        curr_mask |= (1 << x)
                        curr_size += 1
                        improved = True
                        break # Restart loop to check new opportunities
            
            if not improved:
                # 2. Try to swap
                # Pick a random element i in set, and a random element j not in set.
                # If removing i and adding j keeps the set valid, do it.
                # This is a 1-swap move.
                set_bits = [i for i in range(N) if curr_mask & (1 << i)]
                not_set_bits = [j for j in range(N) if not (curr_mask & (1 << j))]
                
                if set_bits and not_set_bits:
                    # Try a few random swaps
                    for _ in range(50):
                        i = rng.choice(set_bits)
                        j = rng.choice(not_set_bits)
                        
                        # Test swap: S' = S \ {i} U {j}
                        # Valid if S \ {i} is valid (it is, subset of valid set)
                        # AND j does not form bad triples with pairs in S \ {i}.
                        S_minus_i = curr_mask & ~(1 << i)
                        
                        conflicts = 0
                        temp = S_minus_i
                        while temp:
                            lsb = temp & -temp
                            a = lsb.bit_length() - 1
                            conflicts += (bad_k[j][a] & S_minus_i).bit_count()
                            temp ^= lsb
                            if conflicts > 0:
                                break
                        
                        if conflicts == 0:
                            curr_mask = S_minus_i | (1 << j)
                            # Size stays same
                            improved = True
                            # Update lists
                            set_bits.remove(i)
                            not_set_bits.append(i)
                            not_set_bits.remove(j)
                            break
        
        if curr_size > best_size:
            best_size = curr_size
            best_mask = curr_mask
            
    # Convert best_mask to list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(elements[i])
            
    if not result:
        return [1, 2, 4, 8, 16, 32, 64, 128]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Final size: {len(res)}")
    print(res)
