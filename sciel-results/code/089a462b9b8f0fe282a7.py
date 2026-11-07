"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

MAX_N = 256
DEADLINE = 85.0  # Safety margin before 90s kill

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _prime_factors(n):
    """Return the set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    
    N = MAX_N  # 256
    # Map numbers 2..256 to indices 0..254
    nums = list(range(2, N + 1))
    n = len(nums)  # 255 elements
    
    # Precompute prime factors
    pf = {}
    for x in range(2, N + 1):
        pf[x] = _prime_factors(x)
    
    # Build the 3-uniform hypergraph constraint:
    # For each pair (i,j), compute the set of vertices k such that {i,j,k} is a bad triple.
    # A triple {a,b,c} is bad if a|b*c or b|a*c or c|a*b.
    
    pair_forbidden = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            forbidden = 0
            a = nums[i]
            b = nums[j]
            ab = a * b
            for k in range(n):
                if k == i or k == j:
                    continue
                c = nums[k]
                if (b * c) % a == 0 or (a * c) % b == 0 or ab % c == 0:
                    forbidden |= (1 << k)
            pair_forbidden[i][j] = forbidden
            pair_forbidden[j][i] = forbidden
    
    # Now build a 2-uniform conflict graph as suggested by the strategy:
    # Two vertices x and y are connected by an edge if there EXISTS any z in [2,256]
    # such that x | y*z. This is a necessary condition for x and y to be incompatible
    # in the sense that if both x and y are in our set, then for the set to be valid,
    # no z that is "witnessed" by this conflict can also be in the set.
    # 
    # However, this 2-uniform graph is NOT sufficient - we still need the 3-uniform check.
    # But we can use it for pruning: if x and y have an edge, then certain z's are forbidden.
    #
    # Actually, let me reconsider. The strategy says to model it as a maximum independent set
    # in a standard 2-uniform conflict graph where an edge connects x and y if there exists
    # any z in [2, 256] such that x | yz.
    #
    # Wait, that's not quite right either. Let me re-read: "an edge connects x and y if there
    # exists any z in [2, 256] such that x | yz". This would mean x and y cannot both be in
    # the set, which is too strong. The actual constraint is 3-way.
    #
    # Let me think about this differently. Perhaps the idea is to use the 2-uniform graph
    # for an upper bound or for a different formulation.
    #
    # Actually, I think the correct interpretation is: we can use the pairwise structure
    # for more efficient branch-and-bound. Let me implement a proper branch-and-bound
    # with better pruning.
    
    # For each vertex v, precompute which other vertices it "conflicts with" in the sense
    # that if v is in the solution, then for any pair (u,w) in the solution, v must not
    # be in pair_forbidden[u][w].
    
    # Let's use a branch-and-bound approach:
    # State: current solution S (as bitset), remaining candidates R (as bitset)
    # Upper bound: |S| + |R| (trivial), but we can do better.
    
    # A better upper bound: for each remaining vertex v, count how many vertices in R
    # are "compatible" with v (i.e., can be in the solution together with v).
    # The maximum number of additional vertices we can add is at most the size of the
    # maximum clique in the compatibility graph, which is hard to compute.
    
    # Let's use a simpler but effective approach: greedy with multiple orderings
    # and local search with 1-swap and 2-swap moves.
    
    def can_add_to(S_bitset, v):
        """Check if vertex v can be added to the set represented by S_bitset."""
        if S_bitset & (1 << v):
            return True
        temp = S_bitset
        while temp:
            lb = temp & (-temp)
            u_idx = lb.bit_length() - 1
            if u_idx != v:
                if S_bitset & pair_forbidden[u_idx][v]:
                    return False
            temp ^= lb
        return True
    
    # Compute vertex "degree" for ordering
    vertex_degree = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            cnt = bin(pair_forbidden[i][j]).count('1')
            vertex_degree[i] += cnt
            vertex_degree[j] += cnt
    
    # Multiple greedy strategies
    best_size = 0
    best_bitset = 0
    
    # Strategy 1: Greedy by ascending degree
    order1 = sorted(range(n), key=lambda x: vertex_degree[x])
    S_bitset = 0
    for v in order1:
        if time.time() - start_time > DEADLINE - 15:
            break
        if can_add_to(S_bitset, v):
            S_bitset |= (1 << v)
    sz = bin(S_bitset).count('1')
    if sz > best_size:
        best_size = sz
        best_bitset = S_bitset
    
    # Strategy 2: Greedy by descending degree
    order2 = sorted(range(n), key=lambda x: -vertex_degree[x])
    S_bitset = 0
    for v in order2:
        if time.time() - start_time > DEADLINE - 15:
            break
        if can_add_to(S_bitset, v):
            S_bitset |= (1 << v)
    sz = bin(S_bitset).count('1')
    if sz > best_size:
        best_size = sz
        best_bitset = S_bitset
    
    # Strategy 3: Random restarts with different seeds
    for restart in range(100):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        random.seed(restart * 7919 + 13)
        S_bitset = 0
        
        order = list(range(n))
        random.shuffle(order)
        
        for v in order:
            if time.time() - start_time > DEADLINE - 10:
                break
            if can_add_to(S_bitset, v):
                S_bitset |= (1 << v)
        
        sz = bin(S_bitset).count('1')
        if sz > best_size:
            best_size = sz
            best_bitset = S_bitset
    
    # Local search: 1-swap moves
    for _ in range(500):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        improved = False
        S_list = []
        temp = best_bitset
        while temp:
            lb = temp & (-temp)
            idx = lb.bit_length() - 1
            S_list.append(idx)
            temp ^= lb
        
        not_S = [i for i in range(n) if not (best_bitset & (1 << i))]
        random.shuffle(not_S)
        
        for v in not_S:
            if time.time() - start_time > DEADLINE - 5:
                break
            for u in S_list:
                if u == v:
                    continue
                new_bitset = (best_bitset & ~(1 << u)) | (1 << v)
                valid = True
                temp2 = new_bitset & ~(1 << v)
                while temp2:
                    lb = temp2 & (-temp2)
                    w_idx = lb.bit_length() - 1
                    if w_idx != v:
                        if new_bitset & pair_forbidden[w_idx][v]:
                            valid = False
                            break
                    temp2 ^= lb
                if valid:
                    best_bitset = new_bitset
                    best_size = bin(best_bitset).count('1')
                    improved = True
                    break
            if improved:
                break
    
    # Local search: 2-swap moves (remove two, add one or two)
    for _ in range(200):
        if time.time() - start_time > DEADLINE - 3:
            break
        
        improved = False
        S_list = []
        temp = best_bitset
        while temp:
            lb = temp & (-temp)
            idx = lb.bit_length() - 1
            S_list.append(idx)
            temp ^= lb
        
        not_S = [i for i in range(n) if not (best_bitset & (1 << i))]
        random.shuffle(not_S)
        
        # Try removing two vertices and adding one
        for u1 in S_list:
            if time.time() - start_time > DEADLINE - 3:
                break
            for u2 in S_list:
                if u2 <= u1:
                    continue
                for v in not_S[:20]:  # Limit candidates for speed
                    new_bitset = (best_bitset & ~(1 << u1) & ~(1 << u2)) | (1 << v)
                    valid = True
                    temp2 = new_bitset & ~(1 << v)
                    while temp2:
                        lb = temp2 & (-temp2)
                        w_idx = lb.bit_length() - 1
                        if w_idx != v:
                            if new_bitset & pair_forbidden[w_idx][v]:
                                valid = False
                                break
                        temp2 ^= lb
                    if valid:
                        sz = bin(new_bitset).count('1')
                        if sz > best_size:
                            best_bitset = new_bitset
                            best_size = sz
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    # Convert best_bitset to list of integers
    result = []
    temp = best_bitset
    while temp:
        lb = temp & (-temp)
        idx = lb.bit_length() - 1
        result.append(nums[idx])
        temp ^= lb
    
    result.sort()
    
    # Ensure we return Python ints in a list
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
