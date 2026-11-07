"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

MAX_N = 256
DEADLINE = 85.0  # Safety margin before 90s kill

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
    nums = list(range(2, N + 1))
    n = len(nums)  # 255 elements
    
    # Precompute: for each pair (i,j), a bitset of forbidden third vertices.
    # pair_forbidden[i][j] = bitset of k such that {i,j,k} is a bad triple
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
    
    # Precompute vertex degree (number of bad triples involving this vertex)
    vertex_degree = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            cnt = bin(pair_forbidden[i][j]).count('1')
            vertex_degree[i] += cnt
            vertex_degree[j] += cnt
    
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
    
    def is_valid(bitset):
        """Check if the set represented by bitset is valid."""
        temp = bitset
        vertices = []
        while temp:
            lb = temp & (-temp)
            idx = lb.bit_length() - 1
            vertices.append(idx)
            temp ^= lb
        
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if bitset & pair_forbidden[vertices[i]][vertices[j]]:
                    return False
        return True
    
    # Greedy construction with multiple orderings
    best_size = 0
    best_bitset = 0
    
    # Ordering 1: by degree ascending
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
    
    # Ordering 2: by degree descending
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
    
    # Random restarts with greedy
    rng = random.Random(42)
    for restart in range(30):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        S_bitset = 0
        order = list(range(n))
        rng.shuffle(order)
        
        for v in order:
            if time.time() - start_time > DEADLINE - 10:
                break
            if can_add_to(S_bitset, v):
                S_bitset |= (1 << v)
        
        sz = bin(S_bitset).count('1')
        if sz > best_size:
            best_size = sz
            best_bitset = S_bitset
    
    # Local search: 1-swap (remove one, add one)
    def local_search_1swap(bitset, max_iters=500):
        current = bitset
        current_size = bin(current).count('1')
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            improved = False
            S_list = []
            temp = current
            while temp:
                lb = temp & (-temp)
                idx = lb.bit_length() - 1
                S_list.append(idx)
                temp ^= lb
            
            not_S = [i for i in range(n) if not (current & (1 << i))]
            rng.shuffle(not_S)
            
            for v in not_S:
                if time.time() - start_time > DEADLINE - 5:
                    break
                for u in S_list:
                    if u == v:
                        continue
                    new_bitset = (current & ~(1 << u)) | (1 << v)
                    # Check validity: only need to check pairs involving v
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
                        current = new_bitset
                        current_size += 1  # size stays same (remove one, add one)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current, current_size
    
    best_bitset, best_size = local_search_1swap(best_bitset)
    
    # Local search: try to add a vertex by removing multiple vertices
    def local_search_add(bitset, max_iters=200):
        current = bitset
        current_size = bin(current).count('1')
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE - 3:
                break
            
            improved = False
            not_S = [i for i in range(n) if not (current & (1 << i))]
            rng.shuffle(not_S)
            
            S_list = []
            temp = current
            while temp:
                lb = temp & (-temp)
                idx = lb.bit_length() - 1
                S_list.append(idx)
                temp ^= lb
            
            for v in not_S:
                if time.time() - start_time > DEADLINE - 3:
                    break
                
                # Find which vertices in S conflict with v
                conflicts = []
                for u in S_list:
                    if u != v and (current & pair_forbidden[u][v]):
                        conflicts.append(u)
                
                # Try removing subsets of conflicts to add v
                # Start with removing 1 conflict
                if len(conflicts) <= 3:
                    # Try removing all conflicts
                    new_bitset = current
                    for u in conflicts:
                        new_bitset &= ~(1 << u)
                    new_bitset |= (1 << v)
                    
                    if bin(new_bitset).count('1') > current_size:
                        # Verify validity
                        if is_valid(new_bitset):
                            current = new_bitset
                            current_size = bin(current).count('1')
                            improved = True
                            break
            
            if not improved:
                break
        
        return current, current_size
    
    best_bitset, best_size = local_search_add(best_bitset)
    
    # More 1-swaps after add search
    best_bitset, best_size = local_search_1swap(best_bitset, max_iters=300)
    
    # Convert best_bitset to list of integers
    result = []
    temp = best_bitset
    while temp:
        lb = temp & (-temp)
        idx = lb.bit_length() - 1
        result.append(nums[idx])
        temp ^= lb
    
    result.sort()
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
