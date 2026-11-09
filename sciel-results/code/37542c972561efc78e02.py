import time
import random
from math import gcd
from itertools import combinations
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_lcms(n_max):
    """Precompute LCM table for all pairs in [1, n_max]."""
    n = n_max
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    return lcm_table

def _build_conflict_table(lcm_table, n_max):
    """
    Build conflict structure: for each element x, store the set of pairs (a,b)
    such that {x,a,b} is a bad triple.
    """
    n = n_max
    bad_pairs = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bc = (min(b,c), max(b,c))
                    ac = (min(a,c), max(a,c))
                    ab = (a, b)
                    bad_pairs[a].add(bc)
                    bad_pairs[b].add(ac)
                    bad_pairs[c].add(ab)
    
    return bad_pairs

def _build_conflict_bitsets(bad_pairs, n_max):
    """
    Build conflict bitsets for fast can_add checks.
    conflict[x][a] is a bitmask of b such that (a,b,x) is a bad triple.
    """
    n = n_max
    conflict = [[0] * (n + 1) for _ in range(n + 1)]
    for x in range(1, n + 1):
        for (a, b) in bad_pairs[x]:
            conflict[x][a] |= (1 << b)
            conflict[x][b] |= (1 << a)
    return conflict

def _can_add_bitset(x, selected_mask, conflict, n_max):
    """Check if x can be added to the set represented by selected_mask using bitsets."""
    for a in range(1, n_max + 1):
        if selected_mask & (1 << a):
            if conflict[x][a] & selected_mask:
                return False
    return True

def _greedy_construct_fast(order, bad_pairs, n_max):
    """Greedily add numbers in the given order if they don't form a bad triple."""
    selected = set()
    selected_list = []
    
    for x in order:
        ok = True
        for (a, b) in bad_pairs[x]:
            if a in selected and b in selected:
                ok = False
                break
        if ok:
            selected.add(x)
            selected_list.append(x)
    
    return selected_list

def _greedy_construct_bitset(order, conflict, n_max):
    """Greedily add numbers in the given order using bitset-based can_add."""
    selected_mask = 0
    selected_list = []
    
    for x in order:
        if selected_mask & (1 << x):
            continue
        if _can_add_bitset(x, selected_mask, conflict, n_max):
            selected_mask |= (1 << x)
            selected_list.append(x)
    
    return selected_list

def _build_weighted_conflict_graph(bad_pairs, n_max):
    """
    Build weighted conflict graph where edge weight = number of bad triples 
    involving that pair.
    """
    n = n_max
    edge_weights = defaultdict(int)
    
    for x in range(1, n + 1):
        for (a, b) in bad_pairs[x]:
            edge_weights[(a, b)] += 1
    
    return edge_weights

def _spectral_ordering(edge_weights, n_max):
    """
    Use spectral methods to order vertices for greedy construction.
    Vertices with lower "conflict centrality" should be selected first.
    """
    import numpy as np
    
    n = n_max
    W = np.zeros((n + 1, n + 1))
    for (a, b), w in edge_weights.items():
        W[a][b] = w
        W[b][a] = w
    
    degrees = W.sum(axis=1)
    max_deg = degrees.max() if degrees.max() > 0 else 1
    centrality = degrees / max_deg
    
    neighbor_cent = np.zeros(n + 1)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if W[i][j] > 0:
                neighbor_cent[i] += centrality[j]
    
    combined = centrality + 0.5 * neighbor_cent
    order = list(range(1, n + 1))
    order.sort(key=lambda x: combined[x])
    
    return order

def _spectral_relaxation(edge_weights, n_max, iterations=100):
    """
    Solve spectral relaxation: minimize sum of edge weights * x_i * x_j
    subject to 0 <= x_i <= 1.
    Use iterative projection method.
    Returns fractional solution.
    """
    import numpy as np
    
    n = n_max
    W = np.zeros((n + 1, n + 1))
    for (a, b), w in edge_weights.items():
        W[a][b] = w
        W[b][a] = w
    
    x = np.full(n + 1, 0.5)
    x[0] = 0
    
    for it in range(iterations):
        grad = 2.0 * (W @ x)
        alpha = 0.01
        x_new = x - alpha * grad
        x_new = np.clip(x_new, 0.0, 1.0)
        x_new[0] = 0
        
        if np.max(np.abs(x_new - x)) < 1e-6:
            x = x_new
            break
        x = x_new
    
    return x

def _threshold_from_spectral(x, n_max, target_size):
    """Convert fractional solution to a candidate set by thresholding."""
    indices = list(range(1, n_max + 1))
    indices.sort(key=lambda i: x[i], reverse=True)
    return indices[:target_size]

def _local_search_improve_bitset(subset, conflict, n_max, time_limit=10.0):
    """
    Local search using bitset-based can_add for speed.
    Tries to add elements, then swap.
    """
    start_time = time.time()
    current_mask = 0
    for x in subset:
        current_mask |= (1 << x)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try adding each excluded number
        for x in range(1, n_max + 1):
            if time.time() - start_time > time_limit:
                break
            if current_mask & (1 << x):
                continue
            if _can_add_bitset(x, current_mask, conflict, n_max):
                current_mask |= (1 << x)
                improved = True
                break
        
        # Try swapping: remove one element, try to add a different one
        if not improved:
            selected = [i for i in range(1, n_max + 1) if current_mask & (1 << i)]
            for i in range(len(selected)):
                if time.time() - start_time > time_limit:
                    break
                candidate_remove = selected[i]
                new_mask = current_mask & ~(1 << candidate_remove)
                for x in range(1, n_max + 1):
                    if time.time() - start_time > time_limit:
                        break
                    if new_mask & (1 << x):
                        continue
                    if _can_add_bitset(x, new_mask, conflict, n_max):
                        new_mask |= (1 << x)
                        if bin(new_mask).count('1') > bin(current_mask).count('1'):
                            current_mask = new_mask
                            improved = True
                            break
                if improved:
                    break
    
    return [i for i in range(1, n_max + 1) if current_mask & (1 << i)]

def _branch_and_bound(bad_pairs, n_max, initial_solution, time_limit=60.0):
    """
    Branch-and-bound solver for maximum independent set in 3-uniform hypergraph.
    """
    start_time = time.time()
    best_solution = list(initial_solution)
    best_size = len(best_solution)
    
    adj = [set() for _ in range(n_max + 1)]
    for z in range(1, n_max + 1):
        for (a, b) in bad_pairs[z]:
            adj[a].add(b)
            adj[b].add(a)
    
    def _greedy_color_bound(vertices, adj):
        if not vertices:
            return 0
        
        remaining = set(vertices)
        num_colors = 0
        
        while remaining:
            num_colors += 1
            v = max(remaining, key=lambda x: len(adj[x] & remaining))
            color_class = {v}
            for u in list(remaining):
                if u != v and u not in adj[v]:
                    color_class.add(u)
            remaining -= color_class
        
        return num_colors
    
    def _can_add(x, selected_set):
        for (a, b) in bad_pairs[x]:
            if a in selected_set and b in selected_set:
                return False
        return True
    
    def _solve(selected, available, depth, current_size):
        nonlocal best_solution, best_size, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        if current_size > best_size:
            best_size = current_size
            best_solution = list(selected)
        
        if not available:
            return
        
        bound = current_size + _greedy_color_bound(available, adj)
        if bound <= best_size:
            return
        
        v = max(available, key=lambda x: len(adj[x] & available))
        
        if _can_add(v, selected):
            new_selected = selected | {v}
            new_available = available - {v}
            _solve(new_selected, new_available, depth + 1, current_size + 1)
        
        _solve(selected, available - {v}, depth + 1, current_size)
    
    all_vertices = set(range(1, n_max + 1))
    _solve(set(), all_vertices, 0, 0)
    
    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    n_max = 128
    numbers = list(range(1, n_max + 1))
    
    # Precompute LCM table
    lcm_table = _precompute_lcms(n_max)
    
    # Build conflict table
    bad_pairs = _build_conflict_table(lcm_table, n_max)
    
    # Build conflict bitsets for fast checks
    conflict = _build_conflict_bitsets(bad_pairs, n_max)
    
    # Build weighted conflict graph
    edge_weights = _build_weighted_conflict_graph(bad_pairs, n_max)
    
    # Strategy 1: Primes and Prime Powers first
    def is_prime_power(n):
        if n < 2:
            return False
        p = n
        while p > 1:
            if p * p > n:
                break
            if n % p == 0:
                n //= p
            else:
                return False
        return True

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in numbers if is_prime(x)]
    prime_powers = [x for x in numbers if is_prime_power(x) and x not in primes]
    others = [x for x in numbers if x not in primes and x not in prime_powers]
    
    order1 = primes + prime_powers + others
    result1 = _greedy_construct_bitset(order1, conflict, n_max)
    
    order2 = numbers
    result2 = _greedy_construct_bitset(order2, conflict, n_max)
    
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    order3 = sorted(numbers, key=num_divisors)
    result3 = _greedy_construct_bitset(order3, conflict, n_max)
    
    order4 = sorted(numbers, key=lambda x: len(bad_pairs[x]))
    result4 = _greedy_construct_bitset(order4, conflict, n_max)
    
    # Spectral ordering
    order5 = _spectral_ordering(edge_weights, n_max)
    result5 = _greedy_construct_bitset(order5, conflict, n_max)
    
    # Spectral relaxation
    try:
        x_frac = _spectral_relaxation(edge_weights, n_max, iterations=50)
        spectral_target = sum(1 for i in range(1, n_max + 1) if x_frac[i] > 0.5)
        for target_size in [spectral_target, spectral_target + 1, spectral_target - 1]:
            if 1 <= target_size <= n_max:
                candidate_order = _threshold_from_spectral(x_frac, n_max, target_size)
                remaining = [x for x in numbers if x not in set(candidate_order)]
                full_order = candidate_order + remaining
                res = _greedy_construct_bitset(full_order, conflict, n_max)
                if len(res) > len(result5):
                    result5 = res
    except:
        pass
    
    best_result = max([result1, result2, result3, result4, result5], key=len)
    
    # Try random shuffles
    random.seed(42)
    for i in range(500):
        if time.time() > deadline:
            break
        shuffled = numbers[:]
        random.shuffle(shuffled)
        res = _greedy_construct_bitset(shuffled, conflict, n_max)
        if len(res) > len(best_result):
            best_result = res
    
    # Local search improvement using bitsets
    if time.time() < deadline - 20:
        improved = _local_search_improve_bitset(best_result, conflict, n_max, time_limit=15.0)
        if len(improved) > len(best_result):
            best_result = improved
    
    # Branch and bound for exact optimization
    if time.time() < deadline - 10:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 5:
            bb_result = _branch_and_bound(bad_pairs, n_max, best_result, time_limit=remaining_time)
            if len(bb_result) > len(best_result):
                best_result = bb_result
    
    # Final verification
    def _is_valid_subset(subset):
        n = len(subset)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = subset[i], subset[j], subset[k]
                    if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
                        return False
        return True
    
    if not _is_valid_subset(best_result):
        best_result = result1
    
    return best_result
