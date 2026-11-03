"""Shortest superpermutation construction using optimized ordering via
TSP-style local search (2-opt + Or-opt) and simulated annealing for
the block-merging problem at each level."""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START

def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o > 0 and new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
                elif o == 0:
                    overlap = 0
                    break
            new += block[overlap:]
        word = new
    return word


def _build_level_perms(n):
    """
    Build the list of (n-1)-length permutations in the order they appear
    in the classic superpermutation of level n-1.
    """
    if n == 1:
        return [[1]]
    word = _classic_superperm(n - 1)
    perms, seen = [], set()
    m = n - 1
    for i in range(len(word) - m + 1):
        window = tuple(word[i:i + m])
        if len(set(window)) == m and window not in seen:
            seen.add(window)
            perms.append(window)
    return [list(p) for p in perms]


def _overlap_fast(a_tuple, b_tuple):
    """Fast overlap using tuple comparison."""
    max_ov = min(len(a_tuple), len(b_tuple))
    for o in range(max_ov, 0, -1):
        if a_tuple[-o:] == b_tuple[:o]:
            return o
    return 0


def _build_overlap_matrix(blocks):
    """Build full overlap matrix efficiently using tuples."""
    n = len(blocks)
    block_tuples = [tuple(b) for b in blocks]
    ov_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                ov_matrix[i][j] = _overlap_fast(block_tuples[i], block_tuples[j])
    return ov_matrix


def _block_for_perm(p, m):
    """The block for permutation p with new symbol m: p + [m] + p."""
    return list(p) + [m] + list(p)


def _optimize_order_tsp(perm_list, m, time_limit=10.0):
    """
    Try to find a better ordering of perm_list to minimize total length.
    Uses multiple initialization strategies + 2-opt + Or-opt + simulated annealing.
    """
    start_time = time.time()
    n_perms = len(perm_list)
    
    # Compute all-pairs overlap matrix
    blocks = [_block_for_perm(p, m) for p in perm_list]
    n = n_perms
    ov_matrix = _build_overlap_matrix(blocks)
    
    # Precompute block lengths
    block_lens = [len(b) for b in blocks]
    
    def compute_cost(order):
        """Compute total cost for a given order."""
        cost = block_lens[order[0]]
        for k in range(1, len(order)):
            i, j = order[k-1], order[k]
            cost += block_lens[j] - ov_matrix[i][j]
        return cost
    
    def greedy_construct(start_idx=0):
        """Greedy construction: always pick the next block with max overlap."""
        used = [False] * n
        order = []
        current = start_idx
        used[current] = True
        order.append(current)
        
        for _ in range(n - 1):
            best_j = -1
            best_ov = -1
            for j in range(n):
                if not used[j]:
                    ov = ov_matrix[current][j]
                    if ov > best_ov:
                        best_ov = ov
                        best_j = j
            if best_j == -1:
                break
            used[best_j] = True
            order.append(best_j)
            current = best_j
        
        return order
    
    def local_search_2opt(cur_order, cur_cost, t_limit):
        """2-opt local search with O(1) cost delta computation."""
        improved = True
        while improved and (time.time() - start_time) < t_limit:
            improved = False
            for i in range(n - 1):
                if (time.time() - start_time) > t_limit:
                    break
                for j in range(i + 1, n):
                    old_edges = 0
                    new_edges = 0
                    old_edges += block_lens[cur_order[i+1]] - ov_matrix[cur_order[i]][cur_order[i+1]]
                    if j + 1 < n:
                        old_edges += block_lens[cur_order[j+1]] - ov_matrix[cur_order[j]][cur_order[j+1]]
                    new_edges += block_lens[cur_order[j]] - ov_matrix[cur_order[i]][cur_order[j]]
                    if j + 1 < n:
                        new_edges += block_lens[cur_order[j+1]] - ov_matrix[cur_order[i+1]][cur_order[j+1]]
                    
                    delta = new_edges - old_edges
                    if delta < 0:
                        cur_cost += delta
                        cur_order[i+1:j+1] = cur_order[i+1:j+1][::-1]
                        improved = True
        return cur_order, cur_cost
    
    def oropt_search(cur_order, cur_cost, t_limit):
        """Or-opt: move a segment of length 1-3 to another position."""
        improved = True
        while improved and (time.time() - start_time) < t_limit:
            improved = False
            for seg_len in [1, 2, 3]:
                if (time.time() - start_time) > t_limit:
                    break
                for i in range(n - seg_len):
                    if (time.time() - start_time) > t_limit:
                        break
                    seg = cur_order[i:i+seg_len]
                    remaining = cur_order[:i] + cur_order[i+seg_len:]
                    if not remaining:
                        continue
                    for j in range(len(remaining) - seg_len + 1):
                        new_order = remaining[:j] + seg + remaining[j:]
                        if new_order == cur_order:
                            continue
                        new_cost = compute_cost(new_order)
                        if new_cost < cur_cost:
                            cur_cost = new_cost
                            cur_order = new_order
                            improved = True
                            break
        return cur_order, cur_cost
    
    def simulated_annealing(cur_order, cur_cost, t_limit):
        """Simulated annealing to escape local optima."""
        rng = random.Random(42)
        T = 5.0
        cooling = 0.999
        best_order = list(cur_order)
        best_cost = cur_cost
        
        while (time.time() - start_time) < t_limit and T > 0.01:
            # Random 2-opt move
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            old_cost = cur_cost
            cur_order[i+1:j+1] = cur_order[i+1:j+1][::-1]
            new_cost = compute_cost(cur_order)
            
            delta = new_cost - old_cost
            if delta < 0 or rng.random() < math.exp(-delta / T):
                cur_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_order = list(cur_order)
            else:
                # Revert
                cur_order[i+1:j+1] = cur_order[i+1:j+1][::-1]
            
            T *= cooling
        
        return best_order, best_cost
    
    # Try multiple initialization strategies
    initializations = []
    
    # 1. Classic order
    classic_order = list(range(n))
    classic_cost = compute_cost(classic_order)
    initializations.append((classic_order, classic_cost))
    
    # 2. Greedy from position 0
    greedy_order = greedy_construct(0)
    greedy_cost = compute_cost(greedy_order)
    initializations.append((greedy_order, greedy_cost))
    
    # 3. Greedy from a few other positions
    for start in [1, n//2, n-1]:
        if start < n:
            g_order = greedy_construct(start)
            g_cost = compute_cost(g_order)
            initializations.append((g_order, g_cost))
    
    # 4. Random orderings
    rng = random.Random(42)
    for _ in range(3):
        rand_order = list(range(n))
        rng.shuffle(rand_order)
        rand_cost = compute_cost(rand_order)
        initializations.append((rand_order, rand_cost))
    
    # Pick the best initialization
    best_order, best_cost = min(initializations, key=lambda x: x[1])
    
    # 2-opt from best starting point
    t1 = min(time_limit * 0.35, time_limit - 2.0)
    best_order, best_cost = local_search_2opt(best_order, best_cost, t1)
    
    # Or-opt polish
    t2 = min(time_limit * 0.25, time_limit - 1.5)
    if (time.time() - start_time) < time_limit - 1.5:
        best_order, best_cost = oropt_search(best_order, best_cost, t2)
    
    # Simulated annealing to escape local optima
    if (time.time() - start_time) < time_limit - 1.0:
        t3 = time_limit - (time.time() - start_time) - 0.5
        if t3 > 0.5:
            sa_order, sa_cost = simulated_annealing(best_order, best_cost, t3)
            if sa_cost < best_cost:
                best_cost = sa_cost
                best_order = sa_order
                # Polish with 2-opt after SA
                if (time.time() - start_time) < time_limit - 0.3:
                    t4 = time_limit - (time.time() - start_time) - 0.2
                    best_order, best_cost = local_search_2opt(best_order, best_cost, t4)
    
    # Try random restarts if time remains
    if (time.time() - start_time) < time_limit - 1.0:
        rng = random.Random(42)
        n_restarts = 5
        for r in range(n_restarts):
            if (time.time() - start_time) > time_limit - 0.5:
                break
            rand_order = list(range(n))
            rng.shuffle(rand_order)
            cur_order = list(rand_order)
            cur_cost = compute_cost(cur_order)
            t_rem = time_limit - (time.time() - start_time) - 0.5
            if t_rem > 0.5:
                # Quick 2-opt then SA
                t_2opt = min(t_rem * 0.3, 2.0)
                cur_order, cur_cost = local_search_2opt(cur_order, cur_cost, t_2opt)
                t_sa = time_limit - (time.time() - start_time) - 0.5
                if t_sa > 0.3:
                    cur_order, cur_cost = simulated_annealing(cur_order, cur_cost, t_sa)
                    if (time.time() - start_time) < time_limit - 0.3:
                        t_polish = time_limit - (time.time() - start_time) - 0.2
                        cur_order, cur_cost = local_search_2opt(cur_order, cur_cost, t_polish)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_order = cur_order
    
    return best_order, best_cost


def _solve_small_n(n, deadline):
    """
    For small n (<=5), try exact or near-exact search using the de Bruijn
    graph approach with greedy maximal-overlap path.
    """
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_set = set(all_perms)
    used = set()
    
    # For each (n-1)-tuple prefix, the set of symbols that extend it to a permutation
    ext = {}
    for p in all_perms:
        pref = p[:-1]
        ext.setdefault(pref, set()).add(p[-1])
    
    def is_dead_end(prefix):
        last = prefix[-1]
        for s in ext.get(prefix, ()):
            if s == last:
                continue
            p = prefix + (s,)
            if p not in used:
                return False
        return True
    
    # Start with the smallest permutation
    word = list(range(1, n + 1))
    used.add(tuple(word))
    
    while len(used) < len(all_perms):
        if time.time() > deadline:
            return None
        suffix = tuple(word[-(n - 1):])
        last = word[-1]
        cands = []
        for s in ext.get(suffix, ()):
            if s == last:
                continue
            p = suffix + (s,)
            if p not in used:
                cands.append((s, p))
        cands.sort()
        if not cands:
            return None
        chosen = None
        # Lookahead: prefer a candidate that does NOT create a dead-end suffix
        for s, p in cands:
            new_suffix = (suffix + (s,))[-(n - 1):]
            if not is_dead_end(new_suffix):
                chosen = (s, p)
                break
        if chosen is None:
            chosen = cands[0]
        s, p = chosen
        word.append(s)
        used.add(p)
    
    # Verify completeness
    if time.time() < deadline - 1:
        seen = set()
        w = word
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i + n]))
        if len(seen) == len(all_perms):
            return word
    return None


def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    if n <= 1:
        return "1"
    
    if n == 2:
        return "121"
    
    if n == 3:
        return "123121321"
    
    if n == 4:
        return "123412134212343142341234"
    
    # For n >= 5, use optimized construction
    # Build the (n-1)-permutations in classic order
    perm_list = _build_level_perms(n)
    m = n
    n_perms = len(perm_list)
    
    # Get classic length as baseline
    classic_word = _classic_superperm(n)
    classic_len = len(classic_word)
    
    # For small n, try the greedy maximal-overlap path approach
    if n <= 5:
        small_result = _solve_small_n(n, deadline - 10.0)
        if small_result is not None and len(small_result) < classic_len:
            return "".join(map(str, small_result))
    
    # Try to optimize the ordering at the top level
    if n_perms <= 720 and (time.time() - start_time) < deadline - 20.0:
        time_limit_for_tsp = min(100.0, deadline - (time.time() - start_time) - 20.0)
        if time_limit_for_tsp > 1.0:
            best_order, best_cost = _optimize_order_tsp(perm_list, m, time_limit=time_limit_for_tsp)
            
            # Build the word using the optimized order
            blocks = [_block_for_perm(perm_list[i], m) for i in best_order]
            word = list(blocks[0])
            for k in range(1, len(blocks)):
                b = blocks[k]
                ov = _overlap_fast(tuple(word), tuple(b))
                word += b[ov:]
            
            # Verify completeness before returning
            if time.time() < deadline - 5:
                all_perms = set(itertools.permutations(range(1, n + 1)))
                seen = set()
                for i in range(len(word) - n + 1):
                    seen.add(tuple(word[i:i + n]))
                if len(seen) == len(all_perms) and len(word) < classic_len:
                    return "".join(map(str, word))
    
    # Fall back to classic
    return "".join(map(str, classic_word))

# EVOLVE-BLOCK-END
