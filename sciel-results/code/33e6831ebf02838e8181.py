import time
from itertools import combinations

L = 5354228880  # lcm(1..24)
N = 24
ALL = list(range(1, 25))

# Precompute reciprocals as L//i
RECIP = [0] + [L // i for i in range(1, 25)]  # RECIP[i] = L//i


def _precompute_hyperedges():
    """
    Find all minimal hyperedges: minimal subsets H of {1..24} such that there
    exists a target t in H with L//t == sum of L//y for y in H \\ {t}.
    These are the minimal violating configurations. A set S is valid iff it
    contains no hyperedge as a subset.
    """
    hyperedges = set()
    # For each target t, find minimal subsets of OTHERS that sum to L//t
    for t in range(1, 25):
        target = L // t
        others = [x for x in range(1, 25) if x != t]
        # We need subsets of others whose reciprocals sum to target.
        # Only consider others with L//x <= target (larger ones can't be in a sum to target)
        candidates = [x for x in others if L // x <= target]
        if not candidates:
            continue
        # Find all minimal subsets of candidates summing to target
        # Use DFS with pruning
        candidates.sort(key=lambda x: L // x, reverse=True)  # largest recip first
        cand_weights = [L // x for x in candidates]
        found = set()
        
        def dfs(idx, current_sum, chosen):
            if current_sum == target:
                if chosen:
                    key = frozenset(chosen)
                    if len(key) <= 8:  # cap size to avoid explosion
                        found.add(key)
                return
            if idx >= len(candidates):
                return
            if current_sum > target:
                return
            # Prune: even taking all remaining can't reach target
            remaining_sum = sum(cand_weights[idx:])
            if current_sum + remaining_sum < target:
                return
            # Option 1: skip
            dfs(idx + 1, current_sum, chosen)
            # Option 2: take
            new_sum = current_sum + cand_weights[idx]
            if new_sum <= target:
                dfs(idx + 1, new_sum, chosen + [candidates[idx]])
        
        dfs(0, 0, [])
        
        # Now find minimal ones (no proper subset is also a solution)
        for h in found:
            is_minimal = True
            # Check if any proper subset of h is also in found
            h_list = list(h)
            for k in range(1, len(h_list)):
                for sub in combinations(h_list, k):
                    if frozenset(sub) in found:
                        is_minimal = False
                        break
                if not is_minimal:
                    break
            if is_minimal:
                hyperedges.add(h)
    
    return list(hyperedges)


# Precompute at module load
HYPEREDGES = _precompute_hyperedges()
# Convert to frozensets for fast subset checks
HYPEREDGES_FROZEN = [frozenset(h) for h in HYPEREDGES]


def _set_contains_hyperedge(S_set):
    """Fast check: does S_set contain any hyperedge as a subset?"""
    for h in HYPEREDGES_FROZEN:
        if h.issubset(S_set):
            return True
    return False


def is_valid(S):
    """
    S: list of ints. Return True if S is a valid reciprocal-subsum-free set.
    """
    if not S:
        return True
    S_set = set(S)
    return not _set_contains_hyperedge(S_set)


def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin

    def now():
        return time.time()

    best_set = None
    best_size = 0

    # Start with a known-valid seed
    seed = list(range(13, 25))
    if is_valid(seed):
        best_set = seed[:]
        best_size = len(seed)
    else:
        best_set = [24]
        best_size = 1

    import random

    # Branch-and-bound on the hypergraph independence problem.
    # State: (decided_prefix, current_set, remaining_candidates)
    # We decide elements in a fixed order. At each step, we can include or exclude.
    # Pruning: if len(current_set) + len(remaining) <= best_size, prune.
    # Pruning: if current_set already contains a hyperedge, prune (shouldn't happen if we check incrementally).
    
    # Use a smarter ordering: process elements that appear in more hyperedges first
    # to get conflicts early.
    edge_count = [0] * 25
    for h in HYPEREDGES_FROZEN:
        for x in h:
            edge_count[x] += 1
    
    # Order: high conflict first
    order = sorted(range(1, 25), key=lambda x: -edge_count[x])
    order_set = set(order)
    
    # Precompute for each position in order, which elements come after it
    # For fast "remaining" count
    
    def bnb_solve(order, deadline):
        """
        Branch and bound over the ordering.
        Returns best valid set found.
        """
        best_found = [best_set[:], best_size]
        
        # Incremental validity: maintain current set as frozenset
        # Check if adding x to current_set creates a hyperedge
        
        def check_add(current_frozen, x):
            """Check if current_frozen | {x} contains any hyperedge."""
            new_frozen = current_frozen | {x}
            for h in HYPEREDGES_FROZEN:
                if h.issubset(new_frozen):
                    return False
            return True
        
        # DFS with pruning
        # Use iterative approach to avoid recursion depth issues
        # State: (pos, current_set_frozen, current_list)
        # We'll use a stack for DFS
        
        stack = [(0, frozenset(), [])]
        nodes_explored = 0
        
        while stack and now() < deadline:
            pos, cur_frozen, cur_list = stack.pop()
            nodes_explored += 1
            
            # Prune by upper bound
            remaining_count = len(order) - pos
            if len(cur_list) + remaining_count <= best_found[1]:
                continue
            
            if pos >= len(order):
                if len(cur_list) > best_found[1]:
                    best_found[1] = len(cur_list)
                    best_found[0] = cur_list[:]
                continue
            
            x = order[pos]
            
            # Option 1: include x (try first for maximization)
            if check_add(cur_frozen, x):
                new_frozen = cur_frozen | {x}
                new_list = cur_list + [x]
                # Push exclude option first so include is explored first (LIFO)
                stack.append((pos + 1, cur_frozen, cur_list))
                stack.append((pos + 1, new_frozen, new_list))
            else:
                # Can't include, only exclude
                stack.append((pos + 1, cur_frozen, cur_list))
        
        return best_found
    
    # Run B&B
    result = bnb_solve(order, deadline)
    if result[1] > best_size:
        best_size = result[1]
        best_set = result[0][:]
    
    # If B&B didn't finish or found nothing better, try greedy + local search as backup
    if now() < deadline:
        def greedy_add(order_g, start_set=None):
            S = set(start_set) if start_set is not None else set()
            S_list = list(S)
            for x in order_g:
                if now() > deadline:
                    break
                if x in S:
                    continue
                trial = S_list + [x]
                if is_valid(trial):
                    S.add(x)
                    S_list = list(S)
            return S_list
        
        orderings = [
            ALL[:],
            ALL[::-1],
            sorted(ALL, key=lambda x: -edge_count[x]),
            sorted(ALL, key=lambda x: L // x),
            sorted(ALL, key=lambda x: L // x, reverse=True),
        ]
        rng = random.Random(42)
        for _ in range(5):
            o = ALL[:]
            rng.shuffle(o)
            orderings.append(o)
        
        for o in orderings:
            if now() > deadline:
                break
            res = greedy_add(o)
            if len(res) > best_size:
                best_size = len(res)
                best_set = res[:]
        
        # Local search on best
        def local_search(S, max_iters=200):
            S = set(S)
            S_list = list(S)
            iters = 0
            while iters < max_iters and now() < deadline:
                iters += 1
                improved = False
                missing = [x for x in ALL if x not in S]
                if missing:
                    rng2 = random.Random(iters + len(S) * 1000)
                    rng2.shuffle(missing)
                    for x in missing:
                        if now() > deadline:
                            break
                        trial = list(S) + [x]
                        if is_valid(trial):
                            S.add(x)
                            S_list = list(S)
                            improved = True
                            break
                if improved:
                    continue
                if len(S) > 0:
                    a = rng.choice(list(S))
                    missing = [x for x in ALL if x not in S]
                    if missing:
                        rng.shuffle(missing)
                        for b in missing:
                            if now() > deadline:
                                break
                            new_set = (S - {a}) | {b}
                            if is_valid(list(new_set)):
                                S = new_set
                                S_list = list(S)
                                improved = True
                                break
                if not improved:
                    break
            return S_list
        
        if best_set is not None and now() < deadline:
            ls = local_search(best_set, max_iters=100)
            if len(ls) > best_size:
                best_size = len(ls)
                best_set = ls[:]
    
    # Final validity check
    if best_set is None or not is_valid(best_set):
        best_set = [24]
        best_size = 1
    else:
        best_set = list(best_set)
        best_size = len(best_set)
    
    return best_set


if __name__ == "__main__":
    print(construct())
