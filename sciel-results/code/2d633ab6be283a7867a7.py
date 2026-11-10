"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 82.0  # safety margin before 90s wall-clock limit

def _within_deadline():
    return (time.time() - _START_TIME) < _DEADLINE

# Precompute GCD table
_GCD = [[0] * 129 for _ in range(129)]
for i in range(1, 129):
    for j in range(1, 129):
        _GCD[i][j] = math.gcd(i, j)

def _bad_triple(a, b, c):
    """Return True if the triple (a,b,c) has all three pairwise gcds equal."""
    return _GCD[a][b] == _GCD[a][c] == _GCD[b][c]

def _build_hypergraph():
    """Build hypergraph: hyperedges are bad triples, compute degrees."""
    full = list(range(1, 129))
    hyperedges = []
    for a, b, c in combinations(full, 3):
        if _bad_triple(a, b, c):
            hyperedges.append((a, b, c))
    # Degree: number of hyperedges containing vertex
    degree = {x: 0 for x in full}
    for a, b, c in hyperedges:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
    return hyperedges, degree

def _conflict_pairs(S):
    """For each element in S, find which other elements form a bad triple with it."""
    S_set = set(S)
    conflicts = {x: set() for x in S}
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a, b = S_list[i], S_list[j]
            # Find c such that (a,b,c) is bad
            for c in S_list:
                if c != a and c != b and _bad_triple(a, b, c):
                    conflicts[a].add(c)
                    conflicts[b].add(c)
                    conflicts[c].add(a)
                    conflicts[c].add(b)
    return conflicts

def _can_add(x, S_list, S_set, conflict_cache):
    """Check if x can be added to S without creating a bad triple."""
    # x conflicts with a pair (a,b) in S if _bad_triple(a,b,x)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            if _bad_triple(S_list[i], S_list[j], x):
                return False
    return True

def _greedy_add(order, S_init=None):
    """Greedy construction using precomputed GCD."""
    S = list(S_init) if S_init else []
    S_set = set(S)
    for x in order:
        if not _within_deadline():
            break
        if x in S_set:
            continue
        ok = True
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                if _bad_triple(S[i], S[j], x):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            S.append(x)
            S_set.add(x)
    return S

def _local_search(S, max_iters=200):
    """Swap-based local search: remove one, try to add multiple others."""
    S = list(S)
    full = list(range(1, 129))
    best_size = len(S)
    
    for _ in range(max_iters):
        if not _within_deadline():
            break
        if len(S) >= 128:
            break
        
        improved = False
        # Try removing each element
        for i in range(len(S)):
            candidate = S[:i] + S[i + 1:]
            cand_set = set(candidate)
            # Try adding all non-members
            added = False
            for x in full:
                if x in cand_set:
                    continue
                ok = True
                for j in range(len(candidate)):
                    for k in range(j + 1, len(candidate)):
                        if _bad_triple(candidate[j], candidate[k], x):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    candidate.append(x)
                    cand_set.add(x)
                    added = True
            
            if len(candidate) > len(S):
                S = candidate
                best_size = len(S)
                improved = True
                break
        
        if not improved:
            break
    
    return S

def _branch_and_bound(hyperedges, degree, initial_set=None):
    """Branch-and-bound on the hypergraph for maximum independent set."""
    full = set(range(1, 129))
    if initial_set is not None:
        S = set(initial_set)
    else:
        S = set()
    
    # Vertices to consider (not in S and not excluded)
    candidates = list(full - S)
    # Sort by degree descending (high-degree vertices are more constraining)
    candidates.sort(key=lambda x: degree.get(x, 0), reverse=True)
    
    best = list(S)
    best_size = len(S)
    
    # Precompute: for each candidate, which hyperedges it's in
    vertex_edges = {x: [] for x in candidates}
    for idx, (a, b, c) in enumerate(hyperedges):
        for v in (a, b, c):
            if v in vertex_edges:
                vertex_edges[v].append(idx)
    
    # For each pair in S, find which candidates conflict
    # A candidate x conflicts with S if there exist a,b in S with (a,b,x) bad
    def _conflicts_with_S(x, S_list):
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if _bad_triple(S_list[i], S_list[j], x):
                    return True
        return False
    
    # Filter candidates that can be added to S
    valid_candidates = []
    for x in candidates:
        if _conflicts_with_S(x, list(S)):
            continue
        valid_candidates.append(x)
    
    # Now do branch-and-bound on valid_candidates
    # State: current set, remaining candidates
    def _bbs(current_set, remaining, upper_bound):
        nonlocal best, best_size
        if not _within_deadline():
            return
        if len(current_set) + len(remaining) <= best_size:
            return
        if not remaining:
            if len(current_set) > best_size:
                best = list(current_set)
                best_size = len(current_set)
            return
        
        # Pick vertex with highest degree among remaining
        v = remaining[0]
        rest = remaining[1:]
        
        # Branch 1: include v
        new_set = current_set | {v}
        # Filter rest: remove any that conflict with v given current_set
        filtered_rest = []
        for u in rest:
            if u in new_set:
                continue
            # Check if u conflicts with new_set
            ok = True
            new_list = list(new_set)
            for i in range(len(new_list)):
                for j in range(i + 1, len(new_list)):
                    if _bad_triple(new_list[i], new_list[j], u):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                filtered_rest.append(u)
        
        # Upper bound: len(new_set) + len(filtered_rest)
        if len(new_set) + len(filtered_rest) > best_size:
            _bbs(new_set, filtered_rest, len(new_set) + len(filtered_rest))
        
        # Branch 2: exclude v
        _bbs(current_set, rest, len(current_set) + len(rest))
    
    _bbs(S, valid_candidates, len(S) + len(valid_candidates))
    return best

def construct():
    """Construct a large subset of [1,128] with no three distinct elements
    whose three pairwise gcds are all equal."""
    random.seed(42)
    full = list(range(1, 129))
    best_set = None
    best_size = 0
    
    # Build hypergraph once
    hyperedges, degree = _build_hypergraph()
    
    # Strategy 1: Structured - prime powers + 1
    def structured_prime_powers():
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
        prime_powers = set()
        for p in primes:
            val = p
            while val <= 128:
                prime_powers.add(val)
                val *= p
        prime_powers.add(1)
        S = list(prime_powers)
        # Validate
        S_list = S
        valid = True
        for a, b, c in combinations(S_list, 3):
            if _bad_triple(a, b, c):
                valid = False
                break
        return S if valid else None
    
    if _within_deadline():
        S_struct = structured_prime_powers()
        if S_struct is not None and len(S_struct) > best_size:
            best_size = len(S_struct)
            best_set = S_struct
    
    # Strategy 2: Greedy max conflict removal
    def greedy_max_conflict():
        S = set(full)
        # For each node, count bad triples
        node_conflict = {x: 0 for x in full}
        active_triples = set(hyperedges)
        for a, b, c in hyperedges:
            node_conflict[a] += 1
            node_conflict[b] += 1
            node_conflict[c] += 1
        
        order = sorted(full, key=lambda x: node_conflict[x], reverse=True)
        for v in order:
            if not _within_deadline():
                break
            if node_conflict[v] == 0:
                continue
            S.discard(v)
            to_remove = []
            for triple in active_triples:
                if v in triple:
                    to_remove.append(triple)
                    other_nodes = [x for x in triple if x != v]
                    for u in other_nodes:
                        node_conflict[u] = max(0, node_conflict[u] - 1)
            for triple in to_remove:
                active_triples.discard(triple)
        
        return list(S)
    
    if _within_deadline():
        S_greedy = greedy_max_conflict()
        if len(S_greedy) > best_size:
            best_size = len(S_greedy)
            best_set = S_greedy
    
    # Strategy 3: Randomized greedy - many passes with precomputed GCD
    if _within_deadline():
        for _ in range(500):
            if not _within_deadline():
                break
            order = list(full)
            random.shuffle(order)
            S_rand = _greedy_add(order)
            if len(S_rand) > best_size:
                best_size = len(S_rand)
                best_set = S_rand
    
    # Strategy 4: Local search on best so far
    if _within_deadline() and best_set is not None:
        S_improved = _local_search(best_set, max_iters=300)
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved
    
    # Strategy 5: Branch-and-bound (structural change)
    if _within_deadline():
        # Start from current best
        S_bb = _branch_and_bound(hyperedges, degree, initial_set=best_set)
        if S_bb and len(S_bb) > best_size:
            best_size = len(S_bb)
            best_set = S_bb
    
    # Final local search
    if _within_deadline() and best_set is not None:
        S_final = _local_search(best_set, max_iters=200)
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_set = S_final
    
    # Ensure nonempty and valid
    if best_set is None or len(best_set) == 0:
        return [1]
    
    # Verify validity
    S_list = list(best_set)
    valid = True
    for a, b, c in combinations(S_list, 3):
        if _bad_triple(a, b, c):
            valid = False
            break
    
    if not valid:
        # Remove offending elements one by one
        for i in range(len(S_list)):
            candidate = S_list[:i] + S_list[i + 1:]
            c_valid = True
            for a, b, c in combinations(candidate, 3):
                if _bad_triple(a, b, c):
                    c_valid = False
                    break
            if c_valid:
                best_set = candidate
                break
        else:
            return [1]
    
    return best_set
# EVOLVE-BLOCK-END
