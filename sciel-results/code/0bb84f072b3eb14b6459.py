"""Packing edge-disjoint Hamiltonian cycles on the fixed 21-vertex host graph."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

# Precompute base adjacency (frozenset of neighbors) and edge set.
N = P['n']
BASE_ADJ = [set() for _ in range(N)]
EDGE_SET = set()
for a, b in P['edges']:
    BASE_ADJ[a].add(b)
    BASE_ADJ[b].add(a)
    EDGE_SET.add((a, b) if a < b else (b, a))


def _find_hamiltonian(remaining_edges, adj, deadline, rng, max_steps=5000000):
    """Find one Hamiltonian cycle using only edges in remaining_edges.
    Returns list of vertices (length N, distinct) or None.
    Warnsdorff-guided backtracking with cheap cut-set pruning and forced-move detection."""
    n = N
    # adjacency restricted to remaining edges
    adj = [set() for _ in range(n)]
    for (u, v) in remaining_edges:
        adj[u].add(v)
        adj[v].add(u)
    # quick degree feasibility: every vertex needs degree >= 2 in remaining graph
    for v in range(n):
        if len(adj[v]) < 2:
            return None

    best_cycle = None
    steps = 0

    def _warnsdorff_key(x):
        """Warnsdorff: number of unvisited neighbors of x.
        Tie-break: prefer candidates whose own forward options are fewer (more constrained)."""
        cnt = 0
        min_fwd = 99
        for y in adj[x]:
            if not used[y]:
                cnt += 1
                f = 0
                for z in adj[y]:
                    if not used[z] and z != x:
                        f += 1
                if f < min_fwd:
                    min_fwd = f
        return (cnt, min_fwd)

    def _cheap_prune(path_len):
        """Cheap pruning: check no unvisited vertex is isolated from the rest
        (i.e., has no unvisited neighbors AND no edge back to the current path end).
        Also check that if path_len == n-1, the last unvisited vertex connects to start."""
        if path_len >= n:
            return False
        unvisited_count = n - path_len
        if unvisited_count <= 1:
            if unvisited_count == 1:
                # The single unvisited vertex must connect to start
                for v in range(n):
                    if not used[v]:
                        return start in adj[v]
            return True
        # Check for isolated unvisited vertices
        for v in range(n):
            if not used[v]:
                has_unvis_nb = False
                for w in adj[v]:
                    if not used[w]:
                        has_unvis_nb = True
                        break
                if not has_unvis_nb:
                    return False
        return True

    def _forced_move_check(cand, cur, path_len):
        """Detect forced moves: unvisited vertices with exactly one unvisited neighbor.
        Also handle the closing constraint near the end."""
        if path_len < 4:
            return None
        unvisited = [v for v in range(n) if not used[v]]
        if len(unvisited) <= 1:
            return None
        forced = None
        for v in unvisited:
            unvis_neighbors = [u for u in adj[v] if not used[u]]
            if len(unvis_neighbors) == 1:
                only_nb = unvis_neighbors[0]
                if only_nb in cand:
                    forced = only_nb
                    break
        # Near the end, check if the last few unvisited vertices force a specific next move
        if path_len >= n - 3:
            # The remaining unvisited vertices must form a path from cur to start
            # Check if any candidate leads to a dead end
            for x in cand:
                if x not in unvisited:
                    continue
                # After going to x, can we still reach start through remaining unvisited?
                remaining_unvis = [v for v in unvisited if v != x]
                if len(remaining_unvis) == 0:
                    if start in adj[x]:
                        forced = x
                        break
                elif len(remaining_unvis) == 1:
                    v = remaining_unvis[0]
                    if v in adj[x] and start in adj[v]:
                        forced = x
                        break
                elif len(remaining_unvis) == 2:
                    v1, v2 = remaining_unvis
                    if (v1 in adj[x] and v2 in adj[v1] and start in adj[v2]) or \
                       (v2 in adj[x] and v1 in adj[v2] and start in adj[v1]):
                        forced = x
                        break
        return forced

    def backtrack():
        nonlocal steps, best_cycle, used_count
        if best_cycle is not None:
            return True
        if time.time() >= deadline:
            return False
        if len(path) == n:
            cur = path[-1]
            if start in adj[cur]:
                best_cycle = list(path)
                return True
            return False
        cur = path[-1]
        cand = [x for x in adj[cur] if not used[x]]
        if not cand:
            return False
        
        # Cheap cut-set pruning
        if not _cheap_prune(len(path)):
            return False
        
        # Forced-move detection
        forced = _forced_move_check(cand, cur, len(path))
        if forced is not None and forced in cand:
            cand = [forced]
        else:
            # Sort by Warnsdorff (fewest forward options first), with random tie-breaking
            cand.sort(key=lambda x: _warnsdorff_key(x))
            if len(cand) > 1:
                groups = {}
                for c in cand:
                    k = _warnsdorff_key(c)
                    groups.setdefault(k, []).append(c)
                new_cand = []
                for k in sorted(groups.keys()):
                    g = groups[k]
                    rng.shuffle(g)
                    new_cand.extend(g)
                cand = new_cand
        
        for x in cand:
            path.append(x)
            used[x] = True
            used_count += 1
            steps += 1
            if steps > max_steps:
                path.pop()
                used[x] = False
                used_count -= 1
                return False
            if backtrack():
                return True
            path.pop()
            used[x] = False
            used_count -= 1
        return False

    # Choose start vertex with minimum degree (more constrained)
    min_deg = min(len(adj[v]) for v in range(n))
    candidates_start = [v for v in range(n) if len(adj[v]) == min_deg]
    rng.shuffle(candidates_start)
    
    for start in candidates_start:
        if time.time() >= deadline:
            break
        best_cycle = None
        steps = 0
        path = [start]
        used = [False] * n
        used[start] = True
        used_count = 1
        backtrack()
        if best_cycle is not None:
            return best_cycle
    return None


def _is_valid_cycles(cycles, all_edges):
    seen = set()
    vset = None
    for cyc in cycles:
        if len(cyc) != N:
            return False
        if len(set(cyc)) != N:
            return False
        if vset is None:
            vset = set(cyc)
        elif set(cyc) != vset:
            return False
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (a, b) if a < b else (b, a)
            if e not in all_edges:
                return False
            if e in seen:
                return False
            seen.add(e)
    return True


def construct():
    deadline = time.time() + 55.0  # safety margin under 60s / 90s limits
    rng = random.Random(12345)
    remaining = set(EDGE_SET)
    cycles = []
    # seed fallback: one known cycle (full range) if it is valid in the graph
    seed_cycle = list(range(N))
    seed_ok = all((a, b) if a < b else (b, a) in EDGE_SET
                  for a, b in zip(seed_cycle, seed_cycle[1:] + seed_cycle[:1]))
    fallback = [seed_cycle] if seed_ok else []

    max_cycles = 3  # 70 edges / 21 per cycle ≈ 3.33, so max 3
    for cycle_idx in range(max_cycles):
        if time.time() >= deadline:
            break
        # Allocate time budget per cycle: leave 5s for remaining cycles + validation
        time_left = deadline - time.time()
        per_cycle_budget = min(time_left, (time_left - 3.0) / (max_cycles - cycle_idx)) if max_cycles > cycle_idx else time_left
        per_cycle_deadline = time.time() + max(1.0, per_cycle_budget)
        # try to find a cycle in the remaining edge set
        found = None
        # multiple restarts with different random seeds for diversity
        for attempt in range(200):
            if time.time() >= per_cycle_deadline:
                break
            attempt_rng = random.Random(12345 + attempt * 7919 + cycle_idx * 104729)
            found = _find_hamiltonian(remaining, None, per_cycle_deadline, attempt_rng)
            if found is not None:
                break
        if found is None:
            break
        # remove its edges from remaining
        cyc = found
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (a, b) if a < b else (b, a)
            remaining.discard(e)
        cycles.append(list(cyc))

    # Validate; if anything is off, fall back.
    if cycles and _is_valid_cycles(cycles, EDGE_SET):
        return [list(c) for c in cycles]
    if fallback:
        return [list(c) for c in fallback]
    # last resort: any valid single cycle via one more attempt
    remaining = set(EDGE_SET)
    for attempt in range(200):
        if time.time() >= deadline:
            break
        attempt_rng = random.Random(99999 + attempt * 31)
        f = _find_hamiltonian(remaining, None, deadline, attempt_rng)
        if f is not None and _is_valid_cycles([f], EDGE_SET):
            return [list(f)]
    return [list(range(N))]
# EVOLVE-BLOCK-END
