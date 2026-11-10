"""Rewritten program: large clique subdivisions in a pinned graph.
Structural change: model the problem as a k-core / k-truss constrained branch set,
then route disjoint paths via a backtracking-with-BFS engine that prefers
shortest paths and uses MRV ordering.  The k-core filter prunes the search
space for the disjoint-path routing phase.
"""
import time
import sys
from itertools import combinations
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

# EVOLVE-BLOCK-START
N = P['n']
adj = [set() for _ in range(N)]
adj_list = [list() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)
for i in range(N):
    adj_list[i] = sorted(adj[i])

DEADLINE = time.time() + 82  # 82 s safety margin before 90 s kill


def time_left():
    return DEADLINE - time.time()


def compute_core_numbers():
    """Compute k-core number for every vertex."""
    deg = [len(adj[i]) for i in range(N)]
    core = [0] * N
    bucket = [set() for _ in range(N + 1)]
    for v in range(N):
        bucket[deg[v]].add(v)
    cur = N - 1
    for v in range(N):
        while cur >= 0 and not bucket[cur]:
            cur -= 1
        if cur < 0:
            break
        u = bucket[cur].pop()
        core[u] = cur
        for w in adj[u]:
            if core[w] == 0:
                bucket[deg[w]].discard(w)
                deg[w] -= 1
                bucket[deg[w]].add(w)
    return core


def compute_truss_numbers():
    """Compute 3-truss number (edge truss) for every vertex."""
    # Count triangles per edge
    tri_count = {}
    for u in range(N):
        for v in adj[u]:
            if u < v:
                common = adj[u] & adj[v]
                tri_count[(u, v)] = len(common)
    # Truss number of an edge = 1 + min(tri_count of incident edges in triangle)
    # Simplified: vertex truss number = max over incident edges of (1 + min triangle count)
    truss = [0] * N
    for u in range(N):
        for v in adj[u]:
            tc = tri_count.get((min(u, v), max(u, v)), 0)
            if tc + 1 > truss[u]:
                truss[u] = tc + 1
    return truss


def bfs_path(src, dst, blocked):
    """Find a simple path from src to dst avoiding blocked vertices (except src and dst)."""
    if src == dst:
        return [src]
    parent = {src: None}
    queue = deque([src])
    while queue:
        node = queue.popleft()
        if node == dst:
            path = []
            cur = dst
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for neighbor in adj_list[node]:
            if neighbor not in parent and neighbor not in blocked:
                parent[neighbor] = node
                queue.append(neighbor)
    return None


def find_paths_for_branches(branches):
    """
    Try to find disjoint internal paths for all pairs of branches.
    Returns dict of (u, v) -> path, or None if infeasible.
    Uses backtracking with MRV ordering and BFS for path finding.
    """
    branch_set = set(branches)
    pairs = list(combinations(branches, 2))

    # Identify direct edges and compute options for non-direct pairs
    pair_info = {}
    non_direct_pairs = []
    for u, v in pairs:
        if u in adj[v]:
            pair_info[(u, v)] = ('direct', None)
        else:
            # Find internal vertices adjacent to both (for length-2 paths)
            common = set()
            for w in adj_list[u]:
                if w not in branch_set and w in adj[v]:
                    common.add(w)
            pair_info[(u, v)] = ('need', common)
            non_direct_pairs.append((u, v))

    # Feasibility: every non-direct pair must have at least one common neighbor
    for pair in non_direct_pairs:
        if len(pair_info[pair][1]) == 0:
            return None

    # Sort non-direct pairs by fewest options first (MRV heuristic)
    sorted_non_direct = sorted(non_direct_pairs, key=lambda p: len(pair_info[p][1]))

    assignment = {}
    used_internals = set()

    def backtrack(idx):
        if time_left() < 0.5:
            return False
        if idx == len(sorted_non_direct):
            return True
        u, v = sorted_non_direct[idx]

        blocked = branch_set | used_internals
        blocked.discard(u)
        blocked.discard(v)

        # Try BFS path
        path = bfs_path(u, v, blocked)
        if path is None:
            return False

        internal = path[1:-1]

        # Verify path validity
        if len(internal) != len(set(internal)):
            return False
        for w in internal:
            if w in branch_set:
                return False

        for w in internal:
            used_internals.add(w)
        assignment[(u, v)] = internal

        if backtrack(idx + 1):
            return True

        for w in internal:
            used_internals.discard(w)
        del assignment[(u, v)]

        return False

    if not backtrack(0):
        return None

    result_paths = {}
    for pair in pairs:
        u, v = pair
        if pair_info[pair][0] == 'direct':
            result_paths[pair] = [u, v]
        else:
            internal = assignment[pair]
            result_paths[pair] = [u] + internal + [v]

    return result_paths


def check_feasible(branches):
    """Check if a branch set is feasible (can route disjoint paths)."""
    result = find_paths_for_branches(branches)
    return result is not None


def greedy_build_branches(core_nums, truss_nums, deadline_check=None):
    """
    Greedily build a branch set by adding vertices one at a time,
    checking feasibility after each addition.
    Uses k-core and k-truss numbers to prioritize candidates.
    Returns (branches, paths) or None.
    """
    deg = [len(adj[i]) for i in range(N)]

    # Sort by: core number descending, truss number descending, degree descending
    order = sorted(range(N), key=lambda x: (-core_nums[x], -truss_nums[x], -deg[x]))

    branches = []
    for candidate in order:
        if deadline_check and deadline_check() < 2:
            break
        trial = branches + [candidate]
        paths = find_paths_for_branches(trial)
        if paths is not None:
            branches = trial

    if len(branches) < 2:
        return None
    paths = find_paths_for_branches(branches)
    if paths is None:
        return None
    return branches, paths


def local_search(branches, paths, core_nums, truss_nums, deadline_check=None):
    """
    Try to improve by adding one more vertex or swapping.
    """
    current_n = len(branches)
    branch_set = set(branches)

    # Try adding one more vertex
    candidates = [v for v in range(N) if v not in branch_set]
    # Sort candidates by core/truss/degree
    deg = [len(adj[i]) for i in range(N)]
    candidates.sort(key=lambda x: (-core_nums[x], -truss_nums[x], -deg[x]))

    for candidate in candidates:
        if deadline_check and deadline_check() < 1:
            break
        trial = list(branches) + [candidate]
        trial_paths = find_paths_for_branches(trial)
        if trial_paths is not None:
            return trial, trial_paths

    # Try swapping
    for i in range(len(branches)):
        for candidate in candidates:
            if deadline_check and deadline_check() < 0.5:
                break
            trial = list(branches)
            trial[i] = candidate
            trial_paths = find_paths_for_branches(trial)
            if trial_paths is not None:
                # Same size; try to extend
                extended, ext_paths = local_search(trial, trial_paths, core_nums, truss_nums, deadline_check)
                if extended is not None and len(extended) > len(branches):
                    return extended, ext_paths
    return branches, paths


def search_max_branches():
    """
    Main search: try to find the maximum number of branches.
    """
    def deadline_check():
        return time_left()

    core_nums = compute_core_numbers()
    truss_nums = compute_truss_numbers()

    best_branches = None
    best_paths = None
    best_count = 1

    # Fallback: always valid
    u, v = P['edges'][0]
    fallback = ([u, v], [[u, v]])

    # Strategy 1: Greedy from highest core/truss/degree
    result = greedy_build_branches(core_nums, truss_nums, deadline_check)
    if result is not None:
        b, p = result
        if len(b) > best_count:
            best_count = len(b)
            best_branches = b
            best_paths = p

    # Strategy 2: Try extending from each edge
    if time_left() > 10:
        for (u, v) in P['edges']:
            if time_left() < 5:
                break
            trial = [u, v]
            paths = find_paths_for_branches(trial)
            if paths is None:
                continue
            extended, ext_paths = local_search(trial, paths, core_nums, truss_nums, deadline_check)
            if extended is not None and len(extended) > best_count:
                best_count = len(extended)
                best_branches = extended
                best_paths = ext_paths

    # Strategy 3: Random restarts
    if time_left() > 5:
        rng = random.Random(42)
        for _ in range(200):
            if time_left() < 3:
                break
            u = rng.randint(0, N - 1)
            v = rng.choice(adj_list[u])
            trial = [u, v]
            paths = find_paths_for_branches(trial)
            if paths is None:
                continue
            extended, ext_paths = local_search(trial, paths, core_nums, truss_nums, deadline_check)
            if extended is not None and len(extended) > best_count:
                best_count = len(extended)
                best_branches = extended
                best_paths = ext_paths

    # Final: also try to extend the best found
    if best_branches is not None and time_left() > 3:
        extended, ext_paths = local_search(best_branches, best_paths, core_nums, truss_nums, deadline_check)
        if extended is not None and len(extended) > best_count:
            best_count = len(extended)
            best_branches = extended
            best_paths = ext_paths

    if best_branches is None:
        return fallback

    return best_branches, best_paths


def construct():
    global DEADLINE
    DEADLINE = time.time() + 82

    branches, paths = search_max_branches()

    # Ensure types are correct
    branches_out = [int(b) for b in branches]
    paths_out = [[int(v) for v in p] for p in paths]

    return {'branches': branches_out, 'paths': paths_out}
# EVOLVE-BLOCK-END
