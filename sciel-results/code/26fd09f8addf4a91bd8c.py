"""Randomized search with diverse heuristics to find long simple path in pinned graph."""
import random
import time
from collections import deque

N = 100
EDGES = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]

ADJ = [set() for _ in range(N)]
DEG = [0] * N
for u, v in EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)
    DEG[u] += 1
    DEG[v] += 1

# Find connected components
def find_components():
    visited = set()
    comps = []
    for s in range(N):
        if s not in visited:
            comp = set()
            stack = [s]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.add(u)
                for v in ADJ[u]:
                    if v not in visited:
                        stack.append(v)
            comps.append(comp)
    return comps

def build_skeleton(comp):
    """
    Decompose component into skeleton:
    - Branching nodes (degree >= 3 in original graph)
    - Degree-2 chains between branching nodes or self-loops
    Returns:
      - branch_nodes: list of branching node IDs
      - chains: dict mapping (branch_a, branch_b) -> list of vertices in chain (excluding endpoints)
      - chain_length: dict mapping (branch_a, branch_b) -> length of chain (number of edges)
    """
    comp_set = set(comp)
    branch_nodes = [v for v in comp if DEG[v] >= 3]
    
    if not branch_nodes:
        # Entire component is a cycle or path
        return comp, {}, {}
    
    # Build chains between branching nodes
    chains = {}
    chain_length = {}
    visited_in_chain = set()
    
    for bn in branch_nodes:
        for nb in ADJ[bn]:
            if nb in visited_in_chain:
                continue
            # Follow the chain from nb until we hit another branching node
            chain = [nb]
            prev = bn
            curr = nb
            while curr not in branch_nodes and curr not in visited_in_chain:
                visited_in_chain.add(curr)
                chain.append(curr)
                # Find next node
                nexts = [x for x in ADJ[curr] if x != prev]
                if len(nexts) == 0:
                    break
                prev = curr
                curr = nexts[0]
            if curr in branch_nodes and curr != bn:
                key = tuple(sorted([bn, curr]))
                chains[key] = chain[1:]  # Exclude starting branch node
                chain_length[key] = len(chain)
            elif curr in branch_nodes and curr == bn:
                # Self-loop chain (cycle through bn)
                key = (bn, bn)
                chains[key] = chain[1:]
                chain_length[key] = len(chain)
    
    return branch_nodes, chains, chain_length

def skeleton_dfs(branch_nodes, chains, chain_length, start_branch, deadline):
    """
    DFS on skeleton tree to find longest path.
    Each move along a chain adds chain_length vertices.
    """
    best_path = []
    visited_branches = set()
    
    def dfs(current_branch, path_vertices, visited):
        nonlocal best_path
        if time.time() > deadline:
            return
        if len(path_vertices) > len(best_path):
            best_path = path_vertices[:]
        
        # Try all unvisited neighboring branches
        for nb in ADJ[current_branch]:
            if nb not in visited:
                key = tuple(sorted([current_branch, nb]))
                if key in chains:
                    chain_verts = chains[key]
                    # Check if chain is fully unvisited
                    if all(v not in visited for v in chain_verts):
                        new_visited = visited | {nb} | set(chain_verts)
                        new_path = path_vertices + chain_verts + [nb]
                        dfs(nb, new_path, new_visited)
        
        # Also try going through a self-loop if exists
        key_self = (current_branch, current_branch)
        if key_self in chains:
            chain_verts = chains[key_self]
            if all(v not in visited for v in chain_verts):
                new_path = path_vertices + chain_verts + [current_branch]
                dfs(current_branch, new_path, visited | set(chain_verts))
    
    dfs(start_branch, [start_branch], {start_branch})
    return best_path

def greedy_warnsdorff(start, rng):
    """Warnsdorff's rule: pick neighbor with fewest unvisited neighbors."""
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = min(cand, key=lambda w: (len(ADJ[w] - seen), rng.random()))
        path.append(v)
        seen.add(v)
    return path

def greedy_max_degree(start, rng):
    """Pick neighbor with highest degree."""
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = max(cand, key=lambda w: (DEG[w], rng.random()))
        path.append(v)
        seen.add(v)
    return path

def greedy_random(start, rng):
    """Random neighbor choice."""
    seen = {start}
    path = [start]
    while True:
        cand = list(ADJ[path[-1]] - seen)
        if not cand:
            break
        v = rng.choice(cand)
        path.append(v)
        seen.add(v)
    return path

def greedy_mixed(start, rng):
    """Mixed heuristic: combine Warnsdorff and degree."""
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        # Score: negative unvisited count + degree * weight
        weights = rng.random() * 2
        v = max(cand, key=lambda w: (-len(ADJ[w] - seen), DEG[w] * weights, rng.random()))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path, seen):
    """Extend path from both ends using Warnsdorff."""
    path = list(path)
    seen = set(seen)
    improved = True
    while improved:
        improved = False
        # Extend front
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        # Extend back
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path

def local_improve(path, seen, rng, deadline):
    """Try to insert unvisited vertices into the path, extend ends, and try reversals."""
    path = list(path)
    seen = set(seen)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try inserting each unvisited vertex (multiple passes)
        for u in list(unvisited):
            found = False
            # Try all positions, not just first valid one
            best_pos = -1
            best_score = -1
            for i in range(len(path) - 1):
                if path[i] in ADJ[u] and path[i+1] in ADJ[u]:
                    # Score by how many other unvisited neighbors u has (prefer fewer to avoid trapping)
                    score = len(ADJ[u] - seen - {path[i], path[i+1]})
                    if best_pos == -1 or score < best_score:
                        best_pos = i + 1
                        best_score = score
            if best_pos >= 0:
                path.insert(best_pos, u)
                seen.add(u)
                unvisited.remove(u)
                improved = True
                found = True
            # Also try at ends
            if not found:
                if path[0] in ADJ[u]:
                    path.insert(0, u)
                    seen.add(u)
                    unvisited.remove(u)
                    improved = True
                elif path[-1] in ADJ[u]:
                    path.append(u)
                    seen.add(u)
                    unvisited.remove(u)
                    improved = True
        # Extend ends with Warnsdorff
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path

def dfs_search(start, seen, path, best, deadline, max_depth, node_count):
    """DFS to find longer path from current state with node budget."""
    if time.time() > deadline or node_count[0] > 2000000:
        return
    if len(path) > len(best):
        best.clear()
        best.extend(path)
    if len(path) >= max_depth:
        return
    cand = ADJ[path[-1]] - seen
    # Warnsdorff ordering: prefer neighbors with fewer unvisited options
    cand = sorted(cand, key=lambda w: (len(ADJ[w] - seen), w))
    for v in cand:
        node_count[0] += 1
        path.append(v)
        seen.add(v)
        dfs_search(start, seen, path, best, deadline, max_depth, node_count)
        path.pop()
        seen.remove(v)
        if time.time() > deadline or node_count[0] > 2000000:
            return

def construct():
    t0 = time.time()
    deadline = t0 + 75  # Safety margin (evaluator kills at 90s)
    
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    rng = random.Random(42)
    
    for comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        comp_list = sorted(comp)
        target = comp_size  # We want to find a Hamiltonian path in the component
        
        # Phase 1: Skeleton-based search
        branch_nodes, chains, chain_length = build_skeleton(comp)
        if branch_nodes:
            for bn in branch_nodes:
                if time.time() > deadline:
                    break
                skeleton_path = skeleton_dfs(branch_nodes, chains, chain_length, bn, deadline)
                if len(skeleton_path) > len(best_path):
                    best_path = skeleton_path[:]
        
        # Phase 2: Diverse greedy starts
        for start in comp_list:
            if time.time() > deadline:
                break
            for func in [greedy_warnsdorff, greedy_max_degree, greedy_random, greedy_mixed]:
                if time.time() > deadline:
                    break
                p = func(start, rng)
                p = extend_both_ends(p, set(p))
                if len(p) > len(best_path):
                    best_path = p[:]
        
        # Phase 3: Local improvement on best paths found
        if len(best_path) > 0 and len(best_path) < target:
            if time.time() > deadline:
                break
            p = local_improve(best_path, set(best_path), rng, deadline)
            if len(p) > len(best_path):
                best_path = p
        
        # Phase 4: If not Hamiltonian, try DFS from promising starts
        if len(best_path) < target:
            # Try DFS from endpoints of current best and from all vertices
            remaining_time = deadline - time.time()
            if remaining_time > 0.5:
                # Try starting from each vertex in the component
                for start in comp_list:
                    if time.time() > deadline:
                        break
                    remaining = deadline - time.time()
                    if remaining < 0.1:
                        break
                    # Scale budget based on remaining time
                    budget = min(5.0, remaining * 0.3)
                    dl = time.time() + budget
                    node_count = [0]
                    local_best = list(best_path)
                    dfs_search(start, {start}, [start], local_best, dl, target, node_count)
                    if len(local_best) > len(best_path):
                        best_path = local_best
                        # Try to improve the new best with local_improve
                        if len(best_path) < target and time.time() < deadline:
                            p2 = local_improve(best_path, set(best_path), rng, deadline)
                            if len(p2) > len(best_path):
                                best_path = p2
                        if len(best_path) >= target:
                            break
    
    # Final safety check
    if not best_path:
        best_path = [0]
    
    # Final polish pass: try to extend/improve one more time
    if len(best_path) < max(len(c) for c in comps) and time.time() < deadline - 1.0:
        p_final = local_improve(best_path, set(best_path), rng, deadline - 0.5)
        if len(p_final) > len(best_path):
            best_path = p_final
    
    # Verify path validity
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in ADJ[best_path[i]]:
            return [0]
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        return fixed
    
    return best_path
