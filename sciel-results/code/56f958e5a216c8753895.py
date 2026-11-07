"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
ALL_VERTS = frozenset(range(N))
EDGE_SET = set(frozenset(e) for e in P['edges'])
ADJ = defaultdict(set)
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

EDGE_LOOKUP = set()
for u, v in P['edges']:
    EDGE_LOOKUP.add((u, v))
    EDGE_LOOKUP.add((v, u))


def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)


def _find_ham_cycle_warnsdorff(used_edges, deadline, max_steps=200000, start=None):
    """Find a Hamiltonian cycle avoiding used_edges using Warnsdorff's heuristic."""
    n = N
    if start is None:
        start = 0
    
    # Build available adjacency
    avail = {}
    for v in range(n):
        avail[v] = [u for u in ADJ[v] if _edge_key(v, u) not in used_edges]
    
    # Check basic feasibility: every vertex must have at least 2 available neighbors
    for v in range(n):
        if len(avail[v]) < 2:
            return None
    
    path = [start]
    visited = {start}
    steps = 0
    
    # Iterative DFS with Warnsdorff's heuristic (fewest unvisited neighbors first)
    stack = [(start, list(avail[start]))]
    
    while stack:
        if time.time() > deadline:
            return None
        steps += 1
        if steps > max_steps:
            return None
        
        current, neighbors = stack[-1]
        
        if len(path) == n:
            last = path[-1]
            if _edge_key(last, start) not in used_edges and (last, start) in EDGE_LOOKUP:
                return list(path)
            # Dead end
            path.pop()
            visited.discard(current)
            stack.pop()
            continue
        
        # Sort neighbors by Warnsdorff: fewest unvisited neighbors first
        def warnsdorff_key(v):
            cnt = 0
            for u in avail[v]:
                if u not in visited and u != start:
                    cnt += 1
            return cnt
        
        sorted_neighbors = sorted(neighbors, key=warnsdorff_key)
        
        found_next = False
        for nxt in sorted_neighbors:
            if nxt in visited:
                continue
            if nxt == start:
                continue
            
            visited.add(nxt)
            path.append(nxt)
            
            # Check if we can continue
            if len(path) == n:
                if _edge_key(nxt, start) not in used_edges and (nxt, start) in EDGE_LOOKUP:
                    return list(path)
                # Dead end at full length
                path.pop()
                visited.discard(nxt)
                continue
            
            # Check if nxt has any unvisited neighbors (other than start)
            has_more = False
            for nnxt in avail[nxt]:
                if nnxt not in visited and nnxt != start:
                    has_more = True
                    break
            
            if has_more:
                stack.append((nxt, list(avail[nxt])))
                found_next = True
                break
            else:
                # Dead end, backtrack
                path.pop()
                visited.discard(nxt)
        
        if not found_next:
            path.pop()
            visited.discard(current)
            stack.pop()
    
    return None


def _find_ham_cycle_random(used_edges, deadline, max_steps=100000, start=None):
    """Find a Hamiltonian cycle with random neighbor ordering."""
    n = N
    if start is None:
        start = random.randint(0, n - 1)
    
    avail = {}
    for v in range(n):
        avail[v] = [u for u in ADJ[v] if _edge_key(v, u) not in used_edges]
    
    for v in range(n):
        if len(avail[v]) < 2:
            return None
        random.shuffle(avail[v])
    
    path = [start]
    visited = {start}
    steps = 0
    stack = [(start, list(avail[start]))]
    
    while stack:
        if time.time() > deadline:
            return None
        steps += 1
        if steps > max_steps:
            return None
        
        current, neighbors = stack[-1]
        
        if len(path) == n:
            last = path[-1]
            if _edge_key(last, start) not in used_edges and (last, start) in EDGE_LOOKUP:
                return list(path)
            path.pop()
            visited.discard(current)
            stack.pop()
            continue
        
        found_next = False
        for nxt in neighbors:
            if nxt in visited:
                continue
            if nxt == start:
                continue
            
            visited.add(nxt)
            path.append(nxt)
            
            if len(path) == n:
                if _edge_key(nxt, start) not in used_edges and (nxt, start) in EDGE_LOOKUP:
                    return list(path)
                path.pop()
                visited.discard(nxt)
                continue
            
            has_more = False
            for nnxt in avail[nxt]:
                if nnxt not in visited and nnxt != start:
                    has_more = True
                    break
            
            if has_more:
                stack.append((nxt, list(avail[nxt])))
                found_next = True
                break
            else:
                path.pop()
                visited.discard(nxt)
        
        if not found_next:
            path.pop()
            visited.discard(current)
            stack.pop()
    
    return None


def _cycle_edges(cycle):
    """Get the set of edge keys used by a cycle."""
    edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        edges.add(_edge_key(u, v))
    return edges


def _residual_feasible(used_edges):
    """Quick check: can the residual graph still support a Hamiltonian cycle?"""
    for v in range(N):
        deg = 0
        for u in ADJ[v]:
            if _edge_key(v, u) not in used_edges:
                deg += 1
        if deg < 2:
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Phase 1: Generate many candidate cycles from the full graph
    candidates = []
    seen_cycle_signatures = set()
    
    # First, try deterministic Warnsdorff from multiple starts
    for s in range(N):
        if time.time() > deadline - 30:
            break
        c = _find_ham_cycle_warnsdorff(set(), deadline=min(deadline - 25, time.time() + 3), max_steps=150000, start=s)
        if c is not None:
            # Canonical form for dedup
            sig = frozenset(_cycle_edges(c))
            if sig not in seen_cycle_signatures:
                seen_cycle_signatures.add(sig)
                candidates.append(c)
    
    # Random restarts to get more diverse candidates
    random.seed(42)
    for attempt in range(200):
        if time.time() > deadline - 20:
            break
        if len(candidates) >= 30:
            break
        c = _find_ham_cycle_random(set(), deadline=min(deadline - 18, time.time() + 2), max_steps=80000)
        if c is not None:
            sig = frozenset(_cycle_edges(c))
            if sig not in seen_cycle_signatures:
                seen_cycle_signatures.add(sig)
                candidates.append(c)
    
    if not candidates:
        # Fallback
        fallback = list(range(N))
        return [fallback]
    
    # Phase 2: Greedy selection with lookahead
    # Try to pick 3 edge-disjoint cycles
    best_packing = []
    
    # Sort candidates by some heuristic (e.g., prefer ones that leave good residual)
    # For each candidate, count how many other candidates are edge-disjoint from it
    def count_disjoint(c1):
        e1 = _cycle_edges(c1)
        cnt = 0
        for c2 in candidates:
            e2 = _cycle_edges(c2)
            if not e1 & e2:
                cnt += 1
        return cnt
    
    # Greedy: pick the cycle that has the most disjoint partners
    remaining_candidates = list(candidates)
    used_edges = set()
    
    for round_num in range(3):
        if not remaining_candidates:
            break
        if time.time() > deadline - 10:
            break
        
        # Score each remaining candidate
        best_c = None
        best_score = -1
        best_disjoint_count = -1
        
        for c in remaining_candidates:
            e_c = _cycle_edges(c)
            if e_c & used_edges:
                continue
            # Count how many other remaining candidates are disjoint from this one
            disjoint_cnt = 0
            for c2 in remaining_candidates:
                if c2 is c:
                    continue
                e2 = _cycle_edges(c2)
                if not (e_c | e2) & used_edges and not (e_c & e2):
                    disjoint_cnt += 1
            score = disjoint_cnt
            if score > best_score or (score == best_score and best_c is None):
                best_score = score
                best_c = c
                best_disjoint_count = score
        
        if best_c is None:
            break
        
        used_edges |= _cycle_edges(best_c)
        best_packing.append(best_c)
        remaining_candidates = [c for c in remaining_candidates if c is not best_c]
        
        # Check if residual is still feasible
        if not _residual_feasible(used_edges) and round_num < 2:
            # Try to continue anyway, but be cautious
            pass
    
    if len(best_packing) >= 1:
        result = [list(c) for c in best_packing[:3]]
        for i, cycle in enumerate(result):
            result[i] = [int(v) for v in cycle]
        return result
    
    # Fallback: return first candidate
    return [list(candidates[0])]


# EVOLVE-BLOCK-END
