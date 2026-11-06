"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_square(n):
    """Check if n is a perfect square using integer sqrt."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _build_conflict_graph():
    """
    Build the conflict graph where an edge (a, b) exists if a+b is a perfect square.
    We want a maximum independent set in this graph (no edges within the set).
    Vertices are 1..256.
    """
    n = 256
    # conflict[i] is a bitmask (as a Python int) where bit j is set if vertex (i+1) conflicts with vertex (j+1)
    # Actually, let's use 0-indexed: vertex i represents number (i+1)
    conflict = [0] * n
    
    squares = set()
    for k in range(1, 23):  # 1^2 to 22^2 = 484, max sum is 256+256=512
        squares.add(k * k)
    
    for i in range(n):
        a = i + 1
        for j in range(n):
            b = j + 1
            if a + b in squares:
                conflict[i] |= (1 << j)
    
    return conflict, n

def _max_clique_complement(conflict, n):
    """
    Find maximum independent set in conflict graph = maximum clique in complement graph.
    Complement graph: edge (i,j) exists if NOT (a+b is square).
    We use a branch-and-bound maximum clique algorithm on the complement.
    
    For maximum clique, we want a set of vertices where every pair is connected in the complement,
    i.e., no pair has a+b as a perfect square. This is exactly the independent set in the original.
    
    We'll implement a simple branch and bound with coloring bound.
    """
    # Complement adjacency: comp_adj[i] has bit j set if i and j are NOT in conflict
    comp_adj = [0] * n
    for i in range(n):
        # All bits set except self and conflict bits
        all_bits = (1 << n) - 1
        comp_adj[i] = all_bits & ~(conflict[i] | (1 << i))
    
    best_clique = []
    best_size = 0
    
    def expand(clique, candidates):
        nonlocal best_clique, best_size
        
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        
        # Coloring bound: greedily color the candidates to get an upper bound on clique size
        # from this subgraph
        # We process candidates in order of their index for determinism
        cand_list = []
        for i in range(n):
            if (candidates >> i) & 1:
                cand_list.append(i)
        
        # Greedy coloring: assign colors to vertices in cand_list
        # color[i] = color number assigned to vertex i
        colors = {}
        used_colors = [set() for _ in range(n)]  # used_colors[c] = set of vertices with color c
        
        # Order by degree in the induced subgraph for better bounds (descending)
        cand_list_sorted = sorted(cand_list, key=lambda v: bin(comp_adj[v] & candidates).count('1'), reverse=True)
        
        color_order = []  # list of (vertex, color)
        for v in cand_list_sorted:
            # Find the smallest color that doesn't conflict
            assigned = False
            for c in range(len(used_colors)):
                # Check if any vertex with color c is adjacent to v in comp graph
                conflict_with_color = False
                for u in used_colors[c]:
                    if (comp_adj[v] >> u) & 1:
                        conflict_with_color = True
                        break
                if not conflict_with_color:
                    used_colors[c].add(v)
                    colors[v] = c
                    color_order.append((v, c))
                    assigned = True
                    break
            if not assigned:
                c = len(used_colors)
                used_colors.append({v})
                colors[v] = c
                color_order.append((v, c))
        
        max_color = max(colors.values()) + 1 if colors else 0
        
        if len(clique) + max_color <= best_size:
            return
        
        # Branch: try adding vertices in reverse color order
        for v, c in reversed(color_order):
            if len(clique) + c <= best_size:
                break
            
            new_clique = clique + [v]
            new_candidates = candidates & comp_adj[v]
            expand(new_clique, new_candidates)
    
    all_candidates = (1 << n) - 1
    expand([], all_candidates)
    
    return best_clique

def _verify_solution(solution):
    """Verify that no pair (a,b) with a<=b in solution has a+b as a perfect square."""
    for i in range(len(solution)):
        for j in range(i, len(solution)):
            a, b = solution[i], solution[j]
            if _is_square(a + b):
                return False
    return True

def _greedy_heuristic(conflict, n):
    """Greedy heuristic to get a good initial solution."""
    # Order vertices by degree (number of conflicts) ascending - pick least constrained first
    degrees = [bin(conflict[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda i: degrees[i])
    
    selected = set()
    for v in order:
        # Check if v conflicts with any selected vertex
        can_add = True
        for u in selected:
            if (conflict[v] >> u) & 1:
                can_add = False
                break
        if can_add:
            selected.add(v)
    
    return [v + 1 for v in selected]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    conflict, n = _build_conflict_graph()
    
    # Get a good initial solution via greedy
    initial = _greedy_heuristic(conflict, n)
    best_solution = initial
    best_size = len(initial)
    
    # Try to improve using branch and bound on complement (maximum clique)
    # But this might be too slow for n=256. Let's try a local search approach instead.
    
    # Local search: start from greedy, try to add vertices or swap
    solution_set = set(initial)
    
    # Try to add vertices that don't conflict
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if v in solution_set:
                continue
            # Check if v can be added
            can_add = True
            for u in solution_set:
                if (conflict[v] >> u) & 1:
                    can_add = False
                    break
            if can_add:
                solution_set.add(v)
                improved = True
                if time.time() > deadline:
                    return [v + 1 for v in solution_set]
    
    best_solution = [v + 1 for v in solution_set]
    best_size = len(best_solution)
    
    # Try local search: remove one vertex, add two? Or swap?
    # Simple hill climbing: try removing a vertex and then greedily adding
    for _ in range(50):  # Limited iterations
        if time.time() > deadline:
            break
        
        # Try each possible removal
        for v in list(solution_set):
            trial_set = solution_set - {v}
            
            # Try to greedily add vertices back
            for u in range(n):
                if u in trial_set:
                    continue
                can_add = True
                for w in trial_set:
                    if (conflict[u] >> w) & 1:
                        can_add = False
                        break
                if can_add:
                    trial_set.add(u)
            
            if len(trial_set) > len(solution_set):
                solution_set = trial_set
                best_solution = [v + 1 for v in solution_set]
                best_size = len(solution_set)
                break
    
    # Final verification
    if not _verify_solution(best_solution):
        # Fallback to a known safe solution
        return list(range(1, 257, 3))
    
    return best_solution

# EVOLVE-BLOCK-END
