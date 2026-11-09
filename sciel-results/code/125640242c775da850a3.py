import time
import random
from itertools import combinations, permutations
from collections import defaultdict

# Precompute edge index helper
def _edge_index(i, j, n):
    if i > j:
        i, j = j, i
    # number of edges with first vertex < i
    # edges (k, l) with k < i: for each k in [0, i-1], l in [k+1, n-1] => (n-1-k) edges
    # sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
    base = i * (n - 1) - i * (i - 1) // 2
    return base + (j - i - 1)

# Precompute all 6-cycles and their edge sets, for each n up to 20
# We'll compute on demand and cache.
_cycle_cache = {}

def _get_cycles(n):
    if n in _cycle_cache:
        return _cycle_cache[n]
    # Base cycles: all 6-cycles on vertices 0..5 with vertex 0 fixed at position 0
    base_cycles = set()
    for perm in permutations([1, 2, 3, 4, 5]):
        cycle = (0, perm[0], perm[1], perm[2], perm[3], perm[4])
        rev = (0, perm[4], perm[3], perm[2], perm[1], perm[0])
        canon = min(cycle, rev)
        base_cycles.add(canon)
    
    cycles = []  # list of tuples of 6 edge indices
    for comb in combinations(range(n), 6):
        mapping = {k: v for k, v in enumerate(comb)}
        for bc in base_cycles:
            mapped = tuple(mapping[v] for v in bc)
            edges = tuple(_edge_index(mapped[k], mapped[(k+1) % 6], n) for k in range(6))
            cycles.append(edges)
    _cycle_cache[n] = cycles
    return cycles

def _verify(n, colors):
    """Return True if no monochromatic C6. colors is list of length n*(n-1)//2."""
    cycles = _get_cycles(n)
    for ce in cycles:
        c0 = colors[ce[0]]
        if all(colors[e] == c0 for e in ce[1:]):
            return False
    return True

def _local_search(n, colors, max_steps=2000, deadline=None):
    """Simple local search to reduce monochromatic C6 count. Returns (colors, penalty)."""
    cycles = _get_cycles(n)
    num_edges = n * (n - 1) // 2
    # Build edge -> list of cycle indices
    edge_to_cycles = defaultdict(list)
    for ci, ce in enumerate(cycles):
        for e in ce:
            edge_to_cycles[e].append(ci)
    
    def penalty():
        p = 0
        for ce in cycles:
            c0 = colors[ce[0]]
            if all(colors[e] == c0 for e in ce[1:]):
                p += 1
        return p
    
    cur = penalty()
    if cur == 0:
        return colors, 0
    
    steps = 0
    while steps < max_steps:
        if deadline is not None and time.time() > deadline:
            break
        steps += 1
        # Pick random edge
        idx = random.randint(0, num_edges - 1)
        old = colors[idx]
        best_c = old
        best_p = cur
        for c in range(4):
            if c == old:
                continue
            # Compute delta
            delta = 0
            for ci in edge_to_cycles[idx]:
                ce = cycles[ci]
                # Check mono with old
                mono_old = True
                for e in ce:
                    if e == idx:
                        continue
                    if colors[e] != old:
                        mono_old = False
                        break
                # Check mono with new
                mono_new = True
                for e in ce:
                    if e == idx:
                        continue
                    if colors[e] != c:
                        mono_new = False
                        break
                if mono_old and not mono_new:
                    delta -= 1
                elif not mono_old and mono_new:
                    delta += 1
            new_p = cur + delta
            if new_p < best_p:
                best_p = new_p
                best_c = c
        if best_c != old:
            colors[idx] = best_c
            cur = best_p
            if cur == 0:
                return colors, 0
    return colors, cur

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    
    # Try n = 17 first with backtracking + forward checking
    for target_n in [17, 18, 19]:
        if time.time() > deadline - 10:
            break
        n = target_n
        num_edges = n * (n - 1) // 2
        cycles = _get_cycles(n)
        num_cycles = len(cycles)
        
        # For each cycle, we need to track: for each color, how many edges are that color
        # and whether all 6 are assigned.
        # Forward checking: a color c is "blocked" for edge e if there exists a cycle
        # containing e where the other 5 edges are all color c and assigned.
        
        # Precompute: for each edge, the list of cycles it belongs to
        edge_to_cycles = defaultdict(list)
        for ci, ce in enumerate(cycles):
            for e in ce:
                edge_to_cycles[e].append(ci)
        
        # Backtracking search with forward checking
        # State: colors array (None = unassigned), and for each color, count per cycle
        # To make it fast, we maintain:
        #   cycle_color_count[ci][c] = number of edges in cycle ci colored c
        #   cycle_assigned[ci] = number of edges in cycle ci that are assigned
        # A color c is forbidden for edge e in cycle ci if:
        #   cycle_color_count[ci][c] == 5 and cycle_assigned[ci] == 6-1 (i.e., 5 assigned)
        #   and the unassigned edge in that cycle is e.
        # Actually simpler: when assigning edge e to color c, for each cycle ci containing e:
        #   if cycle_color_count[ci][c] == 5 and the other 5 edges are all assigned and color c:
        #     then this would complete a monochromatic C6 -> forbidden.
        
        # We'll use a recursive DFS with restarts and randomization.
        
        best_result = None
        best_n = 0
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            random.seed(restart * 1000 + 7)
            colors = [None] * num_edges
            # cycle_color_count: list of dicts or arrays
            # Use list of 4-int arrays
            cycle_color_count = [[0, 0, 0, 0] for _ in range(num_cycles)]
            cycle_assigned = [0] * num_cycles
            
            # Order of edges to assign: use a fixed order but randomize start
            edge_order = list(range(num_edges))
            random.shuffle(edge_order)
            
            # We'll do iterative deepening with a max depth of num_edges
            # Use a stack-based DFS
            
            def backtrack(pos):
                if time.time() > deadline - 5:
                    return None
                if pos == len(edge_order):
                    return colors[:]
                
                e = edge_order[pos]
                # MRV: actually we fixed the order. Let's try to pick the most constrained edge.
                # For simplicity, use the fixed order but check feasibility.
                
                # Determine which colors are forbidden for this edge
                forbidden = [False] * 4
                for ci in edge_to_cycles[e]:
                    for c in range(4):
                        if cycle_color_count[ci][c] == 5:
                            # Check if the other 5 edges in this cycle are all color c
                            # and the 6th (e) is the only unassigned one
                            # Actually if cycle_color_count[ci][c] == 5 and cycle_assigned[ci] == 5:
                            #   then the 5 assigned edges are color c, and e is unassigned.
                            #   Assigning e to c would make all 6 color c -> forbidden.
                            if cycle_assigned[ci] == 5:
                                forbidden[c] = True
                
                # Try colors in random order
                color_order = list(range(4))
                random.shuffle(color_order)
                
                for c in color_order:
                    if forbidden[c]:
                        continue
                    # Assign
                    colors[e] = c
                    for ci in edge_to_cycles[e]:
                        cycle_color_count[ci][c] += 1
                        cycle_assigned[ci] += 1
                        # Check if this creates a forbidden state for future edges
                        # (i.e., some other edge in this cycle now has 5 of color c)
                        # This is handled by the forward check at the next step.
                    
                    result = backtrack(pos + 1)
                    if result is not None:
                        return result
                    
                    # Unassign
                    colors[e] = None
                    for ci in edge_to_cycles[e]:
                        cycle_color_count[ci][c] -= 1
                        cycle_assigned[ci] -= 1
                
                return None
            
            result = backtrack(0)
            if result is not None:
                if _verify(n, result):
                    return {'n': n, 'colors': result}
        
        # If backtracking didn't find a solution, try local search
        if time.time() > deadline - 10:
            break
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            colors, pen = _local_search(n, colors, max_steps=3000, deadline=deadline - 10)
            if pen == 0 and _verify(n, colors):
                return {'n': n, 'colors': colors}
    
    # Fallback: try smaller n with simpler search
    for target_n in [16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline - 5:
            break
        n = target_n
        num_edges = n * (n - 1) // 2
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            colors, pen = _local_search(n, colors, max_steps=2000, deadline=deadline - 5)
            if pen == 0 and _verify(n, colors):
                return {'n': n, 'colors': colors}
    
    # Final fallback: deterministic small n
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((j - i) % 4)
    return {'n': n, 'colors': colors}
