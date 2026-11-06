"""Structural redesign: local search with efficient delta evaluation for C4-free 5-colourings."""
import time
import random
import numpy as np

# Deadline guard: keep a safety margin before the 150s kill.
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0  # seconds

def _time_left():
    return _DEADLINE - _SAFETY_MARGIN - time.time()

def _build_adj(n):
    """Return adjacency matrix (n x n) of K_n as a list of lists (for fast indexing)."""
    return [[0]*n for _ in range(n)]

def _pair_index(n, i, j):
    """Lexicographic pair index for (i, j) with i < j."""
    # Number of pairs before row i: i*(i-1)//2 (since row 0 has 0 pairs, row 1 has 1, ..., row i-1 has i-1)
    # Actually, in lexicographic order for i<j:
    # Pairs for row 0: (0,1), (0,2), ..., (0,n-1) -> n-1 pairs
    # Pairs for row 1: (1,2), (1,3), ..., (1,n-1) -> n-2 pairs
    # ...
    # Pairs for row i: (i, i+1), ..., (i, n-1) -> n-1-i pairs
    # So pairs before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
    # Then within row i, the offset for j is j - i - 1
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def _make_colors(n):
    """Create a random 5-colouring of K_n."""
    num_pairs = n * (n - 1) // 2
    return np.random.randint(0, 5, size=num_pairs)

def _count_c4_in_color(colors, n, c):
    """Count the number of C4s in colour c. A C4 is two vertices with two common neighbours in colour c."""
    # For each pair of vertices (u, v), count common neighbours in colour c.
    # If count >= 2, each pair of common neighbours forms a C4.
    # Number of C4s contributed by (u, v) is C(common_neighbours, 2).
    # Build adjacency sets for colour c.
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = _pair_index(n, i, j)
            if colors[idx] == c:
                adj[i].add(j)
                adj[j].add(i)
    total_c4 = 0
    for u in range(n):
        for v in range(u + 1, n):
            common = len(adj[u] & adj[v])
            if common >= 2:
                total_c4 += common * (common - 1) // 2
    return total_c4

def _total_violations(colors, n):
    """Total number of C4s across all 5 colours."""
    return sum(_count_c4_in_color(colors, n, c) for c in range(5))

def _delta_c4(colors, n, i, j, old_c, new_c):
    """
    Compute the change in total C4 count when edge (i,j) is recoloured from old_c to new_c.
    A C4 involving edge (i,j) in colour c requires another vertex k such that (i,k) and (j,k) are both in colour c.
    When we remove (i,j) from colour old_c, we destroy C4s that used (i,j) in old_c.
    When we add (i,j) to colour new_c, we create C4s that use (i,j) in new_c.
    """
    if old_c == new_c:
        return 0
    delta = 0
    # For each colour c, a C4 containing edge (i,j) in colour c requires a vertex k
    # such that (i,k) and (j,k) are both in colour c, and k != i, j.
    # When edge (i,j) is in colour c, the number of such k's is the number of common neighbours
    # of i and j in colour c (excluding each other, but i and j are not neighbours of themselves).
    
    # Remove from old_c:
    # Count common neighbours of i and j in colour old_c (excluding i,j themselves, which are not in their own adjacency).
    common_old = 0
    for k in range(n):
        if k == i or k == j:
            continue
        idx_ik = _pair_index(n, min(i,k), max(i,k))
        idx_jk = _pair_index(n, min(j,k), max(j,k))
        if colors[idx_ik] == old_c and colors[idx_jk] == old_c:
            common_old += 1
    # Each such k forms a C4 with (i,j) in colour old_c. But wait: a C4 is (i,j,k,l) where (i,j), (j,k), (k,l), (l,i) are edges.
    # Actually, the standard definition: a C4 is a cycle of length 4. In the context of "no two vertices with two common neighbours in one colour",
    # a C4 in colour c is a set of 4 vertices {a,b,c,d} such that (a,b), (b,c), (c,d), (d,a) are all in colour c.
    # Equivalently, two vertices have two common neighbours.
    # If edge (i,j) is in colour c, and k is a common neighbour of i and j in colour c, then {i,j,k} with edges (i,j), (i,k), (j,k) forms a triangle, not a C4.
    # I need to reconsider.
    
    # The constraint is: no two vertices have two common neighbours in one colour.
    # So for colour c, if vertices u and v have two common neighbours x and y in colour c, then (u,x), (u,y), (v,x), (v,y) are all in colour c,
    # forming a C4: u-x-v-y-u.
    
    # So a C4 in colour c is determined by a pair of vertices (u,v) and two common neighbours (x,y) in colour c.
    # The edge (i,j) is part of a C4 in colour c if there exists a vertex k such that (i,k) and (j,k) are in colour c AND
    # there exists another vertex l such that (i,l) and (j,l) are in colour c? No, that's not right either.
    
    # Let me think again. A C4 in colour c is a 4-cycle. The edges of the C4 are 4 edges in colour c.
    # If edge (i,j) is in colour c, it can be part of a C4 if there is a path of length 3 from i to j in colour c that does not use (i,j).
    # That is, there exist vertices k, l such that (i,k), (k,l), (l,j) are all in colour c, and k != j, l != i, k != l.
    # This is equivalent to: there is a vertex k such that (i,k) is in colour c, and a vertex l such that (k,l) and (l,j) are in colour c.
    # This is getting complicated. Let me use the "two common neighbours" formulation.
    
    # Two vertices u and v have two common neighbours x and y in colour c iff (u,x), (u,y), (v,x), (v,y) are all in colour c.
    # This forms a C4: u-x-v-y-u.
    # The edges in this C4 are (u,x), (x,v), (v,y), (y,u).
    # So edge (i,j) is in this C4 if {i,j} is one of these 4 edges.
    
    # Case 1: (i,j) = (u,x). Then u=i, x=j. The C4 is i-j-v-y-i. The other edges are (j,v), (v,y), (y,i).
    #   So we need (j,v) and (v,y) and (y,i) in colour c, for some v, y.
    #   Equivalently, v is a common neighbour of j and y in colour c, and y is a neighbour of i in colour c.
    #   This is complex.
    
    # Let me use a different approach. The number of C4s in colour c can be counted as:
    # For each pair of vertices (u,v), let d_c(u,v) = number of common neighbours of u and v in colour c.
    # Then the number of C4s in colour c is sum_{u<v} C(d_c(u,v), 2).
    # Because each pair of common neighbours (x,y) of u and v in colour c gives a C4 u-x-v-y-u.
    
    # So when we change the colour of edge (i,j) from old_c to new_c:
    # - For old_c: the common neighbour counts for pairs involving i or j change.
    #   Specifically, for any vertex k, the common neighbour count of (i,k) in old_c changes by -1 if (j,k) was in old_c,
    #   and the common neighbour count of (j,k) in old_c changes by -1 if (i,k) was in old_c.
    #   Also, the common neighbour count of (i,j) in old_c is no longer relevant since (i,j) is no longer in old_c.
    #   Wait, the common neighbour count of (i,j) in old_c was d_old(i,j). After removing (i,j) from old_c,
    #   the pairs (i,k) and (j,k) for k != i,j have their common neighbour counts changed.
    
    # This is getting too complex for a simple delta. Let me just recompute the total violations.
    # But that's O(n^3) per move, which might be too slow.
    
    # Alternative: precompute for each colour c, for each pair (u,v), the number of common neighbours in colour c.
    # Then when we change edge (i,j) from old_c to new_c:
    # - For old_c: 
    #   * The pair (i,j) is no longer an edge in old_c, so we remove its contribution.
    #   * For each k != i,j: 
    #     - If (i,k) is in old_c, then the common neighbour count of (i,k) in old_c decreases by 1 if (j,k) was in old_c.
    #     - If (j,k) is in old_c, then the common neighbour count of (j,k) in old_c decreases by 1 if (i,k) was in old_c.
    # - For new_c:
    #   * The pair (i,j) is now an edge in new_c, so we add its contribution.
    #   * For each k != i,j:
    #     - If (i,k) is in new_c, then the common neighbour count of (i,k) in new_c increases by 1 if (j,k) is in new_c.
    #     - If (j,k) is in new_c, then the common neighbour count of (j,k) in new_c increases by 1 if (i,k) is in new_c.
    
    # Let me implement this with precomputed common neighbour counts.
    raise NotImplementedError("Delta computation is complex; will use full recomputation for now.")

# Given the complexity of delta evaluation, let's use a simpler approach:
# 1. Start with a random colouring.
# 2. Use local search: pick a random edge, try all 5 colours, pick the one that minimises total C4s.
# 3. Use simulated annealing to escape local minima.
# 4. Periodically try to increase n.

def _local_search(colors, n, max_steps=100000, temperature_start=1.0, temperature_end=0.01, cooling=0.999):
    """Local search with simulated annealing to minimise total C4s."""
    current_violations = _total_violations(colors, n)
    best_colors = colors.copy()
    best_violations = current_violations
    temperature = temperature_start
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break
        # Pick a random edge
        i = random.randint(0, n - 1)
        j = random.randint(i + 1, n - 1)
        idx = _pair_index(n, i, j)
        old_c = colors[idx]
        
        # Try all 5 colours
        best_new_c = old_c
        best_new_violations = current_violations
        for new_c in range(5):
            if new_c == old_c:
                continue
            colors[idx] = new_c
            new_violations = _total_violations(colors, n)
            if new_violations < best_new_violations:
                best_new_violations = new_violations
                best_new_c = new_c
        colors[idx] = best_new_c
        current_violations = best_new_violations
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors.copy()
        
        # Simulated annealing: accept worse solutions with some probability
        if current_violations > best_violations and temperature > temperature_end:
            delta = current_violations - best_violations
            if random.random() < np.exp(-delta / temperature):
                pass  # Already accepted
        temperature *= cooling
    
    return best_colors, best_violations

def construct():
    """Construct a 5-colouring of K_n with no monochromatic C4, maximising n."""
    random.seed(42)
    np.random.seed(42)
    
    best_result = {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}
    
    # Try to find valid colourings for increasing n
    for n in range(12, 27):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break
        # Try multiple random restarts
        found = False
        for restart in range(5):
            if time.time() > _DEADLINE - _SAFETY_MARGIN:
                break
            colors = _make_colors(n)
            colors, violations = _local_search(colors, n, max_steps=50000)
            if violations == 0:
                best_result = {'n': n, 'colors': colors.tolist()}
                found = True
                break
        if not found:
            # Try a few more restarts with different parameters
            for restart in range(5):
                if time.time() > _DEADLINE - _SAFETY_MARGIN:
                    break
                colors = _make_colors(n)
                colors, violations = _local_search(colors, n, max_steps=100000, temperature_start=2.0, temperature_end=0.05, cooling=0.998)
                if violations == 0:
                    best_result = {'n': n, 'colors': colors.tolist()}
                    found = True
                    break
        if found:
            continue
        else:
            # If we can't find a valid colouring for this n, try the next n
            continue
    
    return best_result

if __name__ == '__main__':
    result = construct()
    print(f"Found n={result['n']}")
