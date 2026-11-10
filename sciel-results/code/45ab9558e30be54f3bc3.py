import time
import random
import numpy as np
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def _edge_index(i, j):
    """Map pair (i,j) with i<j to edge index 0..119 in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_cycles():
    """Return list of cycles, each cycle is a tuple of 4 edge indices."""
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    cycles.append((_edge_index(a, b), _edge_index(b, c), _edge_index(c, d), _edge_index(a, d)))
                    cycles.append((_edge_index(a, b), _edge_index(b, d), _edge_index(c, d), _edge_index(a, c)))
                    cycles.append((_edge_index(a, c), _edge_index(b, c), _edge_index(b, d), _edge_index(a, d)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

# Precompute cycle matrix for vectorized operations
CYCLE_MATRIX = np.array(CYCLES, dtype=np.int32)  # shape (NUM_CYCLES, 4)

def _edge_to_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

EDGE_TO_CYCLES = _edge_to_cycles()

def _is_valid_vectorized(colors):
    """Check if coloring is valid (no 4-cycle has 4 distinct colors). Returns True if valid."""
    # colors: array of shape (NUM_EDGES,)
    # Get all cycle edge colors: shape (NUM_CYCLES, 4)
    cycle_colors = colors[CYCLE_MATRIX]  # shape (NUM_CYCLES, 4)
    # A cycle has 4 distinct colors iff all pairs are different
    # Check: for each cycle, are all 4 values distinct?
    # Use: max - min < 3 is not sufficient. Better: count distinct.
    # Vectorized: for each row, check if all 4 are distinct
    # c0!=c1, c0!=c2, c0!=c3, c1!=c2, c1!=c3, c2!=c3
    c0 = cycle_colors[:, 0]
    c1 = cycle_colors[:, 1]
    c2 = cycle_colors[:, 2]
    c3 = cycle_colors[:, 3]
    # A cycle is a violation if all 6 pairs are distinct
    violation = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
    return not np.any(violation)

def _count_violations_vectorized(colors):
    """Count number of 4-cycles with 4 distinct edge colors."""
    cycle_colors = colors[CYCLE_MATRIX]
    c0 = cycle_colors[:, 0]
    c1 = cycle_colors[:, 1]
    c2 = cycle_colors[:, 2]
    c3 = cycle_colors[:, 3]
    violation = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
    return int(np.sum(violation))

def _count_distinct(colors):
    return len(set(colors.tolist()))

def _algebraic_init():
    """
    Structured coloring using sum-based scheme.
    Color of edge (i,j) = (i + j) mod K for some K.
    This ensures that in any 4-cycle a-b-c-d-a:
      edges: (a,b), (b,c), (c,d), (d,a)
      colors: (a+b)%K, (b+c)%K, (c+d)%K, (d+a)%K
    For a 4-cycle to have 4 distinct colors, we'd need all these to be different.
    But note: (a+b) + (c+d) = (b+c) + (a+d) = a+b+c+d
    So if K is even, (a+b)%K and (c+d)%K could be equal when a+b+c+d ≡ 2*(a+b) (mod K), etc.
    
    Actually, let's use a different structure. 
    Use color = (i * j) mod K? No, that's not great either.
    
    Let me use: partition vertices into groups and use group-based coloring.
    Split 16 vertices into 4 groups of 4. Color edge by (group_i + group_j) mod M.
    Within a 4-cycle, if vertices are from groups g1,g2,g3,g4, the edge colors depend on group sums.
    
    Actually, the known good structure: use a 2-coloring pattern or modular arithmetic.
    Let's try: color(i,j) = (i + j) % 8. This gives at most 8 colors.
    
    Better: use a proper edge coloring of K16. K16 has chromatic index 15 (odd n) or 16 (even n).
    For K16, we can use a 1-factorization: 15 perfect matchings, each gets a distinct color.
    This gives 15 colors and is valid because any 4-cycle uses at most 2 edges from any matching.
    Wait, that's not right. In a 4-cycle a-b-c-d-a, the edges are ab, bc, cd, da.
    In a 1-factorization of K16, each matching covers all 16 vertices. 
    A 4-cycle has 4 edges. Each matching contributes at most 1 edge to the 4-cycle 
    (since a matching is a set of disjoint edges, and the 4-cycle has 4 vertices, 
    a matching can have at most 2 edges among these 4 vertices, but those 2 edges would 
    be either {ab,cd} or {ac,bd} or {ad,bc}).
    
    Hmm, let me think differently. The key insight from the knowledge:
    "Structured sum-based coloring guarantees 4-cycle property (diagonals share color)"
    
    So if we color such that in every 4-cycle, the two diagonals have the same color,
    then no 4-cycle can have 4 distinct colors (since at least two edges share a color).
    
    For a 4-vertex set {a,b,c,d}, the three 4-cycles are:
    1. a-b-c-d-a: diagonals are ac, bd
    2. a-b-d-c-a: diagonals are ad, bc  
    3. a-c-b-d-a: diagonals are ab, cd
    
    So we need: color(ac) == color(bd), color(ad) == color(bc), color(ab) == color(cd) for every 4-set.
    This is a very strong constraint. It means the coloring is determined by a "symmetric" structure.
    
    One way: color(i,j) = f(i) + f(j) for some function f: {0..15} -> Z_K.
    Then color(ac) = f(a)+f(c), color(bd) = f(b)+f(d).
    For these to be equal for all a<b<c<d, we need f(a)+f(c) = f(b)+f(d) for all such quadruples.
    This implies f is linear: f(i) = i * c for some constant c (mod K).
    Then color(i,j) = (i+j)*c mod K.
    
    With f(i) = i, color(i,j) = (i+j) mod K.
    Check: for a<b<c<d, 
      color(ac) = (a+c) mod K, color(bd) = (b+d) mod K.
      These are equal iff a+c ≡ b+d (mod K), i.e., a-b ≡ d-c (mod K).
      This is NOT true in general! So (i+j) mod K does NOT satisfy the diagonal condition.
    
    Let me reconsider. Maybe the structure is different.
    
    Alternative: use a 4-partition. Split vertices into 4 sets S0, S1, S2, S3 of size 4.
    Color edge (i,j) by (group(i) + group(j)) mod 4.
    For a 4-cycle with vertices in groups g1,g2,g3,g4:
      Cycle 1 (a-b-c-d-a): edges in groups (g1,g2),(g2,g3),(g3,g4),(g4,g1)
      Diagonals: (g1,g3) and (g2,g4)
      color(diag1) = (g1+g3) mod 4, color(diag2) = (g2+g4) mod 4
      These are equal iff g1+g3 ≡ g2+g4 (mod 4), i.e., g1-g2 ≡ g4-g3 (mod 4).
      Not guaranteed.
    
    OK let me just use a simple valid coloring and then optimize.
    Start with all edges color 0 (valid, 1 color), then use local search to add colors.
    """
    # Start with a simple valid coloring: all zeros
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    return colors

def _local_search_vectorized(colors, deadline, best_colors, best_distinct):
    """
    Vectorized local search. Pick random edges, try random new colors,
    check validity efficiently.
    """
    current_distinct = _count_distinct(colors)
    if current_distinct > best_distinct:
        best_distinct = current_distinct
        best_colors[:] = colors
    
    color_counts = np.zeros(MAX_COLOR + 1, dtype=np.int32)
    for c in colors:
        color_counts[c] += 1
    
    T = 2.0
    cooling = 0.99995
    min_T = 0.01
    
    # Precompute for each edge which cycles it's in (as array)
    edge_cycle_indices = []
    for e in range(NUM_EDGES):
        edge_cycle_indices.append(np.array(EDGE_TO_CYCLES[e], dtype=np.int32))
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T  # Don't go below, just keep going
        
        # Pick a random edge
        e_idx = np.random.randint(0, NUM_EDGES)
        old_color = colors[e_idx]
        
        # Pick a new color
        if np.random.random() < 0.6:
            # Pick an unused color
            unused_mask = color_counts == 0
            if np.any(unused_mask):
                unused_indices = np.where(unused_mask)[0]
                new_color = int(unused_indices[np.random.randint(0, len(unused_indices))])
            else:
                new_color = int(np.random.randint(0, MAX_COLOR + 1))
        else:
            new_color = int(np.random.randint(0, MAX_COLOR + 1))
        
        if new_color == old_color:
            continue
        
        # Check if change creates violations
        # Get cycles containing this edge
        cys = edge_cycle_indices[e_idx]
        if len(cys) == 0:
            # Edge not in any cycle (shouldn't happen for K16)
            colors[e_idx] = new_color
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            new_distinct = current_distinct
            if color_counts[old_color] == 0:
                new_distinct -= 1
            if color_counts[new_color] == 1:
                new_distinct += 1
            current_distinct = new_distinct
            if new_distinct > best_distinct:
                best_distinct = new_distinct
                best_colors[:] = colors
            T *= cooling
            continue
        
        # Get the cycle colors for affected cycles
        affected_cycles = CYCLE_MATRIX[cys]  # shape (num_affected, 4)
        old_cycle_colors = colors[affected_cycles]  # shape (num_affected, 4)
        
        # Determine which position in each cycle the edge is at
        # For each affected cycle, find position of e_idx
        # Create new colors array
        new_cycle_colors = old_cycle_colors.copy()
        # For each row, find where e_idx is
        for k in range(len(cys)):
            row = affected_cycles[k]
            pos = np.where(row == e_idx)[0][0]
            new_cycle_colors[k, pos] = new_color
        
        # Check old violations
        oc0 = old_cycle_colors[:, 0]
        oc1 = old_cycle_colors[:, 1]
        oc2 = old_cycle_colors[:, 2]
        oc3 = old_cycle_colors[:, 3]
        old_viol = (oc0 != oc1) & (oc0 != oc2) & (oc0 != oc3) & (oc1 != oc2) & (oc1 != oc3) & (oc2 != oc3)
        
        # Check new violations
        nc0 = new_cycle_colors[:, 0]
        nc1 = new_cycle_colors[:, 1]
        nc2 = new_cycle_colors[:, 2]
        nc3 = new_cycle_colors[:, 3]
        new_viol = (nc0 != nc1) & (nc0 != nc2) & (nc0 != nc3) & (nc1 != nc2) & (nc1 != nc3) & (nc2 != nc3)
        
        # Count old and new violations among affected cycles
        old_viol_count = int(np.sum(old_viol))
        new_viol_count = int(np.sum(new_viol))
        
        # Check if this move creates any NEW violations
        # A cycle is newly violated if it wasn't before but is now
        newly_violated = (~old_viol) & new_viol
        if np.any(newly_violated):
            # This move creates at least one new violation
            # Could still accept with probability if we're using simulated annealing on violations
            # But for simplicity, reject
            T *= cooling
            continue
        
        # Move is safe (no new violations)
        # Apply
        colors[e_idx] = new_color
        color_counts[old_color] -= 1
        color_counts[new_color] += 1
        
        new_distinct = current_distinct
        if color_counts[old_color] == 0:
            new_distinct -= 1
        if color_counts[new_color] == 1:
            new_distinct += 1
        
        old_distinct = current_distinct
        current_distinct = new_distinct
        
        if new_distinct > best_distinct:
            best_distinct = new_distinct
            best_colors[:] = colors
        elif new_distinct == old_distinct:
            # Accept with some probability to explore
            if np.random.random() < 0.3:
                pass  # Keep the change
            else:
                # Revert
                colors[e_idx] = old_color
                color_counts[old_color] += 1
                color_counts[new_color] -= 1
                current_distinct = old_distinct
        else:
            # new_distinct < old_distinct, accept with low probability
            delta = new_distinct - old_distinct
            if np.random.random() < np.exp(delta / T):
                pass  # Keep
            else:
                colors[e_idx] = old_color
                color_counts[old_color] += 1
                color_counts[new_color] -= 1
                current_distinct = old_distinct
        
        T *= cooling
    
    return best_colors, best_distinct

def _multi_start_local_search(deadline, num_starts):
    """
    Run multiple local searches from different starting points.
    """
    global_best_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    global_best_distinct = 1
    
    start_time = time.time()
    
    for start_idx in range(num_starts):
        # Check time
        if time.time() > deadline:
            break
        
        # Create a starting coloring
        # Use different strategies for different starts
        if start_idx == 0:
            # All zeros
            colors = np.zeros(NUM_EDGES, dtype=np.int32)
        elif start_idx == 1:
            # Modular coloring: (i+j) % 8
            colors = np.zeros(NUM_EDGES, dtype=np.int32)
            for i in range(N):
                for j in range(i+1, N):
                    e = _edge_index(i, j)
                    colors[e] = (i + j) % 8
            # Check validity
            if not _is_valid_vectorized(colors):
                colors = np.zeros(NUM_EDGES, dtype=np.int32)
        elif start_idx == 2:
            # 1-factorization based: color by matching index
            # For K16, use 15 perfect matchings
            colors = np.zeros(NUM_EDGES, dtype=np.int32)
            # Simple 1-factorization: for k in 0..14, matching k connects i to (i+k+1)%15 for i in 0..14, and 15 is connected to... 
            # Actually for K_{2m}, 1-factorization: label vertices 0..2m-1 and infinity
            # For K16, m=8. Label 0..14 and inf.
            # Matching k: for i in 0..14, connect i to (i+k)%15. Inf connects to k.
            # But we have 16 vertices 0..15. Let's map: vertex 15 = inf.
            for k in range(15):
                for i in range(15):
                    j = (i + k) % 15
                    if i < j:
                        e = _edge_index(i, j)
                        colors[e] = k
                    elif j < i:
                        e = _edge_index(j, i)
                        colors[e] = k
                # Inf (vertex 15) connects to vertex k
                e = _edge_index(k, 15)
                colors[e] = k
            # Check validity
            if not _is_valid_vectorized(colors):
                colors = np.zeros(NUM_EDGES, dtype=np.int32)
        else:
            # Random valid-ish start: use (i*j) % 16 or similar
            colors = np.zeros(NUM_EDGES, dtype=np.int32)
            for i in range(N):
                for j in range(i+1, N):
                    e = _edge_index(i, j)
                    colors[e] = (i * j) % 16
            if not _is_valid_vectorized(colors):
                colors = np.zeros(NUM_EDGES, dtype=np.int32)
        
        # Run local search with a time budget
        time_per_start = max(1.0, (deadline - time.time()) / max(1, num_starts - start_idx))
        sub_deadline = time.time() + time_per_start
        
        best_colors, best_distinct = _local_search_vectorized(
            colors.copy(), sub_deadline, 
            global_best_colors.copy() if global_best_distinct > 1 else np.zeros(NUM_EDGES, dtype=np.int32),
            global_best_distinct
        )
        
        # Update global best
        if best_distinct > global_best_distinct:
            global_best_distinct = best_distinct
            global_best_colors[:] = best_colors
    
    return global_best_colors, global_best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Multi-start local search
    best_colors, best_distinct = _multi_start_local_search(deadline, num_starts=5)
    
    # Final validation
    if not _is_valid_vectorized(best_colors):
        # Fallback to all zeros
        return [0] * NUM_EDGES
    
    return best_colors.tolist()
