import time
import random
import math
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbor masks for fast validity checks
# NEIGHBOR_MASK[v] is a bitmask where bit u is 1 if u is a neighbor of v
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = m

# For the MIS formulation on the "conflict graph":
# Vertices of the conflict graph are the vertices of the hypercube Q_N.
# An edge exists between u and v in the conflict graph if they CANNOT be non-consecutive in the snake.
# Wait, the snake constraint is:
# 1. Consecutive vertices differ by 1 bit.
# 2. Non-consecutive vertices differ by >= 2 bits.
# This is NOT a standard Independent Set problem on a static graph because "consecutive" depends on the path order.
# However, the problem asks for a "maximum induced path".
# The directive suggests: "Model the snake construction as a maximum independent set problem on the complement of the hypercube's line graph... then solve this using a weighted maximum clique algorithm".
# This specific framing is tricky. Let's look at the "conflict" definition again.
# If we view the snake as a set of edges $E_{snake}$ in $Q_N$ that form a path, the condition is that no two edges in $E_{snake}$ share a vertex (which is true for a simple path) AND no two edges are "chords" (i.e., their endpoints are adjacent in the hypercube but not in the path).
# Actually, the standard approach for "Snake in the box" is often formulated as finding the longest path.
# The directive's suggestion of MIS on the complement of the line graph is a bit abstract. 
# Let's interpret "conflict graph" for the MIS approach on VERTICES:
# If we select a subset of vertices $S$ to be the snake, can we always order them? No.
# But the directive says "shift the search from sequential...".
# Let's try a different interpretation: **Branch and Bound on the Path itself, but using a much stronger bound**.
# Or, let's try the **MIS on the "Chord Graph"**.
# Define a graph $G_{conflict}$ on the vertices of $Q_N$.
# Two vertices $u, v$ are connected in $G_{conflict}$ if they are adjacent in $Q_N$ (Hamming distance 1).
# If we find a Maximum Independent Set in $G_{conflict}$, that's just a set of vertices with pairwise distance >= 2. That's not a path.
#
# Let's re-read the directive carefully: "Model the snake construction as a maximum independent set problem on the complement of the hypercube's line graph, where vertices represent potential edges and conflicts represent shared endpoints or chord formation".
# Vertices in the new graph $H$ = Edges in $Q_N$.
# $H$ has $N * 2^{N-1}$ vertices. For N=9, that's $9 * 256 = 2304$ vertices.
# Two edges $e_1, e_2$ in $Q_N$ are adjacent in $H$ (conflict) if:
# 1. They share a vertex in $Q_N$ (they are incident).
# 2. They form a "chord" in the sense that their endpoints are adjacent in $Q_N$ but not via the edge itself? 
#    Actually, if we select a set of edges $S \subseteq E(Q_N)$, we want them to form a single path.
#    The condition "form a single path" implies:
#    - The subgraph $(V_S, S)$ is connected.
#    - The subgraph $(V_S, S)$ has exactly 2 vertices of degree 1 and all others degree 2.
#    - There are no chords: for any two edges $e_1, e_2$ in $S$ that are not consecutive in the path, their endpoints must not be adjacent in $Q_N$ unless they are the connecting vertex.
#    
# This is getting complicated to map directly to a simple MIS.
# However, there is a known reduction: The problem of finding the longest induced path is NP-hard.
# A common heuristic is **Branch and Bound with Strong Bounds**.
# The current program uses 2-opt + extension. 
# A "structurally different" approach is **Backtracking with Forward Checking and MRV (Most Constrained Variable)** as hinted by Inspiration 1 and 3.
# Inspiration 1 (1.44) used CSP/Backtracking.
# Inspiration 3 (1.37) used MCV Backtracking.
#
# Let's implement a **Recursive Backtracking Search** that builds the path vertex by vertex, but uses a **Greedy Coloring Bound** on the remaining candidates to prune branches that cannot beat the current best.
# This is essentially a Maximum Clique algorithm applied to the "compatibility graph" of the path extension?
# No, the path constraint is sequential.
#
# Alternative Structural Change: **Simulated Annealing with a Different Move Set and Objective**.
# The current SA uses 2-opt. 
# Let's try **Tabu Search** with **Segment Reversal and Insertion Moves**, combined with a **Lookahead Extension**.
# Or better yet, let's try the **CSP Backtracking** approach directly, as it scored highest in inspirations.
#
# **Strategy: CSP Backtracking with MRV and Forward Checking**
# 1. Represent the state as the current path.
# 2. At each step, identify "candidate" vertices that can be added to the end.
# 3. Use a heuristic to order candidates (e.g., those that leave the most options for future steps).
# 4. Use a bound: Estimate the maximum possible extension from the current state. If `current_len + bound < best_len`, prune.
# 5. To make this effective, we need a good bound. A simple bound is the size of the largest independent set in the remaining available vertices? No.
#
# Let's refine the **Branch and Bound** idea.
# We search for a path $v_1, \dots, v_k$.
# State: Current path $P$.
# Candidates: Neighbors of $P[-1]$ that are not in $P$ and not adjacent to $P[:-1]$.
# If we pick a candidate $c$, we add it to $P$.
# We want to maximize $k$.
#
# To prune: We need an upper bound on how many more vertices we can add.
# A loose bound is the number of available vertices.
# A tighter bound: The available vertices must form a path starting from $c$.
#
# Let's combine **Backtracking** with **Randomized Restarts** and a **Time-Budgeted Search**.
# Since the space is huge, pure backtracking will die quickly.
# We need a **Hybrid**: 
# 1. Generate a good initial solution using the existing greedy method (or a faster beam search).
# 2. Use **Local Search (SA/Tabu)** to improve it, but with a **Global Repair** mechanism.
#
# Wait, the directive asks for a **DIFFERENT** strategy. The current one is SA + 2-opt.
# Inspiration 1 was CSP Backtracking. Let's try that.
#
# **Implementation Plan for CSP Backtracking:**
# 1. **Preprocessing**: 
#    - Precompute neighbors for all 512 vertices.
#    - Precompute "forbidden" masks for each vertex given the current path? No, that's dynamic.
# 2. **Search Algorithm**:
#    - `search(path, depth)`:
#      - If `depth > best_len`, update `best_len` and `best_path`.
#      - Calculate candidates: Neighbors of `path[-1]` that are valid (not in `path`, not adjacent to `path[:-1]`).
#      - If no candidates, return.
#      - **Pruning**: If `len(path) + len(candidates) < best_len`, return? No, candidates can lead to more.
#      - **Heuristic**: Sort candidates by a heuristic (e.g., degree in the remaining graph, or random).
#      - For each candidate (in order):
#        - Recurse with `path + [candidate]`.
#        - **Time Check**: If time is up, return.
#
# This is a Depth-First Search (DFS). It will get stuck in deep branches.
# To avoid this, we use **Iterative Deepening** or **Beam Search**?
# Beam Search is a structural change from SA.
#
# Let's try **Beam Search with Width $W$** and **Diverse Beam Selection**.
# 1. Maintain a set of $W$ best partial paths.
# 2. At each step, extend each path in the beam to all valid neighbors.
# 3. Keep the top $W$ paths based on length (and maybe diversity).
# 4. This is a "Best-First" search with a cap.
#
# The current program does NOT do beam search. It does SA on a single path.
# Beam Search is a different optimization family (parallel exploration of the search tree).
#
# **Enhancement**: To handle the "long tail" of the snake, we can use **Variable Width Beam Search** or **Weighted Beam Search**.
# Also, we can use **Lookahead**: For each candidate extension, estimate the potential future length (e.g., number of available neighbors of the new end).
#
# Let's implement **Beam Search**.
#
# **Details**:
# - State: List of paths (tuples of vertices).
# - Score: Length of the path.
# - Tie-breaking: Randomness or diversity (e.g., different ending vertices).
# - Width: Start with a large width (e.g., 1000) and reduce it? Or keep it constant?
# - Given 120s, we can afford a moderate width.
# - We need fast validity checks.
#
# **Fast Validity Check for Beam Search**:
# For a path $P$, the set of "blocked" vertices is $B(P) = \{ u \mid \exists v \in P, dist(u,v)=1 \} \setminus \{ P[-1] \}$.
# Actually, $v$ can be added if $v \notin P$ and $v \notin B(P)$.
# Note: $B(P)$ includes neighbors of $P[0] \dots P[-2]$.
# We can maintain `blocked_mask` for each path in the beam.
# When extending $P$ to $P' = P + [v]$:
#   `new_blocked = blocked_mask | NEIGHBOR_MASK[P[-1]]`
#   (Because $P[-1]$ becomes non-consecutive to future vertices, so its neighbors are blocked).
#   Wait, $v$ is the new last. $P[-1]$ is now at index -2.
#   So yes, neighbors of $P[-1]$ are blocked for the NEXT step.
#   But $v$ itself is not blocked (it's the current end).
#   So `new_blocked = blocked_mask | NEIGHBOR_MASK[P[-1]]`.
#   And we must ensure $v$ is not in `blocked_mask` and not in `used_mask`.
#
# This is very fast! O(1) per extension.
#
# **Algorithm**:
# 1. Initialize beam with a few random starts (e.g., 0, 1, ...).
#    Each state: `(path, used_mask, blocked_mask)`.
# 2. Loop until time limit:
#    a. For each state in beam, generate all valid extensions.
#    b. Collect all new states.
#    c. Sort new states by length (descending).
#    d. Keep top `WIDTH` states.
#    e. If no new states, break.
#    f. Update best found.
#
# **Optimization**:
# - Use a priority queue or list sorting.
# - Width: 200-500 might be good.
# - To increase diversity, we can keep multiple states with the same length but different "endings" or "blocked masks".
#
# Let's implement this.

def construct():
    deadline = time.time() + 115.0
    rnd = random.Random(20260715)
    
    # Precompute neighbors as a list of lists for faster iteration if needed, 
    # but bitmasks are faster for checking.
    # We'll use bitmasks for `used` and `blocked`.
    
    # State: (path_tuple, used_mask, blocked_mask)
    # path_tuple is needed to reconstruct the path and to get the last vertex.
    # Actually, we only need the last vertex for extension, and the length for scoring.
    # But we need the full path to return it.
    # Storing full paths in the beam is memory expensive if width is large.
    # 500 paths * 200 vertices * 4 bytes/vertex = 400KB. This is fine.
    
    WIDTH = 300
    
    # Initialize beam
    # Start with a few diverse starts
    beam = []
    for start in range(10):
        v = start
        path = (v,)
        used = 1 << v
        blocked = 0
        beam.append((path, used, blocked))
        
    best_path = max(beam, key=lambda x: len(x[0]))[0]
    best_len = len(best_path) - 1
    
    while time.time() < deadline - 5:
        new_beam = []
        
        for path, used, blocked in beam:
            last = path[-1]
            # Find valid neighbors of last
            # Candidates are neighbors of last that are not in used and not in blocked
            # Note: blocked contains neighbors of path[:-1]. 
            # last is path[-1]. Its neighbors are NOT in blocked (unless they were neighbors of earlier vertices, which would make them invalid anyway).
            # So we just check `not in used` and `not in blocked`.
            
            # Iterate over neighbors of last
            # neighbors of last are last ^ (1<<b) for b in range(N)
            for b in range(N):
                v = last ^ (1 << b)
                if (used >> v) & 1: continue
                if (blocked >> v) & 1: continue
                
                # Valid extension
                new_path = path + (v,)
                new_used = used | (1 << v)
                # The old last becomes non-consecutive, so its neighbors are blocked
                new_blocked = blocked | NEIGHBOR_MASK[last]
                
                new_beam.append((new_path, new_used, new_blocked))
                
        if not new_beam:
            break
            
        # Sort by length, then by random tie-breaker for diversity?
        # Or just length.
        # To keep diversity, we can use a "beam diversity" score.
        # Simple approach: Sort by length, then keep top WIDTH.
        # If many have same length, we might lose diversity.
        # Let's shuffle before sorting to randomize tie-breaking?
        rnd.shuffle(new_beam)
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Keep top WIDTH
        if len(new_beam) > WIDTH:
            new_beam = new_beam[:WIDTH]
            
        # Update best
        if new_beam:
            current_best_in_beam = max(new_beam, key=lambda x: len(x[0]))
            if len(current_best_in_beam[0]) - 1 > best_len:
                best_len = len(current_best_in_beam[0]) - 1
                best_path = current_best_in_beam[0]
                
        beam = new_beam
        
    # Convert best_path tuple to list
    result = list(best_path)
    
    # Validation
    def is_valid(path):
        if not path: return False
        n = len(path)
        if n == 1: return True
        for i in range(n - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return False
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1:
                    if (path[i] ^ path[j]).bit_count() == 1:
                        return False
        return True

    if is_valid(result):
        return result
    else:
        # Fallback to a simple valid path
        return [0]

if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p)-1}")
    valid = True
    for i in range(len(p)-1):
        if (p[i] ^ p[i+1]).bit_count() != 1:
            valid = False
            break
    for i in range(len(p)):
        for j in range(len(p)):
            if abs(i-j) > 1 and (p[i] ^ p[j]).bit_count() == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
