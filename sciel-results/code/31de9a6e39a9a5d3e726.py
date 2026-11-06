import time
import random
import sys
import math

N = 9
SIZE = 1 << N

# Precompute neighbors for each vertex
NEIGHBORS = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

# Construct the Conflict Graph for Directed Edges
# A directed edge is (u, v) where v is a neighbor of u.
# We can represent a directed edge by (u, bit) where v = u ^ (1 << bit).
# Total directed edges: SIZE * N = 512 * 9 = 4608.

# Let's map each directed edge to an integer ID.
# ID = u * N + bit
# u is in [0, SIZE), bit is in [0, N)

NUM_EDGES = SIZE * N

def get_id(u, bit):
    return u * N + bit

def get_edge_info(e_id):
    u = e_id // N
    bit = e_id % N
    v = u ^ (1 << bit)
    return u, v, bit

# Precompute conflict bitsets for each directed edge.
# A directed edge e1 = (u, v) conflicts with e2 = (x, y) if:
# 1. They are the same edge (u==x and v==y) -> actually we want distinct vertices, so no repeats.
#    In a path, we cannot use the same directed edge twice.
#    Also, we cannot use (u,v) and (v,u) because that would imply u-v-u, which repeats u.
#    Wait, the path is a sequence of distinct vertices.
#    So if we have (u,v), we cannot have (v,u) later because that would require returning to u.
#    Also, we cannot have any edge that touches u or v in a way that creates a chord or repeat.
#    
#    Let's refine the conflict definition for a path of distinct vertices.
#    Let the path be v_0, v_1, ..., v_k.
#    The directed edges are e_i = (v_i, v_{i+1}).
#    
#    Constraints on the set of directed edges E_path:
#    1. For any two distinct edges e_i, e_j in E_path, they must be compatible.
#       Compatibility means they can appear in some order in a valid path.
#       Actually, it's easier to define "Incompatibility".
#       
#       Two directed edges a=(u,v) and b=(x,y) are incompatible if they cannot both be in the path.
#       Cases:
#       a) u == v or x == y (not possible in hypercube)
#       b) They share a vertex in a way that forces a repeat or chord.
#          - If v == x: This is a valid sequence v_i=u, v_{i+1}=v, v_{i+2}=y. 
#            This is allowed IF the path doesn't close a loop or create a chord.
#            However, in the MIS formulation, we are selecting a set of edges. 
#            We need to ensure the selected edges form a single path, not a collection of disjoint paths or cycles.
#            This is the hard part. MIS on the line graph gives a set of edges that form a vertex-disjoint collection of paths and cycles (a matching in the line graph? No, an independent set in the line graph is a matching in the original graph? No.)
#            
#            Let's re-evaluate.
#            An Independent Set in the Line Graph of G is a set of edges in G that are pairwise non-adjacent. 
#            This corresponds to a Matching in G.
#            A matching is a set of edges no two of which share a vertex.
#            This is NOT a path. A path shares vertices between consecutive edges.
#            
#            So MIS on the Line Graph is the wrong formulation for a Path.
#            A Path is a sequence of edges e_1, ..., e_k such that e_i and e_{i+1} share a vertex (head of e_i == tail of e_{i+1}), and no other pairs share vertices.
#            
#            This is equivalent to finding a Longest Path in the Directed Line Graph?
#            Or finding a Maximum Independent Set in a graph where nodes are directed edges, and edges connect "conflicting" pairs.
#            Which pairs conflict?
#            Pair (e1, e2) conflicts if they cannot both be in the path.
#            They CAN be in the path if:
#            1. They are disjoint (no shared vertices) -> They can be in different parts of the path? 
#               No, a path is connected. If e1 and e2 are disjoint, they can be in the path as long as there is a chain of edges connecting them.
#               But if we select a SET of edges, we don't know the order.
#               
#            This "Set of Edges" approach is tricky because of the connectivity requirement.
#            
#            Let's stick to the Path formulation but use a different search strategy.
#            The "Direct Line Graph + Max Flow" idea from the prompt is also flawed for simple paths because max flow finds edge-disjoint paths, not necessarily a single simple path, and vertex capacities don't prevent chords.
#            
#            Let's look at Inspiration 1: CSP with Backtracking and Forward Checking.
#            This is essentially DFS with pruning.
#            
#            How to make it faster?
#            1. Use Bitsets for "Used" and "Forbidden" (neighbors of used).
#            2. Use MRV (Most Constrained Variable) heuristic.
#               In the context of a path, we are at the current end. The "variable" is the next vertex.
#               We can look ahead.
#               
#            Alternative Strategy: **Simulated Annealing / Random Restart Local Search with Splicing**.
#            The current program does some of this.
#            
#            Let's try a **Genetic Algorithm** approach.
#            Chromosome: A valid snake (list of vertices).
#            Fitness: Length.
#            Crossover: Splice two snakes if possible.
#            Mutation: Reverse a segment? Or restart from a prefix?
#            
#            Actually, a very effective method for Snake-in-the-Box is **Randomized Greedy with Beam Search**.
#            Maintain a beam of top-K partial paths.
#            At each step, expand all paths in the beam by all valid neighbors.
#            Keep the top-K longest/valid paths.
#            
#            Why Beam Search?
#            Greedy gets stuck in local optima. Beam Search explores multiple branches.
#            K=100 or 200 might be feasible within 120s if optimized with C-extensions or NumPy?
#            Python loops are slow.
#            
#            Let's optimize the current Greedy/Beam approach with **NumPy** for vectorized validity checks if possible?
#            Validity check: New vertex v is valid if:
#            1. v not in used set.
#            2. v not in forbidden set (neighbors of used).
#            
#            We can maintain `used` and `forbidden` as bitsets (integers).
#            `used` is a bitset of vertices.
#            `forbidden` is a bitset of vertices.
#            
#            When adding v:
#            `used |= (1 << v)`
#            `forbidden |= (1 << v)`
#            `forbidden |= neighbors_bitset[v]`
#            
#            Check if v is valid:
#            `if (used | forbidden) & (1 << v): invalid`
#            
#            This is O(1) per check!
#            
#            Let's implement a **Beam Search** with Bitset Optimizations.
#            
#            State: (path, used_bitset, forbidden_bitset)
#            Actually, storing the full path in the beam is memory heavy if K is large and depth is large.
#            But depth is ~100-200. K=50. 50*200 = 10k integers. Fine.
#            
#            Algorithm:
#            1. Initialize beam with a few random starts (or just 0).
#            2. For each step:
#               - Generate candidates: For each path in beam, for each neighbor of last vertex, check validity.
#               - Collect all valid (new_path, new_used, new_forbidden).
#               - Sort by some heuristic? Or just keep the longest?
#                 Since all paths in a beam step have the same length (if we step synchronously), we just need to diversify.
#                 We can keep the top K based on "potential" (e.g., number of available next moves).
#            3. If beam is empty, stop.
#            
#            To maximize length, we should probably keep the paths that have the most "future potential".
#            Heuristic: Number of valid neighbors for the next step.
#            
#            Let's implement this.

def construct():
    deadline = time.time() + 115.0
    rnd = random.Random(42)
    
    # Precompute neighbor bitsets for fast forbidden update
    # neighbors_bitset[v] is an integer with bits set for all neighbors of v
    neighbors_bitset = [0] * SIZE
    for v in range(SIZE):
        mask = 0
        for b in range(N):
            mask |= (1 << (v ^ (1 << b)))
        neighbors_bitset[v] = mask

    # Beam Search
    # Beam: list of tuples (path, used_mask, forbidden_mask)
    # used_mask: bitset of vertices in path
    # forbidden_mask: bitset of vertices in path OR neighbors of vertices in path
    # Note: A vertex is invalid if it is in used_mask OR if it is in forbidden_mask?
    # Wait. 
    # Condition for new vertex v to be valid:
    # 1. v not in used (distinct vertices)
    # 2. v not adjacent to any u in path[:-1] (no chords)
    #    This is equivalent to: v not in neighbors(path[:-1])
    #    
    # Let's define `blocked` = used_mask | neighbors_of_all_used
    # If we add v, new_blocked = blocked | (1 << v) | neighbors_bitset[v]
    # Check if v is valid: 
    #    v is valid if (blocked & (1 << v)) == 0
    #    Wait, `blocked` contains `used`. So if v is in used, it's blocked.
    #    `blocked` contains `neighbors_of_all_used`. 
    #    If v is a neighbor of any used vertex, it is blocked.
    #    Is this correct?
    #    Yes. If v is a neighbor of ANY vertex in the path, it cannot be added, EXCEPT if it is the neighbor of the LAST vertex (which is the intended next step).
    #    
    #    So, for the next step from `cur`:
    #    Candidate v must be a neighbor of `cur`.
    #    v must NOT be in `used`.
    #    v must NOT be a neighbor of any `u` in `path[:-1]`.
    #    
    #    Let `blocked_except_cur_neighbors` = used_mask | neighbors_of(path[:-1])
    #    This is hard to maintain dynamically if we remove cur from the "neighbors of" set?
    #    
    #    Simpler:
    #    Maintain `used_mask` and `neighbor_mask` (neighbors of all used vertices).
    #    When adding v:
    #       new_used = used | (1<<v)
    #       new_neighbor_mask = neighbor_mask | neighbors_bitset[v]
    #       
    #    Check validity of v (where v is neighbor of cur):
    #       1. v not in used: `(used & (1<<v)) == 0`
    #       2. v not in neighbor_mask \ neighbors_bitset[cur]?
    #          No. v must not be neighbor of ANY u in path[:-1].
    #          `neighbor_mask` contains neighbors of ALL used vertices, including cur.
    #          So v is in `neighbor_mask` if it is neighbor of cur OR neighbor of others.
    #          Since v IS neighbor of cur, it WILL be in `neighbor_mask`.
    #          So we can't just check `neighbor_mask`.
    #          
    #    We need to check if v is neighbor of any u in path[:-1].
    #    This requires knowing the history.
    #    
    #    Optimization:
    #    Maintain `chord_blocker` = bitset of vertices that are neighbors of path[:-1].
    #    When we move from u to v:
    #       Remove u from the "active" set for chord blocking?
    #       No, u is still in the path.
    #       
    #    Let's just maintain `used` and `neighbor_mask`.
    #    To check if v is valid:
    #       v must not be in `used`.
    #       v must not be in `neighbor_mask` EXCEPT for the fact that it is a neighbor of `cur`.
    #       Actually, if v is a neighbor of `cur`, it is in `neighbor_mask`.
    #       We need to know if v is a neighbor of ANY OTHER vertex.
    #       
    #    This check `is_neighbor_of_other(v, path)` is O(N) if we iterate path.
    #    With Bitsets, can we do it faster?
    #    
    #    Let's try a different state representation.
    #    State: (cur, used_mask, chord_mask)
    #    `chord_mask` = bitset of all vertices that are neighbors of any vertex in path[:-1].
    #    When we append v to path (cur -> v):
    #       New path ends with v.
    #       New `chord_mask` for the NEXT step will be:
    #       (old_chord_mask | neighbors_bitset[cur]) 
    #       Wait.
    #       Old path: ..., u, cur.
    #       Old chord_mask: neighbors of ..., u. (cur is NOT in chord_mask because it's the end)
    #       New path: ..., u, cur, v.
    #       New chord_mask: neighbors of ..., u, cur.
    #       So new_chord_mask = old_chord_mask | neighbors_bitset[cur]
    #       
    #       Check if v is valid:
    #       1. v not in used.
    #       2. v not in old_chord_mask. (Because old_chord_mask contains neighbors of path[:-1])
    #       
    #    This is O(1) check!
    #    
    #    Update:
    #    new_used = used | (1 << v)
    #    new_chord_mask = chord_mask | neighbors_bitset[cur]
    #    new_cur = v
    
    # Let's implement Beam Search with this state.
    
    def run_beam_search(K=50, max_depth=200):
        # Start with a few random seeds
        # Or just start with 0
        # To increase diversity, start with multiple random vertices
        beam = []
        
        # Initialize with a few random starts to avoid symmetry traps?
        # The hypercube is symmetric, so start=0 is fine, but random restarts might help with beam diversity.
        # Let's start with a single path [0] for now, or multiple.
        # Multiple starts:
        for _ in range(min(10, K)):
            start = rnd.randint(0, SIZE - 1)
            used = 1 << start
            chord_mask = 0 # No previous vertices
            path = [start]
            beam.append((path, used, chord_mask, start))
        
        best_path = max(beam, key=lambda x: len(x[0]))[0]
        
        for depth in range(max_depth):
            if time.time() > deadline - 5:
                break
            
            next_beam = []
            
            for path, used, chord_mask, cur in beam:
                # Try all neighbors of cur
                for b in range(N):
                    v = cur ^ (1 << b)
                    bit_v = 1 << v
                    
                    # Check validity
                    if used & bit_v:
                        continue
                    if chord_mask & bit_v:
                        continue
                    
                    # Valid!
                    new_path = path + [v]
                    new_used = used | bit_v
                    # Update chord_mask: add neighbors of cur (the previous end)
                    # The new end is v. The vertices in path[:-1] are the same as before, PLUS cur.
                    # So new_chord_mask = old_chord_mask | neighbors_bitset[cur]
                    new_chord_mask = chord_mask | neighbors_bitset[cur]
                    
                    next_beam.append((new_path, new_used, new_chord_mask, v))
            
            if not next_beam:
                break
            
            # Select top K from next_beam
            # Heuristic: Prefer paths with more "options" for the next step?
            # Or just random sampling to maintain diversity?
            # Or sort by length (all same length in this step) and then random.
            # Actually, to find the LONGEST, we should keep the ones that are most "central" or have most exits.
            # Let's calculate the number of valid next moves for each candidate.
            
            scored_beam = []
            for item in next_beam:
                path, used, chord_mask, cur = item
                # Count valid neighbors for cur
                count = 0
                for b in range(N):
                    v = cur ^ (1 << b)
                    bit_v = 1 << v
                    if not (used & bit_v) and not (chord_mask & bit_v):
                        count += 1
                scored_beam.append((count, item))
            
            # Sort by count descending, then take top K
            # To ensure diversity, we can randomize the tie-breaking or sample from top 2K
            scored_beam.sort(key=lambda x: x[0], reverse=True)
            
            # Take top K, but ensure we don't just pick the same ones if counts are equal
            # Simple approach: Take top K
            top_k = scored_beam[:K]
            beam = [item for _, item in top_k]
            
            # Update best
            if beam:
                current_best = max(beam, key=lambda x: len(x[0]))[0]
                if len(current_best) > len(best_path):
                    best_path = current_best
        
        return best_path

    # Run Beam Search
    # We might want to run it multiple times with different K or random seeds if time permits
    best_path = run_beam_search(K=100)
    
    # Validate best_path
    def is_valid(path):
        if not path:
            return False
        n = len(path)
        if n == 1:
            return True
        # Check distinct
        if len(set(path)) != n:
            return False
        # Check consecutive
        for i in range(n - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return False
        # Check non-consecutive
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1:
                    if (path[i] ^ path[j]).bit_count() == 1:
                        return False
        return True

    if not is_valid(best_path):
        # Fallback to a simple greedy if beam search fails validation (shouldn't happen)
        best_path = [0]
        used = {0}
        cur = 0
        while True:
            found = False
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                valid = True
                for p in best_path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    best_path.append(v)
                    used.add(v)
                    cur = v
                    found = True
                    break
            if not found:
                break

    return best_path

if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p)-1}")
    # Validate
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
